"""
SQLite persistence for ticks, candles, signals, dashboard metadata,
and signal outcome tracking.

Writes are serialized with an asyncio lock so concurrent tasks do not corrupt
SQLite. Railway persistence is supported through DATA_DIR=/app/data.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DATA_DIR = Path(
    os.getenv("RAILWAY_VOLUME_MOUNT_PATH", os.getenv("DATA_DIR", "data"))
)

DB_PATH = DATA_DIR / "deriv_signals.db"


OPEN_STATUS = "OPEN"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Storage:
    """Small helper around sqlite3 with schema bootstrap."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_epoch REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    epoch REAL NOT NULL,
                    price REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_ticks_symbol_epoch
                ON ticks (symbol, epoch);

                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    bucket_epoch REAL NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    PRIMARY KEY (symbol, timeframe, bucket_epoch)
                );

                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    score REAL NOT NULL,
                    timeframe TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_epoch REAL NOT NULL,
                    outcome_status TEXT NOT NULL DEFAULT 'OPEN',
                    outcome_epoch REAL,
                    outcome_price REAL,
                    outcome_reason TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_signals_symbol_status
                ON signals (symbol, outcome_status, created_epoch);

                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT,
                    metrics_json TEXT NOT NULL,
                    created_epoch REAL NOT NULL
                );
                """
            )
            self._migrate_signals_table(conn)
            conn.commit()

    def _migrate_signals_table(self, conn: sqlite3.Connection) -> None:
        """Add outcome columns when upgrading an existing Railway volume DB."""
        rows = conn.execute("PRAGMA table_info(signals)").fetchall()
        existing = {str(row["name"]) for row in rows}
        required = {
            "outcome_status": "TEXT NOT NULL DEFAULT 'OPEN'",
            "outcome_epoch": "REAL",
            "outcome_price": "REAL",
            "outcome_reason": "TEXT",
        }
        for col, definition in required.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE signals ADD COLUMN {col} {definition}")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_signals_symbol_status
            ON signals (symbol, outcome_status, created_epoch)
            """
        )

    async def set_meta(self, key: str, value: Mapping[str, Any]) -> None:
        payload = json.dumps(dict(value))
        epoch = time.time()

        async with self._lock:

            def _write() -> None:
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO meta(key, value, updated_epoch)
                        VALUES (?, ?, ?)
                        ON CONFLICT(key) DO UPDATE SET
                            value = excluded.value,
                            updated_epoch = excluded.updated_epoch
                        """,
                        (key, payload, epoch),
                    )
                    conn.commit()

            await asyncio.to_thread(_write)

    async def get_meta(self, key: str) -> dict[str, Any] | None:
        async with self._lock:

            def _read() -> str | None:
                with self._connect() as conn:
                    row = conn.execute(
                        "SELECT value FROM meta WHERE key = ?",
                        (key,),
                    ).fetchone()
                    return str(row["value"]) if row else None

            raw = await asyncio.to_thread(_read)

        if raw is None:
            return None
        return json.loads(raw)

    async def insert_tick(self, symbol: str, epoch: float, price: float) -> None:
        async with self._lock:

            def _write() -> None:
                with self._connect() as conn:
                    conn.execute(
                        "INSERT INTO ticks(symbol, epoch, price) VALUES (?, ?, ?)",
                        (symbol, epoch, price),
                    )
                    conn.commit()

            await asyncio.to_thread(_write)

    async def upsert_candle(
        self,
        symbol: str,
        timeframe: str,
        bucket_epoch: float,
        open_: float,
        high: float,
        low: float,
        close: float,
    ) -> None:
        async with self._lock:

            def _write() -> None:
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO candles(symbol, timeframe, bucket_epoch,
                            open, high, low, close)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(symbol, timeframe, bucket_epoch)
                        DO UPDATE SET
                            open = excluded.open,
                            high = excluded.high,
                            low = excluded.low,
                            close = excluded.close
                        """,
                        (symbol, timeframe, bucket_epoch, open_, high, low, close),
                    )
                    conn.commit()

            await asyncio.to_thread(_write)

    async def insert_signal_record(self, record: Mapping[str, Any]) -> int:
        payload = dict(record)

        created = float(payload.pop("created_epoch", time.time()))
        symbol = str(payload.pop("symbol"))
        side = str(payload.pop("side"))
        score = float(payload.pop("score"))
        timeframe = str(payload.pop("timeframe"))

        async with self._lock:

            def _write() -> int:
                with self._connect() as conn:
                    cur = conn.execute(
                        """
                        INSERT INTO signals(
                            symbol, side, score, timeframe,
                            payload_json, created_epoch, outcome_status
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            symbol,
                            side,
                            score,
                            timeframe,
                            json.dumps(payload),
                            created,
                            OPEN_STATUS,
                        ),
                    )
                    conn.commit()
                    return int(cur.lastrowid)

            return await asyncio.to_thread(_write)

    def _signal_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        base = dict(row)
        payload = base.pop("payload_json", "{}")
        try:
            extra = json.loads(payload)
        except json.JSONDecodeError:
            extra = {}
        base.update(extra)
        return base

    async def recent_signals(self, limit: int = 50) -> list[dict[str, Any]]:
        async with self._lock:

            def _read() -> list[dict[str, Any]]:
                with self._connect() as conn:
                    rows = conn.execute(
                        """
                        SELECT
                            id, symbol, side, score, timeframe,
                            payload_json, created_epoch,
                            outcome_status, outcome_epoch,
                            outcome_price, outcome_reason
                        FROM signals
                        ORDER BY id DESC
                        LIMIT ?
                        """,
                        (limit,),
                    ).fetchall()
                    return [self._signal_row_to_dict(row) for row in rows]

            return await asyncio.to_thread(_read)

    async def evaluate_open_signal_outcomes(
        self,
        symbol: str,
        candle_epoch: float,
        high: float,
        low: float,
        close: float,
        expiry_minutes: int = 180,
    ) -> list[dict[str, Any]]:
        """
        Check open signals for this symbol against the latest completed 1m candle.

        OHLC does not reveal exact intra-candle order. If TP and SL are both touched
        in the same candle, the tracker marks it conservatively as LOSS_SL_AMBIGUOUS.
        """
        expiry_seconds = max(1, expiry_minutes) * 60.0

        def _decide(row: sqlite3.Row) -> tuple[str | None, float | None, str | None]:
            data = self._signal_row_to_dict(row)
            side = str(data.get("side", "")).upper()
            created = float(data.get("created_epoch", candle_epoch))

            try:
                sl = float(data["stop_loss"])
                tp1 = float(data["take_profit_1"])
                tp2 = float(data["take_profit_2"])
            except (KeyError, TypeError, ValueError):
                return "EXPIRED", close, "Missing TP/SL data; cannot evaluate reliably"

            if side == "BUY":
                hit_sl = low <= sl
                hit_tp2 = high >= tp2
                hit_tp1 = high >= tp1
            elif side == "SELL":
                hit_sl = high >= sl
                hit_tp2 = low <= tp2
                hit_tp1 = low <= tp1
            else:
                return "EXPIRED", close, "Unknown signal side"

            if hit_sl and (hit_tp1 or hit_tp2):
                return "LOSS_SL_AMBIGUOUS", sl, "SL and TP were both inside the same 1m candle; marked conservative"
            if hit_tp2:
                return "WIN_TP2", tp2, "TP2 was reached before SL"
            if hit_tp1:
                return "WIN_TP1", tp1, "TP1 was reached before SL"
            if hit_sl:
                return "LOSS_SL", sl, "SL / invalidation was reached before TP"
            if candle_epoch - created >= expiry_seconds:
                return "EXPIRED", close, f"No TP/SL hit within {expiry_minutes} minutes"
            return None, None, None

        async with self._lock:

            def _work() -> list[dict[str, Any]]:
                events: list[dict[str, Any]] = []
                with self._connect() as conn:
                    rows = conn.execute(
                        """
                        SELECT
                            id, symbol, side, score, timeframe,
                            payload_json, created_epoch,
                            outcome_status, outcome_epoch,
                            outcome_price, outcome_reason
                        FROM signals
                        WHERE UPPER(symbol) = UPPER(?)
                          AND outcome_status = ?
                        ORDER BY id ASC
                        """,
                        (symbol, OPEN_STATUS),
                    ).fetchall()

                    for row in rows:
                        status, price, reason = _decide(row)
                        if status is None:
                            continue

                        conn.execute(
                            """
                            UPDATE signals
                            SET outcome_status = ?,
                                outcome_epoch = ?,
                                outcome_price = ?,
                                outcome_reason = ?
                            WHERE id = ?
                            """,
                            (status, candle_epoch, price, reason, row["id"]),
                        )
                        event = self._signal_row_to_dict(row)
                        event.update(
                            {
                                "outcome_status": status,
                                "outcome_epoch": candle_epoch,
                                "outcome_price": price,
                                "outcome_reason": reason,
                            }
                        )
                        events.append(event)

                    conn.commit()
                return events

            return await asyncio.to_thread(_work)

    async def outcome_stats(self, limit_days: int = 30) -> dict[str, Any]:
        cutoff = time.time() - max(1, limit_days) * 86400.0

        async with self._lock:

            def _read() -> dict[str, Any]:
                with self._connect() as conn:
                    rows = conn.execute(
                        """
                        SELECT symbol, side, outcome_status, COUNT(*) AS count
                        FROM signals
                        WHERE created_epoch >= ?
                        GROUP BY symbol, side, outcome_status
                        ORDER BY symbol ASC, side ASC, outcome_status ASC
                        """,
                        (cutoff,),
                    ).fetchall()

                by_symbol: dict[str, dict[str, Any]] = {}
                totals = {"signals": 0, "wins": 0, "losses": 0, "open": 0, "expired": 0}

                for row in rows:
                    symbol = str(row["symbol"])
                    status = str(row["outcome_status"])
                    count = int(row["count"])
                    rec = by_symbol.setdefault(
                        symbol,
                        {"signals": 0, "wins": 0, "losses": 0, "open": 0, "expired": 0},
                    )
                    rec["signals"] += count
                    totals["signals"] += count

                    if status.startswith("WIN"):
                        rec["wins"] += count
                        totals["wins"] += count
                    elif status.startswith("LOSS"):
                        rec["losses"] += count
                        totals["losses"] += count
                    elif status == OPEN_STATUS:
                        rec["open"] += count
                        totals["open"] += count
                    elif status == "EXPIRED":
                        rec["expired"] += count
                        totals["expired"] += count

                for rec in by_symbol.values():
                    closed = rec["wins"] + rec["losses"] + rec["expired"]
                    rec["win_rate_closed"] = rec["wins"] / max(1, closed)

                closed_total = totals["wins"] + totals["losses"] + totals["expired"]
                totals["win_rate_closed"] = totals["wins"] / max(1, closed_total)

                return {"totals": totals, "by_symbol": by_symbol}

            return await asyncio.to_thread(_read)

    async def load_candles_df_sync(self, symbol: str, timeframe: str):
        """Used by dashboard/backtest helper — synchronous pandas in caller thread."""
        import pandas as pd

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT bucket_epoch AS epoch, open, high, low, close
                FROM candles
                WHERE symbol = ? AND timeframe = ?
                ORDER BY bucket_epoch ASC
                """,
                (symbol, timeframe),
            ).fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(row) for row in rows])
        df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
        df.set_index("datetime", inplace=True)

        return df[["open", "high", "low", "close"]]
