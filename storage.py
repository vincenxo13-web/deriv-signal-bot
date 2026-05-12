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
WATCH_ONLY_STATUS = "WATCH_ONLY"


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

                CREATE TABLE IF NOT EXISTS signal_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER NOT NULL UNIQUE,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    alert_stage TEXT NOT NULL,
                    score REAL NOT NULL,
                    features_json TEXT NOT NULL,
                    created_epoch REAL NOT NULL,
                    outcome_status TEXT NOT NULL DEFAULT 'OPEN',
                    outcome_epoch REAL,
                    outcome_price REAL,
                    outcome_reason TEXT,
                    FOREIGN KEY(signal_id) REFERENCES signals(id)
                );

                CREATE INDEX IF NOT EXISTS idx_signal_features_stage_outcome
                ON signal_features (alert_stage, outcome_status, created_epoch);


                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT,
                    metrics_json TEXT NOT NULL,
                    created_epoch REAL NOT NULL
                );
                """
            )
            self._migrate_signals_table(conn)
            self._migrate_signal_features_table(conn)
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
        self._migrate_prep_signals_to_watch_only(conn)

    def _migrate_prep_signals_to_watch_only(self, conn: sqlite3.Connection) -> None:
        """Do not score old PREP/watchlist alerts as trade outcomes.

        Earlier two-stage builds inserted PREP alerts as OPEN, so the outcome
        tracker could incorrectly mark watch-only alerts as wins/losses.
        """
        rows = conn.execute(
            """
            SELECT id, payload_json
            FROM signals
            WHERE outcome_status = ?
            """,
            (OPEN_STATUS,),
        ).fetchall()

        prep_ids: list[int] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except json.JSONDecodeError:
                continue
            stage = str(payload.get("alert_stage", payload.get("stage", ""))).upper()
            if stage == "PREP":
                prep_ids.append(int(row["id"]))

        for signal_id in prep_ids:
            conn.execute(
                "UPDATE signals SET outcome_status = ? WHERE id = ?",
                (WATCH_ONLY_STATUS, signal_id),
            )

    def _migrate_signal_features_table(self, conn: sqlite3.Connection) -> None:
        """Create/upgrade the ML feature table on existing Railway SQLite DBs.

        Older deployments may already have a database volume but not the
        signal_features table or its outcome columns. This migration is safe to
        run on every startup and does not delete old candle/signal data.
        """
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signal_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                alert_stage TEXT NOT NULL,
                score REAL NOT NULL,
                features_json TEXT NOT NULL,
                created_epoch REAL NOT NULL,
                outcome_status TEXT NOT NULL DEFAULT 'OPEN',
                outcome_epoch REAL,
                outcome_price REAL,
                outcome_reason TEXT
            )
            """
        )

        rows = conn.execute("PRAGMA table_info(signal_features)").fetchall()
        existing = {str(row["name"]) for row in rows}

        required = {
            "signal_id": "INTEGER",
            "symbol": "TEXT NOT NULL DEFAULT ''",
            "side": "TEXT NOT NULL DEFAULT ''",
            "alert_stage": "TEXT NOT NULL DEFAULT 'SIGNAL'",
            "score": "REAL NOT NULL DEFAULT 0",
            "features_json": "TEXT NOT NULL DEFAULT '{}'",
            "created_epoch": "REAL NOT NULL DEFAULT 0",
            "outcome_status": "TEXT NOT NULL DEFAULT 'OPEN'",
            "outcome_epoch": "REAL",
            "outcome_price": "REAL",
            "outcome_reason": "TEXT",
        }

        for col, definition in required.items():
            if col not in existing:
                conn.execute(
                    f"ALTER TABLE signal_features ADD COLUMN {col} {definition}"
                )

        # This unique index is required for INSERT ... ON CONFLICT(signal_id).
        # It is created separately so older tables can be upgraded safely.
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_signal_features_signal_id_unique
            ON signal_features (signal_id)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_signal_features_stage_outcome
            ON signal_features (alert_stage, outcome_status, created_epoch)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_signal_features_symbol_stage
            ON signal_features (symbol, alert_stage, created_epoch)
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


    async def load_recent_candles(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 300,
    ) -> list[dict[str, float]]:
        """
        Load recent saved candles from SQLite for warm-start after redeploy.

        Returns candles oldest-first so MarketStreamRouter can append them
        directly into each symbol runtime deque.
        """
        async with self._lock:

            def _read() -> list[dict[str, float]]:
                with self._connect() as conn:
                    rows = conn.execute(
                        """
                        SELECT bucket_epoch, open, high, low, close
                        FROM candles
                        WHERE UPPER(symbol) = UPPER(?)
                          AND timeframe = ?
                        ORDER BY bucket_epoch DESC
                        LIMIT ?
                        """,
                        (symbol, timeframe, limit),
                    ).fetchall()

                    candles = [dict(row) for row in rows]
                    candles.reverse()
                    return candles

            return await asyncio.to_thread(_read)

    async def insert_signal_record(self, record: Mapping[str, Any]) -> int:
        payload = dict(record)

        created = float(payload.pop("created_epoch", time.time()))
        symbol = str(payload.pop("symbol"))
        side = str(payload.pop("side"))
        score = float(payload.pop("score"))
        timeframe = str(payload.pop("timeframe"))
        alert_stage = str(payload.get("alert_stage", payload.get("stage", ""))).upper()
        initial_outcome_status = OPEN_STATUS if alert_stage == "TRIGGER" else WATCH_ONLY_STATUS

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
                            initial_outcome_status,
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

    async def insert_signal_features(
        self,
        signal_id: int,
        symbol: str,
        side: str,
        alert_stage: str,
        score: float,
        features: Mapping[str, Any],
        created_epoch: float,
        outcome_status: str | None = None,
    ) -> None:
        """Save an ML-ready feature snapshot for one signal."""
        stage = str(alert_stage or "SIGNAL").upper()
        status = outcome_status or (OPEN_STATUS if stage == "TRIGGER" else WATCH_ONLY_STATUS)
        payload = json.dumps(dict(features), sort_keys=True)

        async with self._lock:

            def _write() -> None:
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO signal_features(
                            signal_id, symbol, side, alert_stage, score,
                            features_json, created_epoch, outcome_status
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(signal_id) DO UPDATE SET
                            symbol = excluded.symbol,
                            side = excluded.side,
                            alert_stage = excluded.alert_stage,
                            score = excluded.score,
                            features_json = excluded.features_json,
                            created_epoch = excluded.created_epoch,
                            outcome_status = excluded.outcome_status
                        """,
                        (
                            int(signal_id),
                            str(symbol),
                            str(side).upper(),
                            stage,
                            float(score),
                            payload,
                            float(created_epoch),
                            status,
                        ),
                    )
                    conn.commit()

            await asyncio.to_thread(_write)

    async def update_signal_feature_outcome(
        self,
        signal_id: int,
        outcome_status: str,
        outcome_epoch: float | None,
        outcome_price: float | None,
        outcome_reason: str | None,
    ) -> None:
        """Mirror signal result into the ML feature table."""
        async with self._lock:

            def _write() -> None:
                with self._connect() as conn:
                    conn.execute(
                        """
                        UPDATE signal_features
                        SET outcome_status = ?,
                            outcome_epoch = ?,
                            outcome_price = ?,
                            outcome_reason = ?
                        WHERE signal_id = ?
                        """,
                        (
                            str(outcome_status),
                            outcome_epoch,
                            outcome_price,
                            outcome_reason,
                            int(signal_id),
                        ),
                    )
                    conn.commit()

            await asyncio.to_thread(_write)

    async def ml_dataset_stats(self, limit_days: int = 30) -> dict[str, Any]:
        """Return simple ML dataset counts for dashboard monitoring."""
        cutoff = time.time() - max(1, limit_days) * 86400.0

        async with self._lock:

            def _read() -> dict[str, Any]:
                with self._connect() as conn:
                    rows = conn.execute(
                        """
                        SELECT alert_stage, outcome_status, COUNT(*) AS count
                        FROM signal_features
                        WHERE created_epoch >= ?
                        GROUP BY alert_stage, outcome_status
                        """,
                        (cutoff,),
                    ).fetchall()

                stats: dict[str, Any] = {
                    "total": 0,
                    "trigger_total": 0,
                    "trigger_resolved": 0,
                    "trigger_wins": 0,
                    "trigger_losses": 0,
                    "trigger_open": 0,
                    "prep_watch_only": 0,
                    "by_stage_status": {},
                }

                for row in rows:
                    stage = str(row["alert_stage"] or "SIGNAL").upper()
                    status = str(row["outcome_status"] or "OPEN").upper()
                    count = int(row["count"])
                    stats["total"] += count
                    stats["by_stage_status"].setdefault(stage, {})[status] = count

                    if stage == "TRIGGER":
                        stats["trigger_total"] += count
                        if status == OPEN_STATUS:
                            stats["trigger_open"] += count
                        elif status.startswith("WIN"):
                            stats["trigger_wins"] += count
                            stats["trigger_resolved"] += count
                        elif status.startswith("LOSS") or status == "EXPIRED":
                            stats["trigger_losses"] += count
                            stats["trigger_resolved"] += count
                    elif status == WATCH_ONLY_STATUS:
                        stats["prep_watch_only"] += count

                resolved = max(1, int(stats["trigger_resolved"]))
                stats["trigger_win_rate"] = float(stats["trigger_wins"]) / resolved
                return stats

            return await asyncio.to_thread(_read)

    async def load_ml_training_rows(
        self,
        limit: int | None = None,
        include_expired: bool = True,
    ) -> list[dict[str, Any]]:
        """Load resolved TRIGGER feature rows for model training/export."""
        statuses = ["WIN_TP1", "WIN_TP2", "LOSS_SL", "LOSS_SL_AMBIGUOUS"]
        if include_expired:
            statuses.append("EXPIRED")
        placeholders = ",".join("?" for _ in statuses)
        limit_clause = "LIMIT ?" if limit else ""
        params: list[Any] = ["TRIGGER", *statuses]
        if limit:
            params.append(int(limit))

        async with self._lock:

            def _read() -> list[dict[str, Any]]:
                with self._connect() as conn:
                    rows = conn.execute(
                        f"""
                        SELECT
                            signal_id, symbol, side, alert_stage, score,
                            features_json, created_epoch, outcome_status,
                            outcome_epoch, outcome_price, outcome_reason
                        FROM signal_features
                        WHERE alert_stage = ?
                          AND outcome_status IN ({placeholders})
                        ORDER BY created_epoch ASC
                        {limit_clause}
                        """,
                        params,
                    ).fetchall()

                out: list[dict[str, Any]] = []
                for row in rows:
                    base = dict(row)
                    try:
                        features = json.loads(base.pop("features_json") or "{}")
                    except json.JSONDecodeError:
                        features = {}
                    success = 1 if str(base.get("outcome_status", "")).startswith("WIN") else 0
                    base.update(features)
                    base["success"] = success
                    out.append(base)
                return out

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
                        conn.execute(
                            """
                            UPDATE signal_features
                            SET outcome_status = ?,
                                outcome_epoch = ?,
                                outcome_price = ?,
                                outcome_reason = ?
                            WHERE signal_id = ?
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
                          AND outcome_status != 'WATCH_ONLY'
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
