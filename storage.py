"""
SQLite persistence: ticks (sampled), candles, signals, dashboard snapshot metadata,
and Telegram-approved demo trade execution records.

Writes are serialized with an asyncio lock so concurrent tasks don't corrupt SQLite.

Persistent storage:
- Locally: data/deriv_signals.db
- Railway Volume: /app/data/deriv_signals.db when DATA_DIR=/app/data
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from config import data_dir


DATA_DIR = data_dir()
DB_PATH = DATA_DIR / "deriv_signals.db"


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
                CREATE INDEX IF NOT EXISTS idx_ticks_symbol_epoch ON ticks (symbol, epoch);

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
                    created_epoch REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT,
                    metrics_json TEXT NOT NULL,
                    created_epoch REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trade_approvals (
                    approval_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    score REAL NOT NULL,
                    stake REAL NOT NULL,
                    currency TEXT NOT NULL,
                    duration INTEGER NOT NULL,
                    duration_unit TEXT NOT NULL,
                    signal_json TEXT NOT NULL,
                    created_epoch REAL NOT NULL,
                    updated_epoch REAL NOT NULL,
                    telegram_message_id TEXT,
                    note TEXT
                );

                CREATE TABLE IF NOT EXISTS trade_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    approval_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_epoch REAL NOT NULL
                );
                """
            )
            conn.commit()

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
                        ON CONFLICT(key) DO UPDATE SET value = excluded.value,
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
                        "SELECT value FROM meta WHERE key = ?", (key,)
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
                        "INSERT INTO ticks(symbol, epoch, price) VALUES (?,?,?)",
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
                        VALUES (?,?,?,?,?,?,?)
                        ON CONFLICT(symbol, timeframe, bucket_epoch) DO UPDATE SET
                            open = excluded.open,
                            high = excluded.high,
                            low = excluded.low,
                            close = excluded.close
                        """,
                        (symbol, timeframe, bucket_epoch, open_, high, low, close),
                    )
                    conn.commit()

            await asyncio.to_thread(_write)

    async def insert_signal_record(self, record: Mapping[str, Any]) -> None:
        payload = dict(record)
        created = float(payload.pop("created_epoch", time.time()))
        symbol = str(payload.pop("symbol"))
        side = str(payload.pop("side"))
        score = float(payload.pop("score"))
        timeframe = str(payload.pop("timeframe"))
        async with self._lock:

            def _write() -> None:
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO signals(symbol, side, score, timeframe, payload_json, created_epoch)
                        VALUES (?,?,?,?,?,?)
                        """,
                        (
                            symbol,
                            side,
                            score,
                            timeframe,
                            json.dumps(payload),
                            created,
                        ),
                    )
                    conn.commit()

            await asyncio.to_thread(_write)

    async def recent_signals(self, limit: int = 50) -> list[dict[str, Any]]:
        async with self._lock:

            def _read() -> list[dict[str, Any]]:
                with self._connect() as conn:
                    rows = conn.execute(
                        """
                        SELECT symbol, side, score, timeframe, payload_json, created_epoch
                        FROM signals
                        ORDER BY id DESC
                        LIMIT ?
                        """,
                        (limit,),
                    ).fetchall()
                    out: list[dict[str, Any]] = []
                    for r in rows:
                        base = dict(r)
                        extra = json.loads(base.pop("payload_json"))
                        base.update(extra)
                        out.append(base)
                    return out

            return await asyncio.to_thread(_read)

    async def create_trade_approval(self, record: Mapping[str, Any]) -> None:
        payload = dict(record)
        now = time.time()
        async with self._lock:

            def _write() -> None:
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO trade_approvals(
                            approval_id, status, symbol, side, score, stake, currency,
                            duration, duration_unit, signal_json, created_epoch,
                            updated_epoch, telegram_message_id, note
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            str(payload["approval_id"]),
                            str(payload.get("status", "pending")),
                            str(payload["symbol"]),
                            str(payload["side"]),
                            float(payload["score"]),
                            float(payload["stake"]),
                            str(payload["currency"]),
                            int(payload["duration"]),
                            str(payload["duration_unit"]),
                            json.dumps(payload["signal"]),
                            float(payload.get("created_epoch", now)),
                            float(payload.get("updated_epoch", now)),
                            payload.get("telegram_message_id"),
                            payload.get("note"),
                        ),
                    )
                    conn.commit()

            await asyncio.to_thread(_write)

    async def update_trade_approval(
        self,
        approval_id: str,
        status: str,
        note: str | None = None,
        telegram_message_id: str | None = None,
    ) -> None:
        async with self._lock:

            def _write() -> None:
                with self._connect() as conn:
                    if telegram_message_id is None:
                        conn.execute(
                            """
                            UPDATE trade_approvals
                            SET status = ?, note = ?, updated_epoch = ?
                            WHERE approval_id = ?
                            """,
                            (status, note, time.time(), approval_id),
                        )
                    else:
                        conn.execute(
                            """
                            UPDATE trade_approvals
                            SET status = ?, note = ?, telegram_message_id = ?, updated_epoch = ?
                            WHERE approval_id = ?
                            """,
                            (status, note, telegram_message_id, time.time(), approval_id),
                        )
                    conn.commit()

            await asyncio.to_thread(_write)

    async def get_trade_approval(self, approval_id: str) -> dict[str, Any] | None:
        async with self._lock:

            def _read() -> dict[str, Any] | None:
                with self._connect() as conn:
                    row = conn.execute(
                        "SELECT * FROM trade_approvals WHERE approval_id = ?",
                        (approval_id,),
                    ).fetchone()
                    if not row:
                        return None
                    data = dict(row)
                    data["signal"] = json.loads(data.pop("signal_json"))
                    return data

            return await asyncio.to_thread(_read)

    async def recent_trade_approvals(self, limit: int = 50) -> list[dict[str, Any]]:
        async with self._lock:

            def _read() -> list[dict[str, Any]]:
                with self._connect() as conn:
                    rows = conn.execute(
                        """
                        SELECT approval_id, status, symbol, side, score, stake, currency,
                               duration, duration_unit, created_epoch, updated_epoch, note
                        FROM trade_approvals
                        ORDER BY created_epoch DESC
                        LIMIT ?
                        """,
                        (limit,),
                    ).fetchall()
                    return [dict(r) for r in rows]

            return await asyncio.to_thread(_read)

    async def insert_trade_execution(self, record: Mapping[str, Any]) -> None:
        payload = dict(record)
        async with self._lock:

            def _write() -> None:
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO trade_executions(
                            approval_id, status, request_json, response_json, created_epoch
                        ) VALUES (?,?,?,?,?)
                        """,
                        (
                            str(payload["approval_id"]),
                            str(payload["status"]),
                            json.dumps(payload.get("request", {})),
                            json.dumps(payload.get("response", {})),
                            float(payload.get("created_epoch", time.time())),
                        ),
                    )
                    conn.commit()

            await asyncio.to_thread(_write)

    async def recent_trade_executions(self, limit: int = 50) -> list[dict[str, Any]]:
        async with self._lock:

            def _read() -> list[dict[str, Any]]:
                with self._connect() as conn:
                    rows = conn.execute(
                        """
                        SELECT approval_id, status, request_json, response_json, created_epoch
                        FROM trade_executions
                        ORDER BY id DESC
                        LIMIT ?
                        """,
                        (limit,),
                    ).fetchall()
                    out: list[dict[str, Any]] = []
                    for r in rows:
                        data = dict(r)
                        data["request"] = json.loads(data.pop("request_json"))
                        data["response"] = json.loads(data.pop("response_json"))
                        out.append(data)
                    return out

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
        df = pd.DataFrame([dict(r) for r in rows])
        df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
        df.set_index("datetime", inplace=True)
        return df[["open", "high", "low", "close"]]
