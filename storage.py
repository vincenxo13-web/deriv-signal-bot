"""
SQLite persistence: ticks (sampled), candles, signals, dashboard snapshot metadata.

Writes are serialized with an asyncio lock so concurrent tasks don't corrupt SQLite.

This version supports persistent Railway storage:
- Locally: data/deriv_signals.db
- Railway with Volume: /app/data/deriv_signals.db
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


# Railway volume path support.
# On Railway, set DATA_DIR=/app/data and mount your Volume to /app/data.
# Locally, it will use the normal "data" folder.
DATA_DIR = Path(
    os.getenv("RAILWAY_VOLUME_MOUNT_PATH", os.getenv("DATA_DIR", "data"))
)

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
                    created_epoch REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT,
                    metrics_json TEXT NOT NULL,
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
                        """
                        INSERT INTO ticks(symbol, epoch, price)
                        VALUES (?, ?, ?)
                        """,
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
                        INSERT INTO candles(
                            symbol, timeframe, bucket_epoch,
                            open, high, low, close
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(symbol, timeframe, bucket_epoch) 
                        DO UPDATE SET
                            open = excluded.open,
                            high = excluded.high,
                            low = excluded.low,
                            close = excluded.close
                        """,
                        (
                            symbol,
                            timeframe,
                            bucket_epoch,
                            open_,
                            high,
                            low,
                            close,
                        ),
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
                        INSERT INTO signals(
                            symbol, side, score, timeframe, 
                            payload_json, created_epoch
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
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
                        SELECT 
                            symbol, side, score, timeframe, 
                            payload_json, created_epoch
                        FROM signals
                        ORDER BY id DESC
                        LIMIT ?
                        """,
                        (limit,),
                    ).fetchall()

                    out: list[dict[str, Any]] = []

                    for row in rows:
                        base = dict(row)
                        extra = json.loads(base.pop("payload_json"))
                        base.update(extra)
                        out.append(base)

                    return out

            return await asyncio.to_thread(_read)

    async def load_candles_df_sync(self, symbol: str, timeframe: str):
        """Used by dashboard/backtest helper — synchronous pandas in caller thread."""
        import pandas as pd

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT 
                    bucket_epoch AS epoch, 
                    open, high, low, close
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