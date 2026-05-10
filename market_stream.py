"""
Tick ingestion, candle aggregation, and spike detection for multiple symbols.

Feeds the signal engine whenever a fresh 1m bar completes.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Any

import numpy as np
import pandas as pd

from config import Settings, get_settings
from indicators import attach_core_indicators, last_bar_spike_metrics
from storage import Storage
from strategy import SpikeContext

logger = logging.getLogger(__name__)

OnBarClosed = Callable[[str, pd.DataFrame, SpikeContext], Awaitable[None]]


@dataclass
class CompletedBar:
    bucket_epoch: float
    open: float
    high: float
    low: float
    close: float


class MinuteBarAggregator:
    """Turn tick stream into 1-minute OHLC bars (bucket by UTC epoch minute)."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._bucket: int | None = None
        self._o = self._h = self._l = self._c = None

    def feed_tick(self, epoch: float, price: float) -> CompletedBar | None:
        minute_bucket = int(epoch) // 60 * 60
        if self._bucket is None:
            self._bucket = minute_bucket
            self._o = self._h = self._l = self._c = float(price)
            return None

        if minute_bucket != self._bucket:
            finished = CompletedBar(
                bucket_epoch=float(self._bucket),
                open=float(self._o),
                high=float(self._h),
                low=float(self._l),
                close=float(self._c),
            )
            self._bucket = minute_bucket
            self._o = self._h = self._l = self._c = float(price)
            return finished

        self._h = max(float(self._h), float(price))
        self._l = min(float(self._l), float(price))
        self._c = float(price)
        return None


class TickVelocityTracker:
    """Tick velocity proxy: absolute price change per second over recent ticks."""

    def __init__(self, maxlen: int = 40) -> None:
        self._dq: deque[tuple[float, float]] = deque(maxlen=maxlen)

    def push(self, epoch: float, price: float) -> None:
        self._dq.append((float(epoch), float(price)))

    def last_velocity(self) -> float:
        if len(self._dq) < 5:
            return 0.0
        first_t, first_p = self._dq[0]
        last_t, last_p = self._dq[-1]
        dt = max(last_t - first_t, 1e-6)
        return abs(last_p - first_p) / dt


class SymbolRuntime:
    """Per-symbol rolling state used by the streaming layer."""

    def __init__(self, symbol: str, settings: Settings) -> None:
        self.symbol = symbol
        self.settings = settings
        self.minute_agg = MinuteBarAggregator(symbol)
        self.velocity = TickVelocityTracker()
        self.bars_1m: deque[CompletedBar] = deque(maxlen=2500)
        self.last_spike_epoch: float | None = None
        self.last_spike_direction: str = "none"
        self.last_spike_strength: float = 0.0
        self.last_tick_epoch: float | None = None
        self.last_price: float | None = None
        self.last_tick_write_epoch: float = 0.0

    def is_boom(self) -> bool:
        return self.symbol.upper().startswith("BOOM")

    def build_spike_context(self) -> SpikeContext:
        return SpikeContext(
            last_spike_epoch=self.last_spike_epoch,
            spike_direction=self.last_spike_direction,  # type: ignore[arg-type]
            spike_strength=self.last_spike_strength,
            tick_velocity=self.velocity.last_velocity(),
        )

    def data_frame_1m(self) -> pd.DataFrame:
        rows = list(self.bars_1m)
        if not rows:
            return pd.DataFrame()
        data = {
            "epoch": [b.bucket_epoch for b in rows],
            "open": [b.open for b in rows],
            "high": [b.high for b in rows],
            "low": [b.low for b in rows],
            "close": [b.close for b in rows],
        }
        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
        df.set_index("datetime", inplace=True)
        return df[["open", "high", "low", "close"]]

    def maybe_flag_spike(self, df_feat: pd.DataFrame) -> None:
        boom = self.is_boom()
        metrics = last_bar_spike_metrics(df_feat)
        atr_mult = (
            self.settings.spike_body_atr_multiplier_boom
            if boom
            else self.settings.spike_body_atr_multiplier_crash
        )
        vel_thresh = (
            self.settings.spike_tick_velocity_threshold_boom
            if boom
            else self.settings.spike_tick_velocity_threshold_crash
        )

        body_flag = metrics["body_atr"] >= atr_mult
        vel = self.velocity.last_velocity()
        velocity_flag = vel >= vel_thresh

        direction = "none"
        if metrics["direction"] > 0:
            direction = "up"
        elif metrics["direction"] < 0:
            direction = "down"

        strength = float(min(2.0, metrics["body_atr"] / max(atr_mult, 1e-6)))
        if velocity_flag:
            strength = min(2.0, strength + 0.6)

        if body_flag or velocity_flag:
            self.last_spike_epoch = time.time()
            self.last_spike_direction = direction
            self.last_spike_strength = strength
            logger.info(
                "Spike flag %s dir=%s strength=%.2f body_atr=%.2f vel=%.6f",
                self.symbol,
                direction,
                strength,
                metrics["body_atr"],
                vel,
            )


class MarketStreamRouter:
    """
    Routes websocket JSON messages to per-symbol runtime state.

    Not all messages are ticks — errors and heartbeats are logged.
    """

    def __init__(
        self,
        storage: Storage,
        settings: Settings | None = None,
        on_bar_closed: OnBarClosed | None = None,
    ) -> None:
        self.storage = storage
        self.settings = settings or get_settings()
        self.on_bar_closed = on_bar_closed
        self.symbols: dict[str, SymbolRuntime] = {
            sym: SymbolRuntime(sym, self.settings) for sym in self.settings.symbols
        }

    def _resolve_symbol(self, tick: dict[str, Any]) -> str | None:
        raw = tick.get("symbol")
        if not raw:
            return None
        needle = str(raw).strip().upper()
        for name in self.symbols.keys():
            if name.upper() == needle:
                return name
        return None

    async def handle_deriv_message(self, message: dict[str, Any]) -> None:
        if message.get("error"):
            logger.error("Deriv API error: %s", message["error"])
            return

        if message.get("msg_type") != "tick":
            return

        tick = message.get("tick") or {}
        symbol = self._resolve_symbol(tick)
        if symbol is None:
            quote_sym = tick.get("symbol")
            logger.debug("Tick for unmanaged symbol=%s — ignoring", quote_sym)
            return

        try:
            price = float(tick["quote"])
            epoch = float(tick["epoch"])
        except (KeyError, TypeError, ValueError):
            logger.warning("Malformed tick payload: %s", tick)
            return

        await self.feed_tick(symbol, epoch, price)

    async def feed_tick(self, symbol: str, epoch: float, price: float) -> None:
        rt = self.symbols.get(symbol)
        if rt is None:
            return

        rt.last_price = price
        rt.last_tick_epoch = epoch
        rt.velocity.push(epoch, price)

        sample = self.settings.tick_sample_seconds
        now_wall = time.time()
        if sample == 0 or (now_wall - rt.last_tick_write_epoch) >= sample:
            rt.last_tick_write_epoch = now_wall
            await self.storage.insert_tick(symbol, epoch, price)

        completed = rt.minute_agg.feed_tick(epoch, price)
        if completed is None:
            return

        rt.bars_1m.append(completed)
        await self._persist_timeframes(symbol, rt)
        await self._maybe_emit(symbol, rt)

    async def _persist_timeframes(self, symbol: str, rt: SymbolRuntime) -> None:
        b = rt.bars_1m[-1]
        await self.storage.upsert_candle(
            symbol, "1m", b.bucket_epoch, b.open, b.high, b.low, b.close
        )

        df1 = rt.data_frame_1m()
        if df1.empty:
            return

        df5 = (
            df1.resample("5min", label="right", closed="right")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna(how="any")
        )
        df15 = (
            df1.resample("15min", label="right", closed="right")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna(how="any")
        )

        for tf_name, df_tf in ("5m", df5), ("15m", df15):
            if df_tf.empty:
                continue
            last = df_tf.iloc[-1]
            bucket = df_tf.index[-1].timestamp()
            await self.storage.upsert_candle(
                symbol,
                tf_name,
                bucket,
                float(last["open"]),
                float(last["high"]),
                float(last["low"]),
                float(last["close"]),
            )

    async def _maybe_emit(self, symbol: str, rt: SymbolRuntime) -> None:
        df1 = rt.data_frame_1m()
        if len(df1) < 220:
            return

        # Spike detection needs the latest bar measured vs ATR — indicators attached here only
        # for that local check. The strategy re-computes indicators on clean OHLC input.
        df_feat = attach_core_indicators(df1)
        rt.maybe_flag_spike(df_feat)

        ctx = rt.build_spike_context()
        if self.on_bar_closed is not None:
            await self.on_bar_closed(symbol, df1, ctx)
