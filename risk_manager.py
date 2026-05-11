"""
Risk management filters for alerting (and future execution).

Even in signal-only mode, we throttle spammy alerts and suppress messages
when conditions are unusually hostile (extreme volatility, post-spike chaos).
"""

from __future__ import annotations

import logging
import time
from collections import deque, defaultdict

import numpy as np
import pandas as pd

from strategy import Signal, SpikeContext

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, settings) -> None:
        self.settings = settings
        self._signal_times: dict[str, deque[float]] = defaultdict(deque)
        self._last_signal_epoch: dict[str, float] = {}
        self._last_loss_epoch: dict[str, float] = {}

    def _prune_hour(self, symbol: str) -> None:
        dq = self._signal_times[symbol]
        cutoff = time.time() - 3600.0
        while dq and dq[0] < cutoff:
            dq.popleft()

    def register_outcome_loss(self, symbol: str, now: float | None = None) -> None:
        """
        Hook for future execution / labelled backtests.

        When a signal would have lost, record the timestamp so we can cool down.
        """
        self._last_loss_epoch[symbol] = now or time.time()

    def allow_signal(
        self,
        symbol: str,
        signal: Signal,
        df_1m: pd.DataFrame,
        spread_points_estimate: float,
        spike_ctx: SpikeContext,
    ) -> tuple[bool, str]:
        now = time.time()
        stage = getattr(signal, "alert_stage", "SIGNAL")
        throttle_key = f"{symbol}:{stage}"
        self._prune_hour(throttle_key)

        if len(self._signal_times[throttle_key]) >= self.settings.max_signals_per_symbol_per_hour:
            return False, f"Hourly {stage} alert cap reached for symbol"

        gap = self.settings.min_minutes_between_signals_same_symbol * 60.0
        last = self._last_signal_epoch.get(throttle_key)
        if last is not None and (now - last) < gap:
            return False, f"Minimum minutes between {stage} alerts not elapsed"

        loss_cool = self._last_loss_epoch.get(symbol)
        if loss_cool is not None and (now - loss_cool) < 45 * 60:
            return False, "Cooldown window after a recorded loss scenario"

        if spread_points_estimate > self.settings.max_spread_points_estimate:
            return (
                False,
                f"Spread / slippage estimate too high ({spread_points_estimate:.2f} pts)",
            )

        if not df_1m.empty and "atr_14" in df_1m.columns:
            atr = float(df_1m["atr_14"].iloc[-1])
            history = df_1m["atr_14"].dropna().iloc[-120:]
            if len(history) > 30:
                med = float(np.median(history))
                ratio = atr / max(med, 1e-9)
                if ratio >= self.settings.extreme_atr_ratio_threshold:
                    return (
                        False,
                        f"Extreme ATR regime (ratio={ratio:.2f}) — skipping alert",
                    )

        if spike_ctx.last_spike_epoch is not None:
            dt = now - spike_ctx.last_spike_epoch
            if (
                dt < self.settings.cooldown_after_spike_seconds
                and spike_ctx.spike_strength > 0.75
            ):
                return False, "Too soon after a major spike — risk manager blocking alert"

        return True, "ok"

    def observe_signal_sent(self, symbol: str, when: float | None = None, stage: str = "SIGNAL") -> None:
        ts = when or time.time()
        key = f"{symbol}:{stage}"
        self._signal_times[key].append(ts)
        self._last_signal_epoch[key] = ts
