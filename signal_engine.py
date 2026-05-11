"""
Glue between indicators + strategy scoring, risk manager, persistence, and alerts.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from config import Settings, execution_allowed, get_settings
from indicators import attach_core_indicators, classify_regime
from notifier import Notifier
from risk_manager import RiskManager
from storage import Storage
from strategy import SpikeContext, evaluate_signal, signal_to_storage_row
from trade_approval import TelegramApprovalTrader

logger = logging.getLogger(__name__)


class SignalEngine:
    """Called whenever a new 1m candle finishes for a symbol."""

    def __init__(
        self,
        storage: Storage,
        risk_manager: RiskManager,
        notifier: Notifier,
        settings: Settings | None = None,
        approval_trader: TelegramApprovalTrader | None = None,
    ) -> None:
        self.storage = storage
        self.risk = risk_manager
        self.notifier = notifier
        self.settings = settings or get_settings()
        self.approval_trader = approval_trader

    async def on_bar_closed(
        self,
        symbol: str,
        df_1m: pd.DataFrame,
        spike_ctx: SpikeContext,
    ) -> None:
        df_feat = attach_core_indicators(df_1m)
        sig = evaluate_signal(
            symbol=symbol,
            df_1m=df_1m,
            spike_ctx=spike_ctx,
            min_score=self.settings.min_signal_score,
            now_epoch=time.time(),
        )
        if sig is None:
            await self._publish_snapshot(symbol, df_feat, None, spike_ctx)
            return

        ok, reason = self.risk.allow_signal(
            symbol=symbol,
            signal=sig,
            df_1m=df_feat,
            spread_points_estimate=self.settings.estimated_spread_points,
            spike_ctx=spike_ctx,
        )
        if not ok:
            logger.info("Risk manager blocked %s signal: %s", symbol, reason)
            await self._publish_snapshot(symbol, df_feat, None, spike_ctx, extra_note=reason)
            return

        row = signal_to_storage_row(sig)
        await self.storage.insert_signal_record(row)
        self.risk.observe_signal_sent(symbol)

        await self.notifier.broadcast(sig)
        if self.approval_trader is not None:
            await self.approval_trader.submit_signal(sig)
        await self._publish_snapshot(symbol, df_feat, sig, spike_ctx)

        ok_exec, exec_reason = execution_allowed(self.settings)
        if self.settings.enable_real_trading:
            logger.warning(
                "ENABLE_REAL_TRADING is true — execution guard state: %s (%s). "
                "Telegram approval trade flow is enabled only when approval buttons are confirmed.",
                ok_exec,
                exec_reason,
            )

    async def _publish_snapshot(
        self,
        symbol: str,
        df_1m: pd.DataFrame,
        sig,
        spike_ctx: SpikeContext,
        extra_note: str | None = None,
    ) -> None:
        """Persist compact UI state for the Streamlit dashboard."""
        snap = await self.storage.get_meta("dashboard_snapshot") or {}
        if not isinstance(snap, dict):
            snap = {}

        row = df_1m.iloc[-1] if not df_1m.empty else None

        def _num(series_name: str):
            if row is None or series_name not in row:
                return None
            val = row[series_name]
            if pd.isna(val):
                return None
            return float(val)

        regime, regime_note = classify_regime(df_1m) if not df_1m.empty else ("n/a", None)

        sym_state: dict[str, Any] = {
            "price": _num("close"),
            "rsi": _num("rsi_14"),
            "macd_hist": _num("macd_hist"),
            "ema20": _num("ema_20"),
            "ema50": _num("ema_50"),
            "ema200": _num("ema_200"),
            "regime": regime,
            "regime_note": regime_note,
            "last_update": time.time(),
            "spike_velocity": spike_ctx.tick_velocity,
        }

        if sig is not None:
            sym_state["last_signal"] = {
                "side": sig.side,
                "score": sig.score,
                "timestamp": sig.timestamp_epoch,
                "summary": "; ".join(sig.reasons[:3]),
            }
        elif extra_note:
            sym_state["risk_note"] = extra_note

        snap[symbol] = sym_state
        await self.storage.set_meta("dashboard_snapshot", snap)
