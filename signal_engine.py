"""Glue between strategy scoring, risk manager, persistence, and Telegram alerts."""
from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from config import Settings, get_settings
from indicators import attach_core_indicators, classify_regime
from notifier import Notifier
from risk_manager import RiskManager
from storage import Storage
from strategy import SpikeContext, evaluate_signal, signal_to_storage_row

logger = logging.getLogger(__name__)


class SignalEngine:
    """Called whenever a new 1m candle finishes for a symbol."""

    def __init__(self, storage: Storage, risk_manager: RiskManager, notifier: Notifier, settings: Settings | None = None) -> None:
        self.storage = storage
        self.risk = risk_manager
        self.notifier = notifier
        self.settings = settings or get_settings()

    async def on_bar_closed(self, symbol: str, df_1m: pd.DataFrame, spike_ctx: SpikeContext) -> None:
        if df_1m.empty:
            return

        await self._track_outcomes(symbol, df_1m)
        df_feat = attach_core_indicators(df_1m)

        sig = evaluate_signal(
            symbol=symbol,
            df_1m=df_1m,
            spike_ctx=spike_ctx,
            min_score=self.settings.min_signal_score,
            now_epoch=time.time(),
            warmup_bars=getattr(self.settings, "signal_warmup_bars", 180),
            trigger_min_score=getattr(self.settings, "trigger_min_signal_score", 80.0),
            trigger_spike_strength=getattr(self.settings, "trigger_spike_strength", 0.8),
            trigger_tick_velocity_min=getattr(self.settings, "trigger_tick_velocity_min", 0.01),
            preparation_alerts_enabled=getattr(self.settings, "preparation_alerts_enabled", False),
            trigger_alerts_enabled=getattr(self.settings, "trigger_alerts_enabled", True),
            entry_zone_atr_multiplier=getattr(self.settings, "entry_zone_atr_multiplier", 0.08),
            stop_loss_atr_multiplier=getattr(self.settings, "stop_loss_atr_multiplier", 2.8),
            take_profit_1_atr_multiplier=getattr(self.settings, "take_profit_1_atr_multiplier", 3.5),
            take_profit_2_atr_multiplier=getattr(self.settings, "take_profit_2_atr_multiplier", 6.0),
            min_risk_reward=getattr(self.settings, "min_risk_reward", 1.2),
            require_trend_alignment=getattr(self.settings, "require_trend_alignment", True),
            require_regime_alignment=getattr(self.settings, "require_regime_alignment", True),
            allow_counter_regime_reversal=getattr(self.settings, "allow_counter_regime_reversal", False),
            regime_conflict_penalty=getattr(self.settings, "regime_conflict_penalty", 35.0),
            require_price_action_confirmation_in_high_vol=getattr(self.settings, "require_price_action_confirmation_in_high_vol", True),
            stoch_enabled=getattr(self.settings, "stoch_enabled", True),
            require_stoch_for_trigger=getattr(self.settings, "require_stoch_for_trigger", True),
            stoch_k_period=getattr(self.settings, "stoch_k_period", 14),
            stoch_d_period=getattr(self.settings, "stoch_d_period", 3),
            stoch_smoothing=getattr(self.settings, "stoch_smoothing", 3),
            stoch_oversold=getattr(self.settings, "stoch_oversold", 20.0),
            stoch_overbought=getattr(self.settings, "stoch_overbought", 80.0),
            require_micro_break_for_trigger=getattr(self.settings, "require_micro_break_for_trigger", True),
            micro_break_lookback=getattr(self.settings, "micro_break_lookback", 3),
            ict_bpr_enabled=getattr(self.settings, "ict_bpr_enabled", True),
            ict_bpr_lookback_candles=getattr(self.settings, "ict_bpr_lookback_candles", 120),
            ict_bpr_score_bonus=getattr(self.settings, "ict_bpr_score_bonus", 5.0),
            ict_bpr_require_for_trigger=getattr(self.settings, "ict_bpr_require_for_trigger", False),
            ict_bpr_max_distance_atr=getattr(self.settings, "ict_bpr_max_distance_atr", 2.0),
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
        signal_id = await self._insert_signal(row)
        self._observe_signal_sent(symbol, sig)
        await self._log_ml_features(signal_id, sig)
        await self.notifier.broadcast(sig)
        await self._publish_snapshot(symbol, df_feat, sig, spike_ctx)

    async def _insert_signal(self, row: dict[str, Any]) -> int | None:
        result = await self.storage.insert_signal_record(row)
        return int(result) if isinstance(result, int) else None

    def _observe_signal_sent(self, symbol: str, sig) -> None:
        try:
            self.risk.observe_signal_sent(symbol, stage=getattr(sig, "alert_stage", "TRIGGER"))
        except TypeError:
            self.risk.observe_signal_sent(symbol)

    async def _log_ml_features(self, signal_id: int | None, sig) -> None:
        if not getattr(self.settings, "ml_feature_logging_enabled", False):
            return
        if not hasattr(self.storage, "insert_signal_features"):
            return
        try:
            await self.storage.insert_signal_features(
                signal_id=signal_id,
                symbol=sig.symbol,
                side=sig.side,
                alert_stage=getattr(sig, "alert_stage", "TRIGGER"),
                score=float(sig.score),
                features=getattr(sig, "features", {}) or {},
                outcome_status="WATCH_ONLY" if getattr(sig, "alert_stage", "TRIGGER") == "PREP" else "OPEN",
                created_epoch=float(sig.timestamp_epoch),
            )
        except Exception:
            logger.exception("Failed to log ML features for %s", sig.symbol)

    async def _track_outcomes(self, symbol: str, df_1m: pd.DataFrame) -> None:
        if not getattr(self.settings, "outcome_tracking_enabled", True) or df_1m.empty:
            return
        if not hasattr(self.storage, "evaluate_open_signal_outcomes"):
            return
        last = df_1m.iloc[-1]
        candle_epoch = float(df_1m.index[-1].timestamp())
        events = await self.storage.evaluate_open_signal_outcomes(
            symbol=symbol,
            candle_epoch=candle_epoch,
            high=float(last["high"]),
            low=float(last["low"]),
            close=float(last["close"]),
            expiry_minutes=getattr(self.settings, "signal_expiry_minutes", 180),
        )
        if events:
            logger.info("Resolved %s signal outcome(s) for %s", len(events), symbol)
            if hasattr(self.storage, "update_signal_feature_outcomes"):
                try:
                    await self.storage.update_signal_feature_outcomes(events)
                except Exception:
                    logger.exception("Failed to update ML feature outcomes")
            if getattr(self.settings, "notify_signal_outcomes", False):
                if hasattr(self.notifier, "broadcast_outcomes"):
                    await self.notifier.broadcast_outcomes(events)
                else:
                    logger.warning("Outcome notification enabled but notifier.broadcast_outcomes is missing")

    async def _publish_snapshot(self, symbol: str, df_1m: pd.DataFrame, sig, spike_ctx: SpikeContext, extra_note: str | None = None) -> None:
        snap = await self.storage.get_meta("dashboard_snapshot") or {}
        if not isinstance(snap, dict):
            snap = {}
        row = df_1m.iloc[-1] if not df_1m.empty else None

        def _num(name: str):
            if row is None or name not in row:
                return None
            val = row[name]
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
            "spike_direction": spike_ctx.spike_direction,
            "spike_strength": spike_ctx.spike_strength,
            "risk_note": extra_note,
        }
        if sig is not None:
            sym_state["last_signal"] = {
                "side": sig.side,
                "stage": getattr(sig, "alert_stage", "TRIGGER"),
                "score": sig.score,
                "summary": "; ".join((sig.reasons or [])[:2]),
                "bpr": getattr(sig, "bpr_context", None),
            }
        snap[symbol] = sym_state
        await self.storage.set_meta("dashboard_snapshot", snap)
