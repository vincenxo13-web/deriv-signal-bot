"""
Boom/Crash signal-only strategy.

Core direction rules:
- BOOM symbols generate BUY triggers only.
- CRASH symbols generate SELL triggers only.

The current trigger model prioritises:
1. Correct symbol direction.
2. Correct broad regime, or a very strong reversal confirmation.
3. Micro-break / rejection / spike pressure confirmation.
4. Stochastic timing as a filter, not as a standalone entry.

This is research / decision-support code only, not financial advice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal
import logging

import numpy as np
import pandas as pd

try:
    from ict_bpr import bpr_context_for_signal
except Exception:  # optional helper; strategy still works without it
    bpr_context_for_signal = None

from indicators import (
    attach_core_indicators,
    bollinger_bandwidth,
    candle_rejection_score,
    classify_regime,
    detect_sr_zones,
)

logger = logging.getLogger(__name__)

Side = Literal["BUY", "SELL"]
AlertStage = Literal["PREP", "TRIGGER"]


@dataclass
class SpikeContext:
    last_spike_epoch: float | None
    spike_direction: str  # "up" | "down" | "none"
    spike_strength: float
    tick_velocity: float


@dataclass
class Signal:
    symbol: str
    side: Side
    score: float
    timeframe: str
    entry_zone_low: float
    entry_zone_high: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward: float
    reasons: list[str]
    volatility_warning: str | None
    regime: str
    timestamp_epoch: float
    alert_stage: AlertStage = "TRIGGER"
    bpr_context: dict[str, Any] | None = None
    features: dict[str, Any] | None = None


def signal_to_storage_row(sig: Signal) -> dict[str, Any]:
    data = asdict(sig)
    ts = float(data.pop("timestamp_epoch"))
    tf = str(data.pop("timeframe"))
    return {
        "symbol": str(data.pop("symbol")),
        "side": str(data.pop("side")),
        "score": float(data.pop("score")),
        "timeframe": tf,
        "created_epoch": ts,
        **data,
    }


def _is_boom(symbol: str) -> bool:
    return str(symbol).upper().startswith("BOOM")


def _is_crash(symbol: str) -> bool:
    return str(symbol).upper().startswith("CRASH")


def allowed_side_for_symbol(symbol: str) -> Side | None:
    if _is_boom(symbol):
        return "BUY"
    if _is_crash(symbol):
        return "SELL"
    return None


def _target_spike_direction(side: Side) -> str:
    return "up" if side == "BUY" else "down"


def _resample_ohlc(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df_1m[["open", "high", "low", "close"]]
        .resample(rule, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


def build_multi_timeframe(df_1m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df_1m.empty:
        return {"1m": df_1m, "5m": df_1m, "15m": df_1m}
    ohlc = df_1m[["open", "high", "low", "close"]].copy()
    return {"1m": ohlc, "5m": _resample_ohlc(ohlc, "5min"), "15m": _resample_ohlc(ohlc, "15min")}


def _attach_all(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        out[tf] = attach_core_indicators(df) if len(df) >= 10 else df
    return out


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _regime_conflicts(side: Side, regime: str) -> bool:
    r = str(regime or "").lower()
    if side == "BUY":
        return "downtrend" in r
    return "uptrend" in r


def _is_high_vol(regime: str, note: str | None = None) -> bool:
    r = str(regime or "").lower()
    n = str(note or "").lower()
    return "high_volatility" in r or "high volatility" in n or "elevated" in n


def _rsi_score(side: Side, rsi: float) -> tuple[float, str | None]:
    if not np.isfinite(rsi):
        return 0.0, None
    if side == "BUY":
        if 38 <= rsi <= 58:
            return 12.0, "RSI in Boom buy-preparation zone"
        if 30 <= rsi < 38:
            return 8.0, "RSI low; possible Boom spring zone"
        if 58 < rsi <= 66:
            return 4.0, "RSI positive but needs confirmation"
    else:
        if 42 <= rsi <= 62:
            return 12.0, "RSI in Crash sell-preparation zone"
        if 62 < rsi <= 72:
            return 8.0, "RSI high; possible Crash exhaustion zone"
        if 34 <= rsi < 42:
            return 4.0, "RSI weak but needs confirmation"
    return 0.0, None


def _stochastic_values(df: pd.DataFrame, k_period: int, d_period: int, smoothing: int) -> tuple[float, float, float, float]:
    needed = max(k_period + d_period + smoothing + 2, 20)
    if len(df) < needed:
        return float("nan"), float("nan"), float("nan"), float("nan")
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    raw_k = 100.0 * (df["close"] - low_min) / denom
    k = raw_k.rolling(max(1, smoothing)).mean()
    d = k.rolling(max(1, d_period)).mean()
    return float(k.iloc[-1]), float(d.iloc[-1]), float(k.iloc[-2]), float(d.iloc[-2])


def _stoch_timing_score(
    df: pd.DataFrame,
    side: Side,
    enabled: bool,
    k_period: int,
    d_period: int,
    smoothing: int,
    oversold: float,
    overbought: float,
) -> tuple[float, str | None, bool, dict[str, float]]:
    if not enabled:
        return 0.0, None, True, {}
    k, d, prev_k, prev_d = _stochastic_values(df, k_period, d_period, smoothing)
    values = {"stoch_k": k, "stoch_d": d, "stoch_prev_k": prev_k, "stoch_prev_d": prev_d}
    if not np.isfinite(k) or not np.isfinite(d):
        return 0.0, "Stoch not ready", False, values
    if side == "BUY":
        oversold_or_recovery = k <= oversold + 10 or (prev_k < oversold and k > prev_k) or (prev_k <= prev_d and k > d)
        if oversold_or_recovery:
            return 14.0, f"Stoch bullish timing ok (K={k:.1f}, D={d:.1f})", True, values
        return 0.0, f"Stoch BUY rejected: K={k:.1f}, D={d:.1f}; need oversold/recovery", False, values
    overbought_or_rejection = k >= overbought - 10 or (prev_k > overbought and k < prev_k) or (prev_k >= prev_d and k < d)
    if overbought_or_rejection:
        return 14.0, f"Stoch bearish timing ok (K={k:.1f}, D={d:.1f})", True, values
    return 0.0, f"Stoch SELL rejected: K={k:.1f}, D={d:.1f}; need overbought/rejection", False, values


def _trend_score(df: pd.DataFrame, side: Side) -> tuple[float, list[str], bool]:
    row = df.iloc[-1]
    close = _safe_float(row.get("close"))
    ema20 = _safe_float(row.get("ema_20"))
    ema50 = _safe_float(row.get("ema_50"))
    ema200 = _safe_float(row.get("ema_200"))
    pts = 0.0
    notes: list[str] = []
    if side == "BUY":
        aligned = (np.isfinite(ema20) and np.isfinite(ema50) and ema20 >= ema50) or (
            np.isfinite(ema200) and close >= ema200
        )
        if np.isfinite(ema200) and close >= ema200:
            pts += 12.0
            notes.append("Price above/near EMA200")
        if np.isfinite(ema20) and np.isfinite(ema50) and ema20 >= ema50:
            pts += 12.0
            notes.append("EMA20 >= EMA50")
        elif np.isfinite(ema50) and close >= ema50:
            pts += 6.0
            notes.append("Pullback holding near EMA50")
    else:
        aligned = (np.isfinite(ema20) and np.isfinite(ema50) and ema20 <= ema50) or (
            np.isfinite(ema200) and close <= ema200
        )
        if np.isfinite(ema200) and close <= ema200:
            pts += 12.0
            notes.append("Price below/near EMA200")
        if np.isfinite(ema20) and np.isfinite(ema50) and ema20 <= ema50:
            pts += 12.0
            notes.append("EMA20 <= EMA50")
        elif np.isfinite(ema50) and close <= ema50:
            pts += 6.0
            notes.append("Pull-up failing near EMA50")
    return pts, notes, bool(aligned)


def _higher_tf_score(dfs: dict[str, pd.DataFrame], side: Side) -> tuple[float, str | None]:
    pts = 0.0
    notes: list[str] = []
    for tf in ("5m", "15m"):
        df = dfs.get(tf)
        if df is None or len(df) < 5 or not {"ema_20", "ema_50", "close"}.issubset(df.columns):
            continue
        row = df.iloc[-1]
        ema20 = _safe_float(row.get("ema_20"))
        ema50 = _safe_float(row.get("ema_50"))
        close = _safe_float(row.get("close"))
        if side == "BUY" and np.isfinite(ema50) and close >= ema50:
            pts += 4.0
            notes.append(f"{tf} holding above EMA50")
        if side == "BUY" and np.isfinite(ema20) and np.isfinite(ema50) and ema20 >= ema50:
            pts += 3.0
            notes.append(f"{tf} EMA20>=EMA50")
        if side == "SELL" and np.isfinite(ema50) and close <= ema50:
            pts += 4.0
            notes.append(f"{tf} failing below EMA50")
        if side == "SELL" and np.isfinite(ema20) and np.isfinite(ema50) and ema20 <= ema50:
            pts += 3.0
            notes.append(f"{tf} EMA20<=EMA50")
    if not notes:
        return 0.0, None
    return min(12.0, pts), "Higher TF context: " + ", ".join(notes[:3])


def _bb_score(df: pd.DataFrame, side: Side) -> tuple[float, str | None]:
    if not {"bb_lower", "bb_upper", "bb_mid"}.issubset(df.columns):
        return 0.0, None
    bw = bollinger_bandwidth(df).dropna()
    if len(bw) < 30:
        return 0.0, None
    close = float(df["close"].iloc[-1])
    lower = float(df["bb_lower"].iloc[-1])
    upper = float(df["bb_upper"].iloc[-1])
    mid = float(df["bb_mid"].iloc[-1])
    last_w = float(bw.iloc[-1])
    med_w = float(bw.iloc[-30:].median())
    expanding = last_w > med_w * 1.05
    if side == "BUY":
        if close <= lower + (mid - lower) * 0.45:
            return 7.0, "Near lower Bollinger area"
        if close > mid and expanding:
            return 5.0, "Mid-BB reclaimed with expansion"
    else:
        if close >= upper - (upper - mid) * 0.45:
            return 7.0, "Near upper Bollinger area"
        if close < mid and expanding:
            return 5.0, "Mid-BB lost with expansion"
    return 0.0, None


def _support_resistance_score(side: Side, price: float, zones: dict[str, float | None], atr: float) -> tuple[float, str | None]:
    tol = max(atr * 0.45, 1e-9)
    if side == "BUY" and zones.get("support") is not None:
        sup = float(zones["support"])
        if abs(price - sup) <= tol * 2.5:
            return 10.0, "Support test before Boom spike"
        if price > sup and (price - sup) <= tol * 5:
            return 6.0, "Price holding near support"
    if side == "SELL" and zones.get("resistance") is not None:
        res = float(zones["resistance"])
        if abs(res - price) <= tol * 2.5:
            return 10.0, "Resistance test before Crash drop"
        if price < res and (res - price) <= tol * 5:
            return 6.0, "Price failing near resistance"
    return 0.0, None


def _micro_break_confirmed(df: pd.DataFrame, side: Side, lookback: int = 3) -> tuple[bool, str]:
    lookback = max(2, int(lookback))
    if len(df) < lookback + 2:
        return False, "Not enough candles for micro-break"
    recent = df.iloc[-lookback - 1 : -1]
    last_close = float(df["close"].iloc[-1])
    if side == "BUY":
        micro_high = float(recent["high"].max())
        if last_close > micro_high:
            return True, "Trigger confirmed by micro-break in target direction"
        return False, f"No bullish micro-break: close <= recent {lookback}-bar high"
    micro_low = float(recent["low"].min())
    if last_close < micro_low:
        return True, "Trigger confirmed by micro-break in target direction"
    return False, f"No bearish micro-break: close >= recent {lookback}-bar low"


def _wick_rejection_score(df: pd.DataFrame, side: Side, atr: float, lookback: int = 8) -> tuple[float, str | None]:
    if len(df) < lookback + 2:
        return 0.0, None
    last = df.iloc[-1]
    recent = df.iloc[-lookback - 1 : -1]
    high = float(last["high"])
    low = float(last["low"])
    close = float(last["close"])
    open_ = float(last["open"])
    rng = max(high - low, 1e-9)
    body = abs(close - open_)
    if side == "BUY":
        lower_wick = min(open_, close) - low
        swept_low = low < float(recent["low"].min())
        closed_off_low = (close - low) / rng >= 0.55
        if swept_low and lower_wick >= body * 1.2 and lower_wick >= atr * 0.25 and closed_off_low:
            return 12.0, "Bullish rejection: swept low and closed back up"
        if lower_wick >= body * 1.5 and closed_off_low:
            return 7.0, "Bullish lower-wick rejection"
    else:
        upper_wick = high - max(open_, close)
        swept_high = high > float(recent["high"].max())
        closed_off_high = (high - close) / rng >= 0.55
        if swept_high and upper_wick >= body * 1.2 and upper_wick >= atr * 0.25 and closed_off_high:
            return 12.0, "Bearish rejection: swept high and closed back down"
        if upper_wick >= body * 1.5 and closed_off_high:
            return 7.0, "Bearish upper-wick rejection"
    return 0.0, None


def _spike_pressure_score(
    spike_ctx: SpikeContext,
    side: Side,
    trigger_spike_strength: float,
    trigger_tick_velocity_min: float,
) -> tuple[float, str | None, bool]:
    direction = str(spike_ctx.spike_direction or "none").lower()
    target = _target_spike_direction(side)
    strength = max(0.0, float(spike_ctx.spike_strength or 0.0))
    velocity = max(0.0, abs(float(spike_ctx.tick_velocity or 0.0)))
    if direction == target and strength >= trigger_spike_strength and velocity >= trigger_tick_velocity_min:
        return 18.0, f"Target-direction spike pressure active ({target}, strength {strength:.2f})", True
    if direction == target and (strength >= trigger_spike_strength * 0.75 or velocity >= trigger_tick_velocity_min):
        return 8.0, f"Target-direction spike pressure building ({target})", False
    return 0.0, None, False


def _anti_chase_penalty(df: pd.DataFrame, atr: float) -> tuple[float, str | None]:
    if len(df) < 3 or atr <= 0:
        return 0.0, None
    last = df.iloc[-1]
    body = abs(float(last["close"]) - float(last["open"]))
    rng = float(last["high"]) - float(last["low"])
    if body >= atr * 2.2 or rng >= atr * 3.0:
        return -12.0, "Large candle already expanded; avoid chasing late entry"
    return 0.0, None


def _build_levels(
    side: Side,
    last_close: float,
    atr: float,
    entry_zone_atr_multiplier: float,
    stop_loss_atr_multiplier: float,
    take_profit_1_atr_multiplier: float,
    take_profit_2_atr_multiplier: float,
) -> tuple[float, float, float, float, float, float]:
    atr = max(float(atr), 1e-9)
    entry_mult = max(0.01, float(entry_zone_atr_multiplier))
    sl_mult = max(0.5, float(stop_loss_atr_multiplier))
    tp1_mult = max(0.5, float(take_profit_1_atr_multiplier))
    tp2_mult = max(tp1_mult, float(take_profit_2_atr_multiplier))
    entry_low = last_close - atr * entry_mult
    entry_high = last_close + atr * entry_mult
    if side == "BUY":
        stop_loss = entry_low - atr * sl_mult
        tp1 = entry_high + atr * tp1_mult
        tp2 = entry_high + atr * tp2_mult
        risk = entry_low - stop_loss
        reward = tp1 - entry_low
    else:
        stop_loss = entry_high + atr * sl_mult
        tp1 = entry_low - atr * tp1_mult
        tp2 = entry_low - atr * tp2_mult
        risk = stop_loss - entry_high
        reward = entry_high - tp1
    rr = float(max(0.0, reward / max(risk, 1e-9)))
    return float(entry_low), float(entry_high), float(stop_loss), float(tp1), float(tp2), rr


def _bpr_context(symbol: str, df_1m: pd.DataFrame, side: Side, atr: float, enabled: bool, lookback: int, max_dist_atr: float) -> dict[str, Any]:
    if not enabled or bpr_context_for_signal is None:
        return {"status": "n/a", "aligned": False, "note": "BPR disabled or helper unavailable"}
    try:
        ctx = bpr_context_for_signal(symbol=symbol, df_1m=df_1m, side=side, atr=atr, enabled=True, lookback_candles=lookback, max_distance_atr=max_dist_atr)
        return ctx.to_dict()
    except Exception as exc:
        logger.debug("BPR context failed for %s: %s", symbol, exc)
        return {"status": "NO_DATA", "aligned": False, "note": "H4 BPR unavailable"}


def evaluate_signal(
    symbol: str,
    df_1m: pd.DataFrame,
    spike_ctx: SpikeContext,
    min_score: float,
    now_epoch: float,
    *,
    signal_warmup_bars: int = 180,
    warmup_bars: int | None = None,
    preparation_alerts_enabled: bool = False,
    trigger_alerts_enabled: bool = True,
    trigger_min_signal_score: float = 80.0,
    trigger_min_score: float | None = None,
    trigger_spike_strength: float = 0.8,
    trigger_tick_velocity_min: float = 0.01,
    entry_zone_atr_multiplier: float = 0.08,
    stop_loss_atr_multiplier: float = 2.8,
    take_profit_1_atr_multiplier: float = 3.5,
    take_profit_2_atr_multiplier: float = 6.0,
    min_risk_reward: float = 1.2,
    require_micro_break_for_trigger: bool = True,
    micro_break_lookback: int = 3,
    require_trend_alignment: bool = True,
    require_regime_alignment: bool = True,
    allow_counter_regime_reversal: bool = False,
    regime_conflict_penalty: float = 35.0,
    require_price_action_confirmation_in_high_vol: bool = True,
    stoch_enabled: bool = True,
    require_stoch_for_trigger: bool = True,
    stoch_k_period: int = 14,
    stoch_d_period: int = 3,
    stoch_smoothing: int = 3,
    stoch_oversold: float = 20.0,
    stoch_overbought: float = 80.0,
    ict_bpr_enabled: bool = True,
    ict_bpr_lookback_candles: int = 120,
    ict_bpr_score_bonus: float = 5.0,
    ict_bpr_require_for_trigger: bool = False,
    ict_bpr_max_distance_atr: float = 2.0,
    **_: Any,
) -> Signal | None:
    side = allowed_side_for_symbol(symbol)
    if side is None:
        return None

    trigger_min = float(trigger_min_score if trigger_min_score is not None else trigger_min_signal_score)
    warmup = max(60, int(warmup_bars if warmup_bars is not None else signal_warmup_bars))
    if df_1m.empty or len(df_1m) < warmup:
        return None

    dfs = _attach_all(build_multi_timeframe(df_1m))
    df1 = dfs["1m"]
    if df1.empty or len(df1) < warmup:
        return None

    required_cols = {"ema_20", "ema_50", "ema_200", "rsi_14", "atr_14"}
    if not required_cols.issubset(df1.columns):
        return None

    row = df1.iloc[-1]
    last_close = float(row["close"])
    atr = max(_safe_float(row.get("atr_14"), 0.0), 1e-9)
    rsi = _safe_float(row.get("rsi_14"))
    regime, vol_note = classify_regime(df1)
    zones = detect_sr_zones(df1)
    reasons: list[str] = []
    score = 0.0

    trend_pts, trend_notes, trend_ok = _trend_score(df1, side)
    score += trend_pts
    reasons.extend(trend_notes)

    rsi_pts, rsi_note = _rsi_score(side, rsi)
    score += rsi_pts
    if rsi_note:
        reasons.append(rsi_note)

    stoch_pts, stoch_note, stoch_ok, stoch_values = _stoch_timing_score(
        df1, side, stoch_enabled, int(stoch_k_period), int(stoch_d_period), int(stoch_smoothing), float(stoch_oversold), float(stoch_overbought)
    )
    score += stoch_pts
    if stoch_note:
        reasons.append(stoch_note)

    sr_pts, sr_note = _support_resistance_score(side, last_close, zones, atr)
    score += sr_pts
    if sr_note:
        reasons.append(sr_note)

    bb_pts, bb_note = _bb_score(df1, side)
    score += bb_pts
    if bb_note:
        reasons.append(bb_note)

    wick_pts, wick_note = _wick_rejection_score(df1, side, atr)
    score += wick_pts
    if wick_note:
        reasons.append(wick_note)

    candle_rej_pts, candle_rej_note = candle_rejection_score(side, df1)
    score += candle_rej_pts
    if candle_rej_note:
        reasons.append(candle_rej_note)

    htf_pts, htf_note = _higher_tf_score(dfs, side)
    score += htf_pts
    if htf_note:
        reasons.append(htf_note)

    spike_pts, spike_note, spike_pressure_confirmed = _spike_pressure_score(spike_ctx, side, trigger_spike_strength, trigger_tick_velocity_min)
    score += spike_pts
    if spike_note:
        reasons.append(spike_note)

    micro_ok, micro_note = _micro_break_confirmed(df1, side, micro_break_lookback)
    if micro_ok:
        score += 18.0
        reasons.append("Trigger confirmed by micro-break in target direction")
    else:
        reasons.append(micro_note)

    chase_pen, chase_note = _anti_chase_penalty(df1, atr)
    score += chase_pen
    if chase_note:
        reasons.append(chase_note)

    bpr_ctx = _bpr_context(symbol, df1, side, atr, ict_bpr_enabled, int(ict_bpr_lookback_candles), float(ict_bpr_max_distance_atr))
    if bool(bpr_ctx.get("aligned")):
        score += float(ict_bpr_score_bonus)
        reasons.append(f"H4 BPR aligned: {bpr_ctx.get('note', 'aligned')}")

    strong_rejection = float(wick_pts or 0.0) >= 10.0 or float(candle_rej_pts or 0.0) >= 10.0
    strong_price_action = bool(micro_ok or strong_rejection)
    hard_confirmation = bool(spike_pressure_confirmed and strong_price_action)
    regime_conflict = _regime_conflicts(side, regime)
    high_vol = _is_high_vol(regime, vol_note)
    regime_block = False
    high_vol_block = False

    if regime_conflict:
        score -= float(regime_conflict_penalty)
        reasons.append(f"Regime conflict: {regime} is against {side}")
        if require_regime_alignment and not allow_counter_regime_reversal:
            regime_block = True

    if require_trend_alignment and not trend_ok:
        reasons.append("Trend alignment filter failed")
        if not hard_confirmation:
            regime_block = True

    if require_stoch_for_trigger and not stoch_ok:
        reasons.append("Stochastic timing filter failed")

    if high_vol:
        score -= 6.0
        if require_price_action_confirmation_in_high_vol and not hard_confirmation:
            high_vol_block = True
            reasons.append("High-volatility regime requires hard price-action confirmation")

    score = float(max(0.0, min(100.0, score)))

    entry_low, entry_high, sl, tp1, tp2, rr = _build_levels(
        side, last_close, atr, entry_zone_atr_multiplier, stop_loss_atr_multiplier, take_profit_1_atr_multiplier, take_profit_2_atr_multiplier
    )
    if rr < float(min_risk_reward):
        return None

    target_candidate = str(spike_ctx.spike_direction or "none").lower() == _target_spike_direction(side) and float(spike_ctx.spike_strength or 0.0) >= 0.6
    trigger_confirmed = (
        bool(trigger_alerts_enabled)
        and not regime_block
        and not high_vol_block
        and score >= trigger_min
        and spike_pressure_confirmed
        and (micro_ok or strong_rejection or not require_micro_break_for_trigger)
        and (stoch_ok or not require_stoch_for_trigger)
        and (bool(bpr_ctx.get("aligned")) or not ict_bpr_require_for_trigger)
    )

    if target_candidate and not trigger_confirmed:
        logger.info(
            "%s rejected %s trigger: score=%.1f min=%.1f spike_ok=%s micro_ok=%s rejection=%s stoch_ok=%s trend_ok=%s regime=%s regime_block=%s high_vol_block=%s rr=%.2f",
            symbol, side, score, trigger_min, spike_pressure_confirmed, micro_ok, strong_rejection, stoch_ok, trend_ok, regime, regime_block, high_vol_block, rr,
        )

    if trigger_confirmed:
        stage: AlertStage = "TRIGGER"
    elif preparation_alerts_enabled and score >= float(min_score):
        stage = "PREP"
    else:
        return None

    cleaned: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if not reason or reason in seen:
            continue
        cleaned.append(reason)
        seen.add(reason)

    return Signal(
        symbol=symbol,
        side=side,
        score=score,
        timeframe="1m setup with 5m/15m context",
        entry_zone_low=float(entry_low),
        entry_zone_high=float(entry_high),
        stop_loss=float(sl),
        take_profit_1=float(tp1),
        take_profit_2=float(tp2),
        risk_reward=float(rr),
        reasons=cleaned,
        volatility_warning=vol_note,
        regime=regime,
        timestamp_epoch=float(now_epoch),
        alert_stage=stage,
        bpr_context=bpr_ctx,
        features={
            "regime": regime,
            "regime_conflict": bool(regime_conflict),
            "high_volatility_regime": bool(high_vol),
            "trend_ok": bool(trend_ok),
            "stoch_ok": bool(stoch_ok),
            **stoch_values,
            "micro_break_confirmed": bool(micro_ok),
            "strong_rejection_confirmed": bool(strong_rejection),
            "spike_pressure_confirmed": bool(spike_pressure_confirmed),
            "spike_direction": spike_ctx.spike_direction,
            "spike_strength": float(spike_ctx.spike_strength or 0.0),
            "tick_velocity": float(spike_ctx.tick_velocity or 0.0),
            "bpr_status": bpr_ctx.get("status"),
            "bpr_aligned": bool(bpr_ctx.get("aligned")),
        },
    )
