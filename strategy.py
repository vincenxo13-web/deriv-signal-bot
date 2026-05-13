"""
Boom/Crash signal-only strategy.

Direction rules:
- BOOM symbols generate BUY triggers only.
- CRASH symbols generate SELL triggers only.

Current model:
- Trend-following spike catching.
- Boom BUY: uptrend/pullback + stochastic/RSI timing + support/BB context + target up-spike pressure.
- Crash SELL: downtrend/pull-up + stochastic/RSI timing + resistance/BB context + target down-spike pressure.
- Price-action confirmation matters more than indicators alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal
import logging

import numpy as np
import pandas as pd

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
    """Recent spike information computed from ticks + last bars."""

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
    entry_rule: str | None = None
    entry_validity: str | None = None
    confirmation_summary: str | None = None
    bpr_context: dict[str, Any] | None = None
    features: dict[str, Any] | None = None


def signal_to_storage_row(sig: Signal) -> dict[str, Any]:
    """Flatten Signal for SQLite row; nested fields go into payload JSON in storage."""
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


def _stochastic_values(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple[float, float, float, float]:
    """Return latest %K, %D, previous %K, previous %D."""
    if {"stoch_k", "stoch_d"}.issubset(df.columns) and len(df) >= 2:
        return (
            _safe_float(df["stoch_k"].iloc[-1]),
            _safe_float(df["stoch_d"].iloc[-1]),
            _safe_float(df["stoch_k"].iloc[-2]),
            _safe_float(df["stoch_d"].iloc[-2]),
        )

    if len(df) < k_period + d_period + 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100.0 * (df["close"] - low_min) / denom
    d = k.rolling(d_period).mean()
    return float(k.iloc[-1]), float(d.iloc[-1]), float(k.iloc[-2]), float(d.iloc[-2])


def _stoch_timing_ok(
    df: pd.DataFrame,
    side: Side,
    oversold: float = 20.0,
    overbought: float = 80.0,
) -> tuple[bool, str, float, float]:
    k, d, pk, pd_ = _stochastic_values(df)
    if np.isnan(k) or np.isnan(d):
        return False, "Stochastic not ready", k, d

    if side == "BUY":
        oversold_now = k <= oversold + 10
        cross_up = pk <= pd_ and k > d
        recovered = pk <= oversold and k > pk
        ok = bool(oversold_now or cross_up or recovered)
        note = f"Stoch bullish timing ok (K={k:.1f}, D={d:.1f})" if ok else f"Stoch BUY rejected: K={k:.1f}, D={d:.1f}"
        return ok, note, k, d

    overbought_now = k >= overbought - 10
    cross_down = pk >= pd_ and k < d
    rejected = pk >= overbought and k < pk
    ok = bool(overbought_now or cross_down or rejected)
    note = f"Stoch bearish timing ok (K={k:.1f}, D={d:.1f})" if ok else f"Stoch SELL rejected: K={k:.1f}, D={d:.1f}"
    return ok, note, k, d


def _micro_break_confirmed(df: pd.DataFrame, side: Side, lookback: int = 3) -> tuple[bool, str]:
    lookback = max(2, int(lookback))
    if len(df) < lookback + 2:
        return False, "Not enough candles for micro-break confirmation"
    last_close = float(df["close"].iloc[-1])
    recent = df.iloc[-(lookback + 1) : -1]
    if side == "BUY":
        micro_high = float(recent["high"].max())
        if last_close > micro_high:
            return True, f"Micro-break confirmed: close above recent {lookback}-candle high"
        return False, f"Waiting for reclaim above recent {lookback}-candle high"
    micro_low = float(recent["low"].min())
    if last_close < micro_low:
        return True, f"Micro-break confirmed: close below recent {lookback}-candle low"
    return False, f"Waiting for rejection below recent {lookback}-candle low"


def _regime_conflicts(side: Side, regime: str) -> bool:
    r = str(regime or "").lower()
    if side == "BUY":
        return "downtrend" in r
    return "uptrend" in r


def _is_high_vol(regime: str, note: str | None = None) -> bool:
    r = str(regime or "").lower()
    n = str(note or "").lower()
    return "high_volatility" in r or "high volatility" in n or "elevated" in n


def _trend_context_score(df: pd.DataFrame, side: Side) -> tuple[float, list[str], bool]:
    row = df.iloc[-1]
    close = _safe_float(row.get("close"))
    ema20 = _safe_float(row.get("ema_20"))
    ema50 = _safe_float(row.get("ema_50"))
    ema200 = _safe_float(row.get("ema_200"))
    pts = 0.0
    reasons: list[str] = []

    if side == "BUY":
        trend_ok = False
        if np.isfinite(ema20) and np.isfinite(ema50) and ema20 >= ema50:
            pts += 14.0
            trend_ok = True
            reasons.append("EMA20 >= EMA50")
        if np.isfinite(ema200) and close >= ema200:
            pts += 10.0
            trend_ok = True
            reasons.append("Price above/near EMA200")
        return pts, reasons, trend_ok

    trend_ok = False
    if np.isfinite(ema20) and np.isfinite(ema50) and ema20 <= ema50:
        pts += 14.0
        trend_ok = True
        reasons.append("EMA20 <= EMA50")
    if np.isfinite(ema200) and close <= ema200:
        pts += 10.0
        trend_ok = True
        reasons.append("Price below/near EMA200")
    return pts, reasons, trend_ok


def _rsi_score(df: pd.DataFrame, side: Side) -> tuple[float, str | None, float]:
    rsi = _safe_float(df["rsi_14"].iloc[-1]) if "rsi_14" in df.columns else float("nan")
    if not np.isfinite(rsi):
        return 0.0, None, rsi
    if side == "BUY":
        if 38 <= rsi <= 62:
            return 10.0, "RSI in Boom buy-preparation zone", rsi
        if 30 <= rsi < 38:
            return 6.0, "RSI low; possible Boom spring zone", rsi
        return 0.0, None, rsi
    if 38 <= rsi <= 62:
        return 10.0, "RSI in Crash sell-preparation zone", rsi
    if 62 < rsi <= 72:
        return 6.0, "RSI elevated; possible Crash pull-up exhaustion", rsi
    return 0.0, None, rsi


def _bb_score(df: pd.DataFrame, side: Side) -> tuple[float, str | None]:
    if not {"bb_lower", "bb_upper", "bb_mid"}.issubset(df.columns):
        return 0.0, None
    close = float(df["close"].iloc[-1])
    lower = float(df["bb_lower"].iloc[-1])
    upper = float(df["bb_upper"].iloc[-1])
    mid = float(df["bb_mid"].iloc[-1])
    if side == "BUY" and close <= lower + (mid - lower) * 0.55:
        return 7.0, "Near lower Bollinger area / pullback zone"
    if side == "SELL" and close >= upper - (upper - mid) * 0.55:
        return 7.0, "Near upper Bollinger area / pull-up zone"
    return 0.0, None


def _bollinger_squeeze_score(df: pd.DataFrame) -> tuple[float, str | None]:
    bw = bollinger_bandwidth(df).dropna()
    if len(bw) < 60:
        return 0.0, None
    last_w = float(bw.iloc[-1])
    med_w = float(bw.iloc[-60:].median())
    if last_w < med_w * 0.75:
        return 4.0, "ATR/Bollinger compression before possible spike"
    if last_w > med_w * 1.15:
        return 4.0, "Bollinger bandwidth expanding"
    return 0.0, None


def _support_resistance_score(side: Side, price: float, zones: dict[str, float | None], atr: float) -> tuple[float, str | None]:
    tol = max(atr * 0.45, 1e-9)
    if side == "BUY" and zones.get("support") is not None:
        sup = float(zones["support"])
        if abs(price - sup) <= tol * 2.5 or (price > sup and price - sup <= tol * 4.0):
            return 9.0, "Support test before Boom spike"
    if side == "SELL" and zones.get("resistance") is not None:
        res = float(zones["resistance"])
        if abs(res - price) <= tol * 2.5 or (price < res and res - price <= tol * 4.0):
            return 9.0, "Resistance test before Crash drop"
    return 0.0, None


def _spike_pressure_score(ctx: SpikeContext, side: Side, min_strength: float, min_velocity: float) -> tuple[float, str | None, bool]:
    target = _target_spike_direction(side)
    direction = str(ctx.spike_direction or "none").lower()
    strength = max(0.0, float(ctx.spike_strength or 0.0))
    velocity = abs(float(ctx.tick_velocity or 0.0))
    if direction == target and strength >= min_strength and velocity >= min_velocity:
        return 18.0, f"Target-direction spike pressure active ({target}, strength={strength:.2f})", True
    if direction == target and (strength >= min_strength * 0.75 or velocity >= min_velocity):
        return 8.0, f"Target-direction spike pressure building ({target})", False
    return 0.0, None, False


def _higher_tf_score(dfs: dict[str, pd.DataFrame], side: Side) -> tuple[float, str | None]:
    pts = 0.0
    notes: list[str] = []
    for tf in ("5m", "15m"):
        df = dfs.get(tf)
        if df is None or len(df) < 5:
            continue
        row = df.iloc[-1]
        ema20 = _safe_float(row.get("ema_20"))
        ema50 = _safe_float(row.get("ema_50"))
        if not (np.isfinite(ema20) and np.isfinite(ema50)):
            continue
        if side == "BUY" and ema20 >= ema50:
            pts += 4.0
            notes.append(f"{tf} EMA aligned")
        elif side == "SELL" and ema20 <= ema50:
            pts += 4.0
            notes.append(f"{tf} EMA aligned")
    if pts:
        return min(8.0, pts), ", ".join(notes)
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
        sl = entry_low - atr * sl_mult
        tp1 = entry_high + atr * tp1_mult
        tp2 = entry_high + atr * tp2_mult
        risk = entry_low - sl
        reward = tp1 - entry_low
    else:
        sl = entry_high + atr * sl_mult
        tp1 = entry_low - atr * tp1_mult
        tp2 = entry_low - atr * tp2_mult
        risk = sl - entry_high
        reward = entry_high - tp1
    rr = float(max(0.0, reward / max(risk, 1e-9)))
    return float(entry_low), float(entry_high), float(sl), float(tp1), float(tp2), rr


def _entry_text(side: Side, entry_low: float, entry_high: float, stop_loss: float) -> tuple[str, str]:
    if side == "BUY":
        rule = (
            f"Wait for price to hold/reclaim above {entry_low:.5f}–{entry_high:.5f}. "
            f"Avoid chasing if price falls through the zone without rejection."
        )
        validity = f"Valid while price holds/reclaims the zone. Invalid below {stop_loss:.5f}."
        return rule, validity
    rule = (
        f"Wait for price to reject/hold below {entry_low:.5f}–{entry_high:.5f}. "
        f"Avoid chasing if price breaks above the zone without rejection."
    )
    validity = f"Valid while price rejects below the zone. Invalid above {stop_loss:.5f}."
    return rule, validity


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
    trigger_min_signal_score: float = 82.0,
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
    trend_following_spike_mode: bool = True,
    require_trend_alignment: bool = True,
    require_regime_alignment: bool = True,
    allow_counter_regime_reversal: bool = False,
    regime_conflict_penalty: float = 35.0,
    require_price_action_confirmation_in_high_vol: bool = True,
    stoch_enabled: bool = True,
    require_stoch_for_trigger: bool = True,
    stoch_oversold: float = 20.0,
    stoch_overbought: float = 80.0,
    score_cap_without_bpr: float = 92.0,
    score_cap_high_volatility: float = 88.0,
    score_cap_no_hard_confirmation: float = 80.0,
    **_: Any,
) -> Signal | None:
    """Evaluate a completed 1m bar and return one signal or None."""
    side = allowed_side_for_symbol(symbol)
    if side is None:
        return None

    required_bars = max(60, int(warmup_bars if warmup_bars is not None else signal_warmup_bars))
    if df_1m.empty or len(df_1m) < required_bars:
        return None

    dfs = _attach_all(build_multi_timeframe(df_1m))
    df1 = dfs["1m"]
    if df1.empty or len(df1) < required_bars:
        return None

    row = df1.iloc[-1]
    last_close = _safe_float(row.get("close"))
    atr = _safe_float(row.get("atr_14"))
    if not np.isfinite(last_close) or not np.isfinite(atr) or atr <= 0:
        return None

    trigger_min = float(trigger_min_score if trigger_min_score is not None else trigger_min_signal_score)
    regime, vol_note = classify_regime(df1)
    zones = detect_sr_zones(df1)
    target_dir = _target_spike_direction(side)
    target_spike_candidate = spike_ctx.spike_direction == target_dir and spike_ctx.spike_strength >= max(0.65, trigger_spike_strength * 0.75)

    reasons: list[str] = []
    score = 0.0

    trend_pts, trend_reasons, trend_ok = _trend_context_score(df1, side)
    score += trend_pts
    reasons.extend(trend_reasons)

    conflict = _regime_conflicts(side, regime)
    hard_reversal_allowed = False
    if conflict:
        if not allow_counter_regime_reversal and require_regime_alignment:
            if target_spike_candidate:
                logger.info("%s rejected %s trigger: regime conflict (%s)", symbol, side, regime)
            return None
        score -= abs(float(regime_conflict_penalty))
        reasons.append(f"Regime conflict warning: {regime}")

    if require_trend_alignment and not trend_ok:
        if target_spike_candidate:
            logger.info("%s rejected %s trigger: trend context not aligned | regime=%s", symbol, side, regime)
        return None

    if stoch_enabled:
        stoch_ok, stoch_note, stoch_k, stoch_d = _stoch_timing_ok(df1, side, stoch_oversold, stoch_overbought)
        if stoch_ok:
            score += 14.0
            reasons.append(stoch_note)
        elif require_stoch_for_trigger:
            if target_spike_candidate:
                logger.info("%s rejected %s trigger: %s | regime=%s", symbol, side, stoch_note, regime)
            return None
        else:
            reasons.append(stoch_note)
    else:
        stoch_k = stoch_d = float("nan")

    rsi_pts, rsi_note, rsi = _rsi_score(df1, side)
    score += rsi_pts
    if rsi_note:
        reasons.append(rsi_note)

    sr_pts, sr_note = _support_resistance_score(side, last_close, zones, atr)
    score += sr_pts
    if sr_note:
        reasons.append(sr_note)

    bb_pts, bb_note = _bb_score(df1, side)
    score += bb_pts
    if bb_note:
        reasons.append(bb_note)

    sq_pts, sq_note = _bollinger_squeeze_score(df1)
    score += sq_pts
    if sq_note:
        reasons.append(sq_note)

    rej_pts, rej_note = candle_rejection_score(side, df1)
    rejection_ok = bool(rej_pts >= 8.0)
    score += min(float(rej_pts or 0.0), 12.0)
    if rej_note:
        reasons.append(rej_note)

    tf_pts, tf_note = _higher_tf_score(dfs, side)
    score += tf_pts
    if tf_note:
        reasons.append("Higher-TF support: " + tf_note)

    spike_pts, spike_note, spike_ok = _spike_pressure_score(spike_ctx, side, trigger_spike_strength, trigger_tick_velocity_min)
    score += spike_pts
    if spike_note:
        reasons.append(spike_note)

    micro_ok, micro_note = _micro_break_confirmed(df1, side, micro_break_lookback)
    hard_confirmation = bool(micro_ok or rejection_ok)
    if hard_confirmation:
        score += 16.0 if micro_ok else 10.0
        reasons.append(micro_note if micro_ok else "Rejection candle confirmation")
    else:
        reasons.append(micro_note)

    high_vol = _is_high_vol(regime, vol_note)
    if high_vol:
        score -= 10.0
        reasons.append("High-volatility caution: needs price-action confirmation")
        if require_price_action_confirmation_in_high_vol and not hard_confirmation:
            if target_spike_candidate:
                logger.info("%s rejected %s trigger: high-volatility regime without hard confirmation", symbol, side)
            return None

    if conflict and allow_counter_regime_reversal:
        hard_reversal_allowed = spike_ok and hard_confirmation and spike_ctx.spike_strength >= max(1.2, trigger_spike_strength)
        if not hard_reversal_allowed:
            if target_spike_candidate:
                logger.info("%s rejected %s trigger: counter-regime reversal not strong enough | regime=%s", symbol, side, regime)
            return None

    entry_low, entry_high, stop_loss, tp1, tp2, rr = _build_levels(
        side,
        last_close,
        atr,
        entry_zone_atr_multiplier,
        stop_loss_atr_multiplier,
        take_profit_1_atr_multiplier,
        take_profit_2_atr_multiplier,
    )
    if rr < float(min_risk_reward):
        if target_spike_candidate:
            logger.info("%s rejected %s trigger: R:R %.2f below minimum %.2f", symbol, side, rr, min_risk_reward)
        return None

    # Score caps make the bot more honest. A signal can be good without being 100/100.
    raw_score = score
    if not hard_confirmation:
        score = min(score, float(score_cap_no_hard_confirmation))
    if high_vol:
        score = min(score, float(score_cap_high_volatility))
    # BPR is context-only in this build. If no explicit BPR context exists, cap confidence slightly.
    score = min(score, float(score_cap_without_bpr))
    score = float(max(0.0, min(100.0, score)))

    trigger_confirmed = (
        trigger_alerts_enabled
        and score >= trigger_min
        and spike_ok
        and (hard_confirmation or not require_micro_break_for_trigger)
    )

    if trigger_confirmed:
        stage: AlertStage = "TRIGGER"
    elif preparation_alerts_enabled and score >= float(min_score):
        stage = "PREP"
    else:
        if target_spike_candidate:
            logger.info(
                "%s rejected %s trigger: score %.1f below trigger minimum %.1f or confirmation incomplete | raw=%.1f spike_ok=%s hard_confirmation=%s regime=%s stoch=%.1f/%.1f",
                symbol,
                side,
                score,
                trigger_min,
                raw_score,
                spike_ok,
                hard_confirmation,
                regime,
                stoch_k,
                stoch_d,
            )
        return None

    entry_rule, entry_validity = _entry_text(side, entry_low, entry_high, stop_loss)
    confirmation_summary = "Micro-break confirmed" if micro_ok else ("Rejection confirmed" if rejection_ok else "Waiting for confirmation")

    cleaned: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if not reason or reason in seen:
            continue
        cleaned.append(reason)
        seen.add(reason)

    logger.info(
        "%s accepted %s %s | score=%.1f trigger_min=%.1f regime=%s spike_dir=%s strength=%.2f vel=%.6f rr=%.2f",
        symbol,
        side,
        stage,
        score,
        trigger_min,
        regime,
        spike_ctx.spike_direction,
        spike_ctx.spike_strength,
        spike_ctx.tick_velocity,
        rr,
    )

    return Signal(
        symbol=symbol,
        side=side,
        score=score,
        timeframe="1m setup with 5m/15m context",
        entry_zone_low=entry_low,
        entry_zone_high=entry_high,
        stop_loss=stop_loss,
        take_profit_1=tp1,
        take_profit_2=tp2,
        risk_reward=rr,
        reasons=cleaned[:6],
        volatility_warning=vol_note,
        regime=regime,
        timestamp_epoch=float(now_epoch),
        alert_stage=stage,
        entry_rule=entry_rule,
        entry_validity=entry_validity,
        confirmation_summary=confirmation_summary,
        bpr_context={"status": "NO_DATA", "note": "H4 BPR not available in this build"},
        features={
            "regime": regime,
            "rsi": rsi,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "spike_direction": spike_ctx.spike_direction,
            "spike_strength": spike_ctx.spike_strength,
            "tick_velocity": spike_ctx.tick_velocity,
            "micro_break_ok": micro_ok,
            "rejection_ok": rejection_ok,
            "hard_confirmation": hard_confirmation,
            "score_raw": raw_score,
            "score_capped": score,
        },
    )
