"""
Score-based confluence strategy for Deriv Crash / Boom synthetic indices.

This is educational / research code — not investment advice.
Scores are capped at 100; default minimum to alert is 75 via config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from indicators import (
    attach_core_indicators,
    bollinger_bandwidth,
    candle_rejection_score,
    classify_regime,
    detect_sr_zones,
    last_bar_spike_metrics,
)


@dataclass
class SpikeContext:
    """Recent spike information computed from ticks + last bars."""

    last_spike_epoch: float | None
    spike_direction: str  # "up" | "down" | "none"
    spike_strength: float  # abstract 0..1+
    tick_velocity: float


@dataclass
class Signal:
    symbol: str
    side: Literal["BUY", "SELL"]
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
    alert_stage: str = "TRIGGER"


def signal_to_storage_row(sig: Signal) -> dict[str, Any]:
    """Flatten Signal for SQLite row (extras become JSON blob)."""
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


def allowed_side_for_symbol(symbol: str) -> Literal["BUY", "SELL"] | None:
    """Crash = SELL only, Boom = BUY only."""
    sym = str(symbol).upper()
    if sym.startswith("BOOM"):
        return "BUY"
    if sym.startswith("CRASH"):
        return "SELL"
    return None


def _target_spike_direction(side: Literal["BUY", "SELL"]) -> str:
    """BUY aims to catch upward Boom spikes; SELL aims to catch downward Crash spikes."""
    return "up" if side == "BUY" else "down"


def _resample_ohlc(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    out = df_1m.resample(rule, label="right", closed="right").agg(agg).dropna()
    return out


def build_multi_timeframe(dfm_1m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build 5m / 15m candles from completed 1m history."""
    if dfm_1m.empty:
        return {"1m": dfm_1m, "5m": dfm_1m, "15m": dfm_1m}
    ohlc = dfm_1m[["open", "high", "low", "close"]].copy()
    df5 = _resample_ohlc(ohlc, "5min")
    df15 = _resample_ohlc(ohlc, "15min")
    return {"1m": ohlc, "5m": df5, "15m": df15}


def _tf_attach(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        if len(df) < 10:
            out[tf] = df
            continue
        out[tf] = attach_core_indicators(df)
    return out


def _macd_hist_improving(series: pd.Series, side: Literal["BUY", "SELL"]) -> tuple[float, str | None]:
    s = series.dropna()
    if len(s) < 5:
        return 0.0, None
    last = float(s.iloc[-1])
    prev = float(s.iloc[-2])
    older = float(s.iloc[-5])
    if side == "BUY":
        improving = last > prev and last > older * 0.6
        turning = prev <= 0 and last > prev
        if turning:
            return 15.0, "MACD histogram turning up"
        if improving:
            return 12.0, "MACD histogram strengthening"
        if last > 0:
            return 7.0, "MACD histogram positive"
        return 0.0, None
    # SELL
    improving = last < prev and last < older * 0.6
    turning = prev >= 0 and last < prev
    if turning:
        return 15.0, "MACD histogram turning down"
    if improving:
        return 12.0, "MACD histogram weakening"
    if last < 0:
        return 7.0, "MACD histogram negative"
    return 0.0, None


def _rsi_zone_score(side: Literal["BUY", "SELL"], rsi: float) -> tuple[float, str | None]:
    if np.isnan(rsi):
        return 0.0, None
    if side == "BUY":
        if 40 <= rsi <= 65:
            # Peak score mid-range
            dist = abs(rsi - 52.5)
            pts = max(0.0, 15.0 - dist * 0.35)
            return pts, "RSI in healthy bullish zone"
        if rsi < 40:
            return 5.0, "RSI not overbought (lower zone)"
        if rsi > 65:
            return 0.0, None
    else:
        if 35 <= rsi <= 60:
            dist = abs(rsi - 47.5)
            pts = max(0.0, 15.0 - dist * 0.35)
            return pts, "RSI in healthy bearish zone"
        if rsi > 60:
            return 5.0, "RSI not oversold (upper zone)"
        if rsi < 35:
            return 0.0, None
    return 0.0, None


def _bb_confluence(df: pd.DataFrame, side: Literal["BUY", "SELL"]) -> tuple[float, str | None]:
    if not {"bb_lower", "bb_upper", "bb_mid"}.issubset(df.columns):
        return 0.0, None
    bw = bollinger_bandwidth(df).dropna()
    if len(bw) < 30:
        return 0.0, None
    last_w = float(bw.iloc[-1])
    med_w = float(bw.iloc[-30:].median())
    close = float(df["close"].iloc[-1])
    lower = float(df["bb_lower"].iloc[-1])
    upper = float(df["bb_upper"].iloc[-1])
    mid = float(df["bb_mid"].iloc[-1])

    if side == "BUY":
        near_lower = close <= lower + (mid - lower) * 0.35
        expansion = last_w > med_w * 1.05
        squeeze_release = med_w < float(bw.iloc[-60:].median()) * 0.9 if len(bw) >= 60 else False
        pts = 0.0
        notes: list[str] = []
        if near_lower:
            pts += 5.0
            notes.append("Price near lower Bollinger Band")
        if expansion:
            pts += 5.0
            notes.append("Bollinger bandwidth expanding")
        if squeeze_release and close > mid:
            pts += 4.0
            notes.append("Squeeze release with reclaim of mid-BB")
        if pts == 0.0:
            return 0.0, None
        return min(10.0, pts), "; ".join(notes)
    # SELL
    near_upper = close >= upper - (upper - mid) * 0.35
    expansion = last_w > med_w * 1.05
    squeeze_release = med_w < float(bw.iloc[-60:].median()) * 0.9 if len(bw) >= 60 else False
    pts = 0.0
    notes = []
    if near_upper:
        pts += 5.0
        notes.append("Price near upper Bollinger Band")
    if expansion:
        pts += 5.0
        notes.append("Bollinger bandwidth expanding")
    if squeeze_release and close < mid:
        pts += 4.0
        notes.append("Squeeze release with loss of mid-BB")
    if pts == 0.0:
        return 0.0, None
    return min(10.0, pts), "; ".join(notes)


def _support_resistance_score(
    side: Literal["BUY", "SELL"],
    price: float,
    zones: dict[str, float | None],
    atr: float,
) -> tuple[float, str | None]:
    tol = max(atr * 0.35, 1e-9)
    if side == "BUY" and zones.get("support") is not None:
        sup = float(zones["support"])
        if abs(price - sup) <= tol * 2.2:
            return 15.0, "Support zone rejection / test"
        if price > sup and (price - sup) <= tol * 4:
            return 10.0, "Price holding above nearby support"
    if side == "SELL" and zones.get("resistance") is not None:
        res = float(zones["resistance"])
        if abs(res - price) <= tol * 2.2:
            return 15.0, "Resistance zone rejection / test"
        if price < res and (res - price) <= tol * 4:
            return 10.0, "Price failing below nearby resistance"
    return 0.0, None


def _spike_penalty(
    symbol: str,
    side: Literal["BUY", "SELL"],
    ctx: SpikeContext,
    boom: bool,
) -> tuple[float, str | None]:
    """
    Crash/Boom spike behavior differs:
      - Boom: violent up-spikes common
      - Crash: violent down-spikes common

    Penalize entries that fight a fresh extreme spike without confirmation.
    """
    if ctx.last_spike_epoch is None or ctx.spike_direction == "none":
        return 0.0, None

    strength = min(2.0, max(0.0, ctx.spike_strength))
    dangerous = False
    if boom and side == "SELL" and ctx.spike_direction == "up" and strength > 0.7:
        dangerous = True
    if boom and side == "BUY" and ctx.spike_direction == "up" and strength > 1.2:
        # Chasing vertical blow-off
        dangerous = True
    if (not boom) and side == "BUY" and ctx.spike_direction == "down" and strength > 0.7:
        dangerous = True
    if (not boom) and side == "SELL" and ctx.spike_direction == "down" and strength > 1.2:
        dangerous = True

    if dangerous:
        return -25.0, "Recent abnormal spike — waiting for confirmation is safer"
    if strength > 0.55:
        return -10.0, "Elevated post-spike drift — lower confidence"
    return 0.0, None


def _tf_alignment_bonus(dfs: dict[str, pd.DataFrame], side: Literal["BUY", "SELL"]) -> tuple[float, str | None]:
    """Reward when 5m + 15m EMA stacks agree with the trade direction."""
    pts = 0.0
    notes: list[str] = []
    for tf in ("5m", "15m"):
        df = dfs.get(tf)
        if df is None or len(df) < 5:
            continue
        row = df.iloc[-1]
        ema20, ema50 = row.get("ema_20"), row.get("ema_50")
        if any(pd.isna(v) for v in (ema20, ema50)):
            continue
        if side == "BUY" and ema20 > ema50:
            pts += 5.0
            notes.append(f"{tf} EMA20>EMA50")
        if side == "SELL" and ema20 < ema50:
            pts += 5.0
            notes.append(f"{tf} EMA20<EMA50")
    if pts == 0.0:
        return 0.0, None
    return min(10.0, pts), "Higher timeframe trend alignment: " + ", ".join(notes)


def _stochastic_values(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple[float, float, float, float]:
    """Return latest %K, %D, previous %K, previous %D. No external dependency."""
    if len(df) < k_period + d_period + 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100.0 * (df["close"] - low_min) / denom
    d = k.rolling(d_period).mean()
    return float(k.iloc[-1]), float(d.iloc[-1]), float(k.iloc[-2]), float(d.iloc[-2])


def _micro_break_ok(df: pd.DataFrame, side: Literal["BUY", "SELL"], lookback: int = 3) -> tuple[bool, str]:
    if len(df) < lookback + 2:
        return False, "not enough candles for micro-break"
    last = df.iloc[-1]
    recent = df.iloc[-(lookback + 1):-1]
    if side == "BUY":
        micro_high = float(recent["high"].max())
        if float(last["close"]) > micro_high:
            return True, "bullish micro-break confirmed"
        return False, f"no bullish micro-break: close {float(last['close']):.5f} <= recent high {micro_high:.5f}"
    micro_low = float(recent["low"].min())
    if float(last["close"]) < micro_low:
        return True, "bearish micro-break confirmed"
    return False, f"no bearish micro-break: close {float(last['close']):.5f} >= recent low {micro_low:.5f}"


def _trend_following_ok(df: pd.DataFrame, side: Literal["BUY", "SELL"]) -> tuple[bool, str]:
    if len(df) < 50:
        return False, "not enough candles for trend check"
    row = df.iloc[-1]
    ema20 = float(row.get("ema_20", np.nan))
    ema50 = float(row.get("ema_50", np.nan))
    ema200 = float(row.get("ema_200", np.nan))
    close = float(row["close"])
    if any(np.isnan(v) for v in (ema20, ema50)):
        return False, "EMA data not ready"
    if side == "BUY":
        ok = ema20 >= ema50 or close >= ema200
        return ok, "Boom uptrend/pullback context ok" if ok else "Boom BUY rejected: not in bullish trend context"
    ok = ema20 <= ema50 or close <= ema200
    return ok, "Crash downtrend/pull-up context ok" if ok else "Crash SELL rejected: not in bearish trend context"


def _stoch_timing_ok(df: pd.DataFrame, side: Literal["BUY", "SELL"], oversold: float = 20.0, overbought: float = 80.0) -> tuple[bool, str, float, float]:
    k, d, pk, pd_ = _stochastic_values(df)
    if np.isnan(k) or np.isnan(d):
        return False, "stochastic not ready", k, d
    if side == "BUY":
        oversold_now = k <= oversold + 10
        cross_up = pk <= pd_ and k > d
        recovered = pk < oversold and k > pk
        ok = bool(oversold_now or cross_up or recovered)
        note = "Stoch bullish timing ok" if ok else f"Stoch BUY rejected: K={k:.1f}, D={d:.1f}, need oversold/recovery"
        return ok, note, k, d
    overbought_now = k >= overbought - 10
    cross_down = pk >= pd_ and k < d
    rejected = pk > overbought and k < pk
    ok = bool(overbought_now or cross_down or rejected)
    note = "Stoch bearish timing ok" if ok else f"Stoch SELL rejected: K={k:.1f}, D={d:.1f}, need overbought/rejection"
    return ok, note, k, d


def evaluate_signal(
    symbol: str,
    df_1m: pd.DataFrame,
    spike_ctx: SpikeContext,
    min_score: float,
    now_epoch: float,
    warmup_bars: int | None = None,
    signal_warmup_bars: int | None = None,
    trigger_min_score: float | None = None,
    trigger_spike_strength: float = 0.8,
    trigger_tick_velocity_min: float = 0.01,
    preparation_alerts_enabled: bool = False,
    trigger_alerts_enabled: bool = True,
    require_trend_alignment: bool = True,
    require_stoch_for_trigger: bool = True,
    **_: Any,
) -> Signal | None:
    """
    Trend-following Crash/Boom spike trigger strategy with useful rejection logs.

    Boom: BUY only, prefer uptrend + pull-down + stochastic oversold/recovery.
    Crash: SELL only, prefer downtrend + pull-up + stochastic overbought/rejection.
    """
    required_bars = int(warmup_bars or signal_warmup_bars or 180)
    if df_1m.empty or len(df_1m) < required_bars:
        return None

    allowed_side = allowed_side_for_symbol(symbol)
    if allowed_side is None:
        return None

    dfs_raw = build_multi_timeframe(df_1m)
    dfs = _tf_attach(dfs_raw)
    df1 = dfs["1m"]
    if df1.empty or len(df1) < required_bars:
        return None

    side: Literal["BUY", "SELL"] = allowed_side
    target_dir = _target_spike_direction(side)

    # Only log useful target-direction spike candidates to avoid spam.
    is_target_spike_candidate = (
        spike_ctx.spike_direction == target_dir
        and spike_ctx.spike_strength >= max(0.65, trigger_spike_strength * 0.75)
    )

    trigger_min = float(trigger_min_score if trigger_min_score is not None else max(min_score, 78.0))

    regime, vol_note = classify_regime(df1)
    zones = detect_sr_zones(df1)
    atr = float(df1["atr_14"].iloc[-1]) if "atr_14" in df1.columns else 0.0
    if not np.isfinite(atr) or atr <= 0:
        return None
    last_close = float(df1["close"].iloc[-1])

    reasons: list[str] = []
    score = 0.0

    trend_ok, trend_note = _trend_following_ok(df1, side)
    if trend_ok:
        score += 22.0
        reasons.append(trend_note)
    elif require_trend_alignment:
        if is_target_spike_candidate:
            logger.info(
                "%s rejected %s trigger: %s | spike_dir=%s strength=%.2f vel=%.6f",
                symbol, side, trend_note, spike_ctx.spike_direction, spike_ctx.spike_strength, spike_ctx.tick_velocity,
            )
        return None
    else:
        reasons.append(trend_note)

    stoch_ok, stoch_note, stoch_k, stoch_d = _stoch_timing_ok(df1, side)
    if stoch_ok:
        score += 18.0
        reasons.append(stoch_note)
    elif require_stoch_for_trigger:
        if is_target_spike_candidate:
            logger.info(
                "%s rejected %s trigger: %s | spike_dir=%s strength=%.2f vel=%.6f",
                symbol, side, stoch_note, spike_ctx.spike_direction, spike_ctx.spike_strength, spike_ctx.tick_velocity,
            )
        return None
    else:
        reasons.append(stoch_note)

    ema20 = float(df1["ema_20"].iloc[-1]) if "ema_20" in df1.columns else np.nan
    ema50 = float(df1["ema_50"].iloc[-1]) if "ema_50" in df1.columns else np.nan
    ema200 = float(df1["ema_200"].iloc[-1]) if "ema_200" in df1.columns else np.nan
    rsi = float(df1["rsi_14"].iloc[-1]) if "rsi_14" in df1.columns else np.nan

    if side == "BUY":
        if np.isfinite(ema20) and np.isfinite(ema50) and ema20 >= ema50:
            score += 10.0
            reasons.append("EMA20 >= EMA50")
        if np.isfinite(ema200) and last_close >= ema200:
            score += 8.0
            reasons.append("Price above/near EMA200")
    else:
        if np.isfinite(ema20) and np.isfinite(ema50) and ema20 <= ema50:
            score += 10.0
            reasons.append("EMA20 <= EMA50")
        if np.isfinite(ema200) and last_close <= ema200:
            score += 8.0
            reasons.append("Price below/near EMA200")

    r_pts, r_note = _rsi_zone_score(side, rsi)
    score += r_pts
    if r_note:
        reasons.append(r_note)

    sr_pts, sr_note = _support_resistance_score(side, last_close, zones, atr)
    score += sr_pts
    if sr_note:
        reasons.append(sr_note)

    bb_pts, bb_note = _bb_confluence(df1, side)
    score += bb_pts
    if bb_note:
        reasons.append(bb_note)

    rej_pts, rej_note = candle_rejection_score(side, df1)
    if rej_pts:
        score += min(12.0, float(rej_pts))
        if rej_note:
            reasons.append(rej_note)

    tf_pts, tf_note = _tf_alignment_bonus(dfs, side)
    score += tf_pts
    if tf_note:
        reasons.append(tf_note)

    spike_ok = (
        spike_ctx.spike_direction == target_dir
        and spike_ctx.spike_strength >= trigger_spike_strength
        and abs(spike_ctx.tick_velocity) >= trigger_tick_velocity_min
    )
    if spike_ok:
        score += 18.0
        reasons.append(f"Target-direction spike pressure: {target_dir} strength {spike_ctx.spike_strength:.2f}")
    else:
        if is_target_spike_candidate:
            logger.info(
                "%s rejected %s trigger: spike pressure below threshold | dir=%s strength=%.2f min=%.2f vel=%.6f min_vel=%.6f",
                symbol, side, spike_ctx.spike_direction, spike_ctx.spike_strength, trigger_spike_strength,
                abs(spike_ctx.tick_velocity), trigger_tick_velocity_min,
            )
        return None

    micro_ok, micro_note = _micro_break_ok(df1, side)
    rejection_ok = bool(rej_pts >= 8.0)
    confirmation_ok = micro_ok or rejection_ok
    if confirmation_ok:
        score += 12.0
        reasons.append(micro_note if micro_ok else "Rejection candle confirmation")
    else:
        if is_target_spike_candidate:
            logger.info(
                "%s rejected %s trigger: no micro-break/rejection confirmation | %s | score=%.1f trigger_min=%.1f spike_strength=%.2f vel=%.6f stoch_k=%.1f stoch_d=%.1f",
                symbol, side, micro_note, score, trigger_min, spike_ctx.spike_strength, spike_ctx.tick_velocity, stoch_k, stoch_d,
            )
        return None

    if regime.endswith("high_volatility") or (isinstance(vol_note, str) and "elevated" in vol_note.lower()):
        score -= 6.0
        reasons.append("High volatility haircut")

    score = float(max(0.0, min(100.0, score)))
    if score < trigger_min:
        if is_target_spike_candidate:
            logger.info(
                "%s rejected %s trigger: score %.1f below trigger minimum %.1f | spike_dir=%s strength=%.2f vel=%.6f micro_ok=%s stoch_k=%.1f stoch_d=%.1f",
                symbol, side, score, trigger_min, spike_ctx.spike_direction, spike_ctx.spike_strength,
                spike_ctx.tick_velocity, micro_ok, stoch_k, stoch_d,
            )
        return None

    # ATR-based TP/SL defaults. Wider than earlier tight targets.
    entry_zone_atr_multiplier = 0.08
    stop_loss_atr_multiplier = 2.8
    tp1_atr_multiplier = 3.5
    tp2_atr_multiplier = 6.0

    entry_low = last_close - atr * entry_zone_atr_multiplier
    entry_high = last_close + atr * entry_zone_atr_multiplier
    if side == "BUY":
        sl = entry_low - atr * stop_loss_atr_multiplier
        tp1 = entry_high + atr * tp1_atr_multiplier
        tp2 = entry_high + atr * tp2_atr_multiplier
        risk = entry_low - sl
        reward = tp1 - entry_low
    else:
        sl = entry_high + atr * stop_loss_atr_multiplier
        tp1 = entry_low - atr * tp1_atr_multiplier
        tp2 = entry_low - atr * tp2_atr_multiplier
        risk = sl - entry_high
        reward = entry_high - tp1
    rr = float(max(0.0, reward / max(risk, 1e-9)))

    if rr < 1.2:
        if is_target_spike_candidate:
            logger.info(
                "%s rejected %s trigger: R:R %.2f below minimum 1.20 | score=%.1f",
                symbol, side, rr, score,
            )
        return None

    reasons = reasons[:6]
    reasons.append(f"Stoch K/D {stoch_k:.1f}/{stoch_d:.1f}")

    logger.info(
        "%s accepted %s TRIGGER | score=%.1f trigger_min=%.1f spike_dir=%s strength=%.2f vel=%.6f rr=%.2f",
        symbol, side, score, trigger_min, spike_ctx.spike_direction, spike_ctx.spike_strength,
        spike_ctx.tick_velocity, rr,
    )

    return Signal(
        symbol=symbol,
        side=side,
        score=score,
        timeframe="1m trigger with trend/stochastic context",
        entry_zone_low=float(entry_low),
        entry_zone_high=float(entry_high),
        stop_loss=float(sl),
        take_profit_1=float(tp1),
        take_profit_2=float(tp2),
        risk_reward=rr,
        reasons=reasons,
        volatility_warning=vol_note,
        regime=regime,
        timestamp_epoch=float(now_epoch),
        alert_stage="TRIGGER",
    )
