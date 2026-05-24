"""
Score-based confluence strategy for Deriv Crash / Boom synthetic indices.

This is educational / research code — not investment advice.
Scores are capped at 100; this stricter build requires real wick/micro confirmation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal
import logging

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
    entry_rule: str | None = None
    entry_validity: str | None = None
    confirmation_summary: str | None = None
    features: dict[str, Any] | None = None


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


def _allowed_side(symbol: str) -> Literal["BUY", "SELL"] | None:
    sym = str(symbol).upper()
    if sym.startswith("BOOM"):
        return "BUY"
    if sym.startswith("CRASH"):
        return "SELL"
    return None


def _target_direction(side: Literal["BUY", "SELL"]) -> str:
    return "up" if side == "BUY" else "down"


def _symbol_sample_adjustment(symbol: str) -> tuple[float, str | None]:
    """Small manual adjustment from the first resolved sample.

    This is intentionally mild; it does not auto-disable symbols. It just stops
    weak symbols from looking like 92/100 signals until their outcomes improve.
    """
    sym = str(symbol).upper()
    adjustments = {
        "CRASH300N": (6.0, "Sample edge bonus: CRASH300N has been strongest so far"),
        "CRASH500": (2.0, "Sample edge bonus: CRASH500 has been acceptable so far"),
        "BOOM600": (-4.0, "Sample caution: BOOM600 has been noisy so far"),
        "BOOM900": (-12.0, "Sample caution: BOOM900 has been weak so far"),
        "CRASH900": (-10.0, "Sample caution: CRASH900 has been weak so far"),
        "CRASH1000": (-8.0, "Sample caution: CRASH1000 has been weak so far"),
    }
    return adjustments.get(sym, (0.0, None))


def _late_chase_check(df: pd.DataFrame, side: Literal["BUY", "SELL"], atr: float) -> tuple[bool, float, str | None]:
    """Reject/penalize entries that occur after the move already stretched.

    The sample showed many losses where the signal fired after a strong micro-break
    or spike candle, then price snapped back to SL. We prefer pullback/reclaim or
    rejection, not buying the top of an already-expanded candle.
    """
    if len(df) < 4 or atr <= 0:
        return False, 0.0, None

    last = df.iloc[-1]
    high = float(last["high"])
    low = float(last["low"])
    open_ = float(last["open"])
    close = float(last["close"])
    rng = max(high - low, 1e-9)
    body = abs(close - open_)
    close_pos = (close - low) / rng

    expanded = rng >= atr * 2.6 or body >= atr * 1.8
    if not expanded:
        return False, 0.0, None

    if side == "BUY" and close_pos >= 0.72:
        return True, -18.0, "Anti-chase: Boom candle already expanded near its high"
    if side == "SELL" and close_pos <= 0.28:
        return True, -18.0, "Anti-chase: Crash candle already expanded near its low"
    return False, -8.0, "Anti-chase caution: candle already expanded"


def _entry_text(side: Literal["BUY", "SELL"], entry_low: float, entry_high: float, stop_loss: float) -> tuple[str, str]:
    if side == "BUY":
        return (
            f"Wait for hold/reclaim above {entry_low:.5f}–{entry_high:.5f}.",
            f"Valid while price holds/reclaims the zone. Invalid below {stop_loss:.5f}.",
        )
    return (
        f"Wait for rejection/hold below {entry_low:.5f}–{entry_high:.5f}.",
        f"Valid while price rejects below the zone. Invalid above {stop_loss:.5f}.",
    )



def _stoch_series(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smoothing: int = 3) -> tuple[pd.Series, pd.Series]:
    """Return stochastic K/D using only OHLC columns.

    Kept local so strategy.py does not depend on extra indicator package functions.
    """
    low_min = df["low"].rolling(k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(k_period, min_periods=k_period).max()
    raw_k = 100.0 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    k = raw_k.rolling(smoothing, min_periods=1).mean().clip(0, 100)
    d = k.rolling(d_period, min_periods=1).mean().clip(0, 100)
    return k, d


def _stoch_timing_ok(df: pd.DataFrame, side: Literal["BUY", "SELL"]) -> tuple[bool, str, float, float]:
    """Stochastic timing for Boom/Crash pullback entries.

    It is not a standalone trigger. It mainly blocks obvious chase entries and
    rewards pullback/reversal timing.
    """
    if len(df) < 20:
        return False, "Stoch unavailable", float("nan"), float("nan")
    k, d = _stoch_series(df)
    k_now = float(k.iloc[-1]) if pd.notna(k.iloc[-1]) else float("nan")
    d_now = float(d.iloc[-1]) if pd.notna(d.iloc[-1]) else float("nan")
    k_prev = float(k.iloc[-2]) if len(k) >= 2 and pd.notna(k.iloc[-2]) else k_now
    if np.isnan(k_now) or np.isnan(d_now):
        return False, "Stoch unavailable", k_now, d_now

    if side == "BUY":
        recovering_from_pullback = k_now <= 55 and k_now >= k_prev
        deeply_oversold = k_now <= 25
        if recovering_from_pullback or deeply_oversold:
            return True, "Stoch pullback/recovery timing ok", k_now, d_now
        return False, f"Stoch not in Boom pullback zone (K={k_now:.1f}, D={d_now:.1f})", k_now, d_now

    rolling_from_pullup = k_now >= 45 and k_now <= k_prev
    overbought_zone = k_now >= 75
    if rolling_from_pullup or overbought_zone:
        return True, "Stoch pull-up/rollover timing ok", k_now, d_now
    return False, f"Stoch not in Crash pull-up zone (K={k_now:.1f}, D={d_now:.1f})", k_now, d_now


def _micro_break_ok(df: pd.DataFrame, side: Literal["BUY", "SELL"], lookback: int = 3) -> tuple[bool, str]:
    """Small structure break/reclaim confirmation.

    BUY: close breaks above recent highs. SELL: close breaks below recent lows.
    """
    lookback = max(2, int(lookback))
    if len(df) < lookback + 2:
        return False, "Not enough bars for micro-break"
    close = float(df["close"].iloc[-1])
    prev = df.iloc[-lookback-1:-1]
    if side == "BUY":
        level = float(prev["high"].max())
        ok = close > level
        return ok, f"Micro-break {'passed' if ok else 'missing'} above {level:.5f}"
    level = float(prev["low"].min())
    ok = close < level
    return ok, f"Micro-break {'passed' if ok else 'missing'} below {level:.5f}"


def _wick_rejection_score(df: pd.DataFrame, side: Literal["BUY", "SELL"], atr: float) -> tuple[float, str | None]:
    """Score a rejection candle around the entry zone.

    BUY needs lower wick + bullish/held close. SELL needs upper wick + bearish/held close.
    """
    if len(df) < 2 or atr <= 0:
        return 0.0, None
    row = df.iloc[-1]
    open_ = float(row["open"]); high = float(row["high"]); low = float(row["low"]); close = float(row["close"])
    rng = max(high - low, 1e-9)
    body = abs(close - open_)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    body_ratio = body / rng

    if side == "BUY":
        strong_lower = lower_wick >= max(body * 1.15, atr * 0.18)
        close_position_ok = (close - low) / rng >= 0.55
        if strong_lower and close_position_ok:
            pts = 12.0 if body_ratio <= 0.65 else 9.0
            return pts, "Bullish rejection wick confirmed"
    else:
        strong_upper = upper_wick >= max(body * 1.15, atr * 0.18)
        close_position_ok = (high - close) / rng >= 0.55
        if strong_upper and close_position_ok:
            pts = 12.0 if body_ratio <= 0.65 else 9.0
            return pts, "Bearish rejection wick confirmed"
    return 0.0, None


def _drift_exhaustion_score(df: pd.DataFrame, side: Literal["BUY", "SELL"]) -> tuple[float, str | None]:
    """Detect small exhaustion after a pullback/pull-up.

    This is intentionally weak. In the stricter build it may add score but should
    not be enough to trigger alone.
    """
    if len(df) < 5:
        return 0.0, None
    closes = df["close"].iloc[-4:].astype(float)
    if side == "BUY":
        falling_then_hold = closes.iloc[0] > closes.iloc[1] >= closes.iloc[2] and closes.iloc[-1] >= closes.iloc[-2]
        if falling_then_hold:
            return 6.0, "Pullback drift exhaustion"
    else:
        rising_then_hold = closes.iloc[0] < closes.iloc[1] <= closes.iloc[2] and closes.iloc[-1] <= closes.iloc[-2]
        if rising_then_hold:
            return 6.0, "Pull-up drift exhaustion"
    return 0.0, None


def _bollinger_squeeze_score(df: pd.DataFrame, side: Literal["BUY", "SELL"]) -> tuple[float, str | None]:
    """Small bonus for compression before move. Never a standalone trigger."""
    if not {"bb_lower", "bb_upper", "bb_mid"}.issubset(df.columns) or len(df) < 60:
        return 0.0, None
    bw = bollinger_bandwidth(df).dropna()
    if len(bw) < 40:
        return 0.0, None
    recent = float(bw.iloc[-10:].median())
    older = float(bw.iloc[-40:-10].median())
    if older <= 0:
        return 0.0, None
    compressed = recent < older * 0.85
    if not compressed:
        return 0.0, None
    close = float(df["close"].iloc[-1])
    mid = float(df["bb_mid"].iloc[-1])
    if side == "BUY" and close >= mid:
        return 4.0, "Bollinger compression with mid-band reclaim"
    if side == "SELL" and close <= mid:
        return 4.0, "Bollinger compression with mid-band rejection"
    return 2.0, "Bollinger compression"

def evaluate_signal(
    symbol: str,
    df_1m: pd.DataFrame,
    spike_ctx: SpikeContext,
    min_score: float,
    now_epoch: float,
    *,
    signal_warmup_bars: int = 220,
    warmup_bars: int | None = None,
    trigger_min_signal_score: float = 74.0,
    trigger_min_score: float | None = None,
    trigger_spike_strength: float = 0.80,
    trigger_tick_velocity_min: float = 0.008,
    entry_zone_atr_multiplier: float = 0.10,
    stop_loss_atr_multiplier: float = 3.0,
    take_profit_1_atr_multiplier: float = 4.0,
    take_profit_2_atr_multiplier: float = 7.0,
    min_risk_reward: float = 1.00,
    preparation_alerts_enabled: bool = False,
    trigger_alerts_enabled: bool = True,
    require_wick_or_micro_for_trigger: bool = True,
    allow_drift_only_confirmation: bool = False,
    min_trigger_score_without_micro_break: float = 78.0,
    min_trigger_score_without_wick: float = 76.0,
    max_trigger_stoch_buy: float = 72.0,
    min_trigger_stoch_sell: float = 28.0,
    **_: Any,
) -> Signal | None:
    """Improved trigger-only strategy from early outcome sample.

    Key changes from the sample:
    - Boom/Crash direction is locked: Boom BUY only, Crash SELL only.
    - Score is less important than price-action confirmation.
    - Avoids chasing after an already-expanded spike candle.
    - Weak sample symbols receive a mild penalty instead of a fake high score.
    - Wider TP/SL defaults for small position sizing.
    """
    side = _allowed_side(symbol)
    if side is None:
        return None

    warmup = max(60, int(warmup_bars or signal_warmup_bars))
    if df_1m.empty or len(df_1m) < warmup:
        return None

    dfs_raw = build_multi_timeframe(df_1m)
    dfs = _tf_attach(dfs_raw)
    df1 = dfs["1m"]
    if df1.empty or len(df1) < warmup:
        return None

    required = {"ema_20", "ema_50", "ema_200", "rsi_14", "atr_14"}
    if not required.issubset(df1.columns):
        return None

    row = df1.iloc[-1]
    regime, vol_note = classify_regime(df1)
    zones = detect_sr_zones(df1)
    atr = max(float(row["atr_14"]), 1e-9)
    last_close = float(row["close"])
    ema20 = float(row["ema_20"])
    ema50 = float(row["ema_50"])
    ema200 = float(row["ema_200"])
    rsi = float(row["rsi_14"])

    target_dir = _target_direction(side)
    trigger_min = float(trigger_min_score if trigger_min_score is not None else trigger_min_signal_score)

    reasons: list[str] = []
    score = 0.0

    # 1) Regime and trend. Correct regime helps, opposite regime blocks unless a very strong reversal exists.
    regime_l = str(regime or "").lower()
    target_spike = str(spike_ctx.spike_direction or "none").lower() == target_dir
    spike_strength = float(spike_ctx.spike_strength or 0.0)
    tick_velocity = abs(float(spike_ctx.tick_velocity or 0.0))

    opposite_regime = (side == "BUY" and "downtrend" in regime_l) or (side == "SELL" and "uptrend" in regime_l)
    strong_reversal_exception = target_spike and spike_strength >= 1.8 and tick_velocity >= trigger_tick_velocity_min * 2
    if opposite_regime and not strong_reversal_exception:
        logger.info(
            "%s rejected %s trigger: opposite regime %s | spike_dir=%s strength=%.2f vel=%.6f",
            symbol, side, regime, spike_ctx.spike_direction, spike_strength, tick_velocity,
        )
        return None
    if opposite_regime:
        score -= 16.0
        reasons.append(f"Counter-regime reversal attempt: {regime}")

    # Trend structure: useful, but not allowed to create a signal alone.
    if side == "BUY":
        if last_close >= ema200:
            score += 10.0
            reasons.append("Price above/near EMA200")
        if ema20 >= ema50:
            score += 10.0
            reasons.append("EMA20 >= EMA50")
    else:
        if last_close <= ema200:
            score += 10.0
            reasons.append("Price below/near EMA200")
        if ema20 <= ema50:
            score += 10.0
            reasons.append("EMA20 <= EMA50")

    # 2) Stochastic/RSI timing. Penalize bad timing learned from losses.
    stoch_ok, stoch_note, stoch_k, stoch_d = _stoch_timing_ok(df1, side)
    if stoch_ok:
        score += 10.0
        reasons.append(f"{stoch_note} (K={stoch_k:.1f}, D={stoch_d:.1f})")
    else:
        score -= 6.0
        reasons.append(stoch_note)

    # Strong chase warnings from sample: Boom buying when stoch/RSI already too hot, or Crash selling when already oversold.
    if side == "BUY" and (stoch_k >= 70 or rsi >= 76):
        score -= 14.0
        reasons.append("Momentum already hot for Boom BUY — chase penalty")
    if side == "SELL" and (stoch_k <= 30 or rsi <= 30):
        score -= 14.0
        reasons.append("Momentum already stretched for Crash SELL — chase penalty")

    # Hard loss-control blocks from live outcomes: do not buy Boom after it is
    # already hot, and do not sell Crash after it is already exhausted down.
    if side == "BUY" and stoch_k > max_trigger_stoch_buy:
        logger.info(
            "%s rejected BUY trigger: stoch chase block K=%.1f above %.1f",
            symbol, stoch_k, max_trigger_stoch_buy,
        )
        return None
    if side == "SELL" and stoch_k < min_trigger_stoch_sell:
        logger.info(
            "%s rejected SELL trigger: stoch exhaustion block K=%.1f below %.1f",
            symbol, stoch_k, min_trigger_stoch_sell,
        )
        return None

    r_pts, r_note = _rsi_zone_score(side, rsi)
    score += min(8.0, r_pts)
    if r_note:
        reasons.append(r_note)

    # 3) Location: support/resistance and BB context are useful, but lower weighted than before.
    pts, note = _support_resistance_score(side, last_close, zones, atr)
    score += min(8.0, pts)
    if note:
        reasons.append(note)

    pts, note = _bb_confluence(df1, side)
    score += min(6.0, pts)
    if note:
        reasons.append(note)

    pts, note = _bollinger_squeeze_score(df1, side)
    score += min(4.0, pts)
    if note:
        reasons.append(note)

    # 4) Price-action confirmation.
    micro_ok, micro_note = _micro_break_ok(df1, side, lookback=3)
    wick_pts, wick_note = _wick_rejection_score(df1, side, atr)
    wick_ok = wick_pts >= 7.0
    drift_pts, drift_note = _drift_exhaustion_score(df1, side)
    drift_ok = drift_pts >= 7.0
    pa_ok = bool(micro_ok or wick_ok or drift_ok)

    if micro_ok:
        score += 18.0
        reasons.append("Micro-break confirmed")
    if wick_ok:
        score += 14.0
        reasons.append(wick_note or "Rejection confirmed")
    if drift_ok:
        score += 6.0
        reasons.append(drift_note or "Drift exhaustion confirmed")

    # Loss-control gate: early spike pressure must not be the only trigger.
    # Require either a structure break or a rejection wick. Drift exhaustion can
    # support a signal, but by default it cannot trigger alone.
    if require_wick_or_micro_for_trigger and not (micro_ok or wick_ok):
        logger.info(
            "%s rejected %s trigger: no wick/micro confirmation | %s drift=%s",
            symbol, side, micro_note, drift_ok,
        )
        return None
    if drift_ok and not (micro_ok or wick_ok) and not allow_drift_only_confirmation:
        logger.info("%s rejected %s trigger: drift-only confirmation disabled", symbol, side)
        return None

    # 5) Spike pressure. Require target-direction pressure, but do not blindly chase it.
    if target_spike and spike_strength >= trigger_spike_strength and tick_velocity >= trigger_tick_velocity_min:
        score += 14.0
        reasons.append(f"Target spike pressure active ({target_dir}, strength={spike_strength:.2f})")
    else:
        logger.info(
            "%s rejected %s trigger: target spike pressure missing/weak | dir=%s strength=%.2f vel=%.6f",
            symbol, side, spike_ctx.spike_direction, spike_strength, tick_velocity,
        )
        return None

    # 6) Anti-chase from outcome sample.
    reject_chase, chase_penalty, chase_note = _late_chase_check(df1, side, atr)
    if chase_note:
        reasons.append(chase_note)
    score += chase_penalty
    if reject_chase and not wick_ok:
        logger.info("%s rejected %s trigger: %s", symbol, side, chase_note)
        return None

    # 7) Higher-timeframe context: smaller bonus than before.
    pts, note = _tf_alignment_bonus(dfs, side)
    score += min(6.0, pts)
    if note:
        reasons.append(note)

    # 8) Volatility and symbol sample adjustment.
    if str(regime).endswith("high_volatility") or (isinstance(vol_note, str) and "elevated" in vol_note.lower()):
        score -= 12.0
        reasons.append("High-volatility caution")
        if not micro_ok:
            logger.info("%s rejected %s trigger: high volatility without micro-break", symbol, side)
            return None

    sym_adj, sym_note = _symbol_sample_adjustment(symbol)
    score += sym_adj
    if sym_note:
        reasons.append(sym_note)

    # Score honesty: cap overconfident messages. The sample showed 90+ did not outperform.
    cap = 90.0
    if not micro_ok:
        cap = min(cap, 84.0)
    if "transition" in regime_l or "ranging" in regime_l:
        cap = min(cap, 86.0)
    if str(symbol).upper() in {"BOOM900", "CRASH900", "CRASH1000"}:
        cap = min(cap, 82.0)
    score = float(max(0.0, min(cap, score)))

    # More protection for signals without the strongest confirmation.
    if not micro_ok and score < float(min_trigger_score_without_micro_break):
        logger.info(
            "%s rejected %s trigger: %.1f below no-micro minimum %.1f | wick=%s drift=%s regime=%s",
            symbol, side, score, float(min_trigger_score_without_micro_break), wick_ok, drift_ok, regime,
        )
        return None
    if not wick_ok and not micro_ok and score < float(min_trigger_score_without_wick):
        logger.info(
            "%s rejected %s trigger: %.1f below no-wick minimum %.1f | drift=%s regime=%s",
            symbol, side, score, float(min_trigger_score_without_wick), drift_ok, regime,
        )
        return None

    if not trigger_alerts_enabled or score < trigger_min:
        if spike_strength >= trigger_spike_strength:
            logger.info(
                "%s rejected %s trigger: score %.1f below %.1f after filters | regime=%s stoch=%.1f micro=%s wick=%s",
                symbol, side, score, trigger_min, regime, stoch_k, micro_ok, wick_ok,
            )
        return None

    entry_mult = max(0.05, float(entry_zone_atr_multiplier))
    sl_mult = max(1.0, float(stop_loss_atr_multiplier))
    tp1_mult = max(1.0, float(take_profit_1_atr_multiplier))
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
    if rr < min_risk_reward:
        return None

    entry_rule, entry_validity = _entry_text(side, entry_low, entry_high, sl)
    confirmation = "micro-break + rejection" if (micro_ok and wick_ok) else "micro-break" if micro_ok else "rejection" if wick_ok else "drift exhaustion"

    # Clean reason list for storage/dashboard.
    cleaned: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason and reason not in seen:
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
        risk_reward=rr,
        reasons=cleaned,
        volatility_warning=vol_note,
        regime=regime,
        timestamp_epoch=float(now_epoch),
        alert_stage="TRIGGER",
        entry_rule=entry_rule,
        entry_validity=entry_validity,
        confirmation_summary=confirmation,
        features={
            "regime": regime,
            "rsi": rsi,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "spike_direction": spike_ctx.spike_direction,
            "spike_strength": spike_strength,
            "tick_velocity": tick_velocity,
            "micro_break_ok": micro_ok,
            "wick_rejection_ok": wick_ok,
            "drift_exhaustion_ok": drift_ok,
            "anti_chase_rejected": reject_chase,
            "score_capped": score,
        },
    )
