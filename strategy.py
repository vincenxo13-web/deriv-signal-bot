"""
Improved two-stage Boom/Crash signal-only strategy.

Directional rules:
- BOOM symbols: BUY only, aiming to anticipate upward spike/sniper entries.
- CRASH symbols: SELL only, aiming to anticipate downward spike/drop entries.

Alert stages:
- PREP: watch-only setup forming near a useful zone.
- TRIGGER: stronger entry confirmation after score + spike pressure + micro-break.

Strategy upgrades:
- Direction locked by market type: Boom=BUY, Crash=SELL.
- Multi-timeframe EMA context: 1m setup with 5m/15m agreement.
- RSI + MACD momentum confirmation.
- Support/resistance and Bollinger-zone confluence.
- Bollinger squeeze / volatility expansion awareness.
- False-break / rejection logic to avoid entering while drift continues.
- Micro-break confirmation for TRIGGER alerts:
  Boom BUY requires close above recent micro-high.
  Crash SELL requires close below recent micro-low.
- Configurable ATR-based entry, SL, TP1, TP2 and minimum R:R.

This is research / decision-support code only, not financial advice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from indicators import (
    attach_core_indicators,
    bollinger_bandwidth,
    candle_rejection_score,
    classify_regime,
    detect_sr_zones,
)

AlertStage = Literal["PREP", "TRIGGER"]
Side = Literal["BUY", "SELL"]


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
    alert_stage: AlertStage = "PREP"
    features: dict[str, Any] = field(default_factory=dict)


def signal_to_storage_row(sig: Signal) -> dict[str, Any]:
    """Flatten Signal for SQLite row. Extras become JSON payload."""
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
    return symbol.upper().startswith("BOOM")


def _is_crash(symbol: str) -> bool:
    return symbol.upper().startswith("CRASH")


def allowed_side_for_symbol(symbol: str) -> Side | None:
    """Only generate BUY for Boom and SELL for Crash."""
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
    return {
        "1m": ohlc,
        "5m": _resample_ohlc(ohlc, "5min"),
        "15m": _resample_ohlc(ohlc, "15min"),
    }


def _attach_all(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        out[tf] = attach_core_indicators(df) if len(df) >= 10 else df
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _rsi_score(side: Side, rsi: float) -> tuple[float, str | None]:
    if np.isnan(rsi):
        return 0.0, None

    if side == "BUY":
        # Boom spike entries tend to work best when RSI is not already stretched.
        if 38 <= rsi <= 58:
            return 14.0, "RSI is in a Boom buy-preparation zone, not overextended"
        if 30 <= rsi < 38:
            return 10.0, "RSI is low; possible spring setup before Boom spike"
        if 58 < rsi <= 66:
            return 5.0, "RSI is positive but needs price confirmation"
    else:
        # Crash drop entries tend to work best when RSI is not already oversold.
        if 42 <= rsi <= 62:
            return 14.0, "RSI is in a Crash sell-preparation zone, not oversold"
        if 62 < rsi <= 72:
            return 10.0, "RSI is high; possible exhaustion before Crash drop"
        if 34 <= rsi < 42:
            return 5.0, "RSI is weak but needs price confirmation"

    return 0.0, None


def _macd_score(series: pd.Series, side: Side) -> tuple[float, str | None]:
    s = series.dropna()
    if len(s) < 6:
        return 0.0, None

    last = float(s.iloc[-1])
    prev = float(s.iloc[-2])
    older = float(s.iloc[-6])

    if side == "BUY":
        if last > prev > older:
            return 13.0, "MACD histogram is improving into a bullish turn"
        if last > prev:
            return 8.0, "MACD histogram is starting to improve"
    else:
        if last < prev < older:
            return 13.0, "MACD histogram is weakening into a bearish turn"
        if last < prev:
            return 8.0, "MACD histogram is starting to weaken"

    return 0.0, None


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
            return 12.0, "Price is near lower Bollinger area: Boom sniper buy zone"
        if close > mid and expanding:
            return 7.0, "Price reclaimed mid-BB with volatility expanding"
    else:
        if close >= upper - (upper - mid) * 0.45:
            return 12.0, "Price is near upper Bollinger area: Crash sniper sell zone"
        if close < mid and expanding:
            return 7.0, "Price lost mid-BB with volatility expanding"

    return 0.0, None


def _bollinger_squeeze_score(df: pd.DataFrame, side: Side) -> tuple[float, str | None]:
    """Reward low-volatility compression that is starting to release in target direction."""
    if len(df) < 80 or not {"bb_mid", "bb_lower", "bb_upper"}.issubset(df.columns):
        return 0.0, None

    bw = bollinger_bandwidth(df).dropna()
    if len(bw) < 60:
        return 0.0, None

    last_w = float(bw.iloc[-1])
    recent_median = float(bw.iloc[-40:].median())
    older_median = float(bw.iloc[-80:-40].median()) if len(bw) >= 80 else recent_median
    close = float(df["close"].iloc[-1])
    mid = float(df["bb_mid"].iloc[-1])

    was_compressed = recent_median < older_median * 0.9 if older_median > 0 else False
    expanding = last_w > recent_median * 1.08 if recent_median > 0 else False

    if not (was_compressed and expanding):
        return 0.0, None

    if side == "BUY" and close >= mid:
        return 6.0, "Bollinger squeeze is releasing upward"
    if side == "SELL" and close <= mid:
        return 6.0, "Bollinger squeeze is releasing downward"
    return 3.0, "Bollinger squeeze present; waiting for directional release"


def _support_resistance_score(side: Side, price: float, zones: dict[str, float | None], atr: float) -> tuple[float, str | None]:
    tol = max(atr * 0.45, 1e-9)

    if side == "BUY" and zones.get("support") is not None:
        support = float(zones["support"])
        if abs(price - support) <= tol * 2.5:
            return 15.0, "Price is testing/holding support before possible Boom spike"
        if price > support and (price - support) <= tol * 5:
            return 9.0, "Price is close above support"

    if side == "SELL" and zones.get("resistance") is not None:
        resistance = float(zones["resistance"])
        if abs(price - resistance) <= tol * 2.5:
            return 15.0, "Price is testing/rejecting resistance before possible Crash drop"
        if price < resistance and (resistance - price) <= tol * 5:
            return 9.0, "Price is close below resistance"

    return 0.0, None


def _trend_score(df: pd.DataFrame, side: Side) -> tuple[float, list[str]]:
    row = df.iloc[-1]
    close = float(row["close"])
    ema20 = float(row["ema_20"])
    ema50 = float(row["ema_50"])
    ema200 = float(row["ema_200"])

    pts = 0.0
    notes: list[str] = []

    if side == "BUY":
        if close > ema200:
            pts += 14.0
            notes.append("Price is above EMA200, giving Boom buy structural support")
        if ema20 >= ema50:
            pts += 10.0
            notes.append("EMA20 is holding above/near EMA50")
        elif close > ema50:
            pts += 6.0
            notes.append("Pullback is still holding around EMA50")
    else:
        if close < ema200:
            pts += 14.0
            notes.append("Price is below EMA200, giving Crash sell structural pressure")
        if ema20 <= ema50:
            pts += 10.0
            notes.append("EMA20 is holding below/near EMA50")
        elif close < ema50:
            pts += 6.0
            notes.append("Pullback is still failing around EMA50")

    return pts, notes


def _higher_tf_score(dfs: dict[str, pd.DataFrame], side: Side) -> tuple[float, str | None]:
    pts = 0.0
    aligned: list[str] = []

    for tf in ("5m", "15m"):
        df = dfs.get(tf)
        if df is None or len(df) < 5 or not {"ema_20", "ema_50"}.issubset(df.columns):
            continue

        row = df.iloc[-1]
        ema20 = row.get("ema_20")
        ema50 = row.get("ema_50")
        close = row.get("close")

        if pd.isna(ema20) or pd.isna(ema50) or pd.isna(close):
            continue

        if side == "BUY" and float(close) >= float(ema50):
            pts += 4.0
            aligned.append(f"{tf} holding above EMA50")
        if side == "BUY" and float(ema20) >= float(ema50):
            pts += 3.0
            aligned.append(f"{tf} EMA20>=EMA50")
        if side == "SELL" and float(close) <= float(ema50):
            pts += 4.0
            aligned.append(f"{tf} failing below EMA50")
        if side == "SELL" and float(ema20) <= float(ema50):
            pts += 3.0
            aligned.append(f"{tf} EMA20<=EMA50")

    if not aligned:
        return 0.0, None
    return min(12.0, pts), "Higher timeframe context agrees: " + ", ".join(aligned[:3])


def _wick_rejection_score(df: pd.DataFrame, side: Side, atr: float, lookback: int = 8) -> tuple[float, str | None]:
    """Detect spring/upthrust style rejection near recent extremes."""
    if len(df) < max(lookback + 2, 10):
        return 0.0, None

    last = df.iloc[-1]
    recent = df.iloc[-lookback - 1 : -1]
    high = float(last["high"])
    low = float(last["low"])
    close = float(last["close"])
    open_ = float(last["open"])
    candle_range = max(high - low, 1e-9)
    body = abs(close - open_)

    if side == "BUY":
        swept_low = low < float(recent["low"].min())
        lower_wick = min(open_, close) - low
        closed_off_low = (close - low) / candle_range >= 0.55
        if swept_low and lower_wick >= body * 1.2 and lower_wick >= atr * 0.25 and closed_off_low:
            return 12.0, "False-break spring: swept recent low then closed back up"
        if lower_wick >= body * 1.5 and closed_off_low:
            return 7.0, "Bullish lower-wick rejection is forming"
    else:
        swept_high = high > float(recent["high"].max())
        upper_wick = high - max(open_, close)
        closed_off_high = (high - close) / candle_range >= 0.55
        if swept_high and upper_wick >= body * 1.2 and upper_wick >= atr * 0.25 and closed_off_high:
            return 12.0, "False-break upthrust: swept recent high then closed back down"
        if upper_wick >= body * 1.5 and closed_off_high:
            return 7.0, "Bearish upper-wick rejection is forming"

    return 0.0, None


def _drift_exhaustion_score(df: pd.DataFrame, side: Side, lookback: int = 8) -> tuple[float, str | None]:
    """Detect slowing opposite drift before target spike/drop."""
    if len(df) < lookback + 3:
        return 0.0, None

    closes = df["close"].iloc[-lookback:].astype(float)
    diffs = closes.diff().dropna()
    if len(diffs) < 4:
        return 0.0, None

    first_half = diffs.iloc[: len(diffs) // 2]
    second_half = diffs.iloc[len(diffs) // 2 :]

    if side == "BUY":
        drift_down = closes.iloc[-1] < closes.iloc[0]
        down_momentum_slowing = abs(second_half[second_half < 0].sum()) < abs(first_half[first_half < 0].sum())
        if drift_down and down_momentum_slowing:
            return 7.0, "Downward drift is slowing before possible Boom spike"
    else:
        drift_up = closes.iloc[-1] > closes.iloc[0]
        up_momentum_slowing = second_half[second_half > 0].sum() < first_half[first_half > 0].sum()
        if drift_up and up_momentum_slowing:
            return 7.0, "Upward drift is slowing before possible Crash drop"

    return 0.0, None


def _micro_break_confirmed(df: pd.DataFrame, side: Side, lookback: int = 3) -> tuple[bool, str]:
    """
    Trigger confirmation filter.

    Boom BUY trigger requires the latest close to break above the recent micro-high.
    Crash SELL trigger requires the latest close to break below the recent micro-low.
    """
    lookback = max(2, int(lookback))
    if len(df) < lookback + 2:
        return False, "Not enough candles for micro-break confirmation"

    last_close = float(df["close"].iloc[-1])
    recent = df.iloc[-lookback - 1 : -1]

    if side == "BUY":
        micro_high = float(recent["high"].max())
        if last_close > micro_high:
            return True, f"Confirmed: close broke above recent {lookback}-candle high"
        return False, f"Waiting for close above recent {lookback}-candle high"

    micro_low = float(recent["low"].min())
    if last_close < micro_low:
        return True, f"Confirmed: close broke below recent {lookback}-candle low"
    return False, f"Waiting for close below recent {lookback}-candle low"


def _spike_pressure_score(
    spike_ctx: SpikeContext,
    side: Side,
    trigger_spike_strength: float,
    trigger_tick_velocity_min: float,
) -> tuple[float, str | None, bool]:
    """Score target-direction spike pressure and decide if it can support a trigger."""
    target = _target_spike_direction(side)
    direction = str(spike_ctx.spike_direction or "none").lower()
    strength = max(0.0, float(spike_ctx.spike_strength or 0.0))
    velocity = max(0.0, float(spike_ctx.tick_velocity or 0.0))

    target_direction = direction == target
    strong_enough = strength >= trigger_spike_strength
    velocity_enough = velocity >= trigger_tick_velocity_min

    if target_direction and strong_enough and velocity_enough:
        return 16.0, "Target-direction spike pressure is active", True

    if target_direction and (strong_enough or velocity_enough):
        return 10.0, "Target-direction spike pressure is building", False

    if velocity_enough:
        return 4.0, "Tick velocity is elevated; waiting for correct direction", False

    return 0.0, None, False


def _anti_chase_penalty(df: pd.DataFrame, side: Side, atr: float) -> tuple[float, str | None]:
    """Reduce late entries after a huge candle has already travelled too far."""
    if len(df) < 3:
        return 0.0, None

    last = df.iloc[-1]
    body = abs(float(last["close"]) - float(last["open"]))
    candle_range = float(last["high"]) - float(last["low"])
    if atr <= 0:
        return 0.0, None

    if body >= atr * 2.2 or candle_range >= atr * 3.0:
        return -12.0, "Large candle already expanded; avoid chasing late entry"

    return 0.0, None


def _volatility_filter_score(df: pd.DataFrame, regime: str, vol_note: str | None) -> tuple[float, list[str], str | None]:
    pts = 0.0
    notes: list[str] = []
    warning = vol_note

    if regime.endswith("high_volatility") or (isinstance(vol_note, str) and "elevated" in vol_note.lower()):
        pts -= 10.0
        notes.append("High volatility haircut applied")
        warning = vol_note or "High volatility regime."

    if "atr_14" in df.columns and len(df["atr_14"].dropna()) >= 80:
        atr = float(df["atr_14"].iloc[-1])
        med = float(df["atr_14"].dropna().iloc[-80:].median())
        ratio = atr / max(med, 1e-9)
        if ratio > 2.5:
            pts -= 8.0
            notes.append(f"ATR is stretched vs recent median ({ratio:.2f}x)")
            warning = warning or "ATR is stretched; entry risk is higher."

    return pts, notes, warning


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


def evaluate_signal(
    symbol: str,
    df_1m: pd.DataFrame,
    spike_ctx: SpikeContext,
    min_score: float,
    now_epoch: float,
    *,
    signal_warmup_bars: int = 220,
    preparation_alerts_enabled: bool = True,
    trigger_alerts_enabled: bool = True,
    trigger_min_signal_score: float = 88.0,
    trigger_spike_strength: float = 1.4,
    trigger_tick_velocity_min: float = 0.05,
    entry_zone_atr_multiplier: float = 0.08,
    stop_loss_atr_multiplier: float = 2.8,
    take_profit_1_atr_multiplier: float = 3.5,
    take_profit_2_atr_multiplier: float = 6.0,
    min_risk_reward: float = 1.4,
    require_micro_break_for_trigger: bool = True,
    micro_break_lookback: int = 3,
) -> Signal | None:
    """
    Run the confluence engine on completed 1m OHLC history.

    This function is intentionally backwards-compatible with older callers:
    the newer strategy controls are keyword-only and have safe defaults.
    """
    side = allowed_side_for_symbol(symbol)
    if side is None:
        return None

    warmup = max(60, int(signal_warmup_bars))
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
    atr = max(float(row["atr_14"]), 1e-9)
    rsi = float(row["rsi_14"])

    regime, vol_note = classify_regime(df1)
    zones = detect_sr_zones(df1)

    reasons: list[str] = []
    score = 0.0

    pts, notes = _trend_score(df1, side)
    score += pts
    reasons.extend(notes)

    pts, note = _rsi_score(side, rsi)
    score += pts
    if note:
        reasons.append(note)

    if "macd_hist" in df1.columns:
        pts, note = _macd_score(df1["macd_hist"], side)
        score += pts
        if note:
            reasons.append(note)

    pts, note = _support_resistance_score(side, last_close, zones, atr)
    score += pts
    if note:
        reasons.append(note)

    pts, note = _bb_score(df1, side)
    score += pts
    if note:
        reasons.append(note)

    pts, note = _bollinger_squeeze_score(df1, side)
    score += pts
    if note:
        reasons.append(note)

    pts, note = _wick_rejection_score(df1, side, atr)
    score += pts
    if note:
        reasons.append(note)

    pts, note = _drift_exhaustion_score(df1, side)
    score += pts
    if note:
        reasons.append(note)

    rej_pts, rej_note = candle_rejection_score(side, df1)
    score += rej_pts
    if rej_note:
        reasons.append(rej_note)

    pts, note = _higher_tf_score(dfs, side)
    score += pts
    if note:
        reasons.append(note)

    pts, note, spike_pressure_confirmed = _spike_pressure_score(
        spike_ctx=spike_ctx,
        side=side,
        trigger_spike_strength=trigger_spike_strength,
        trigger_tick_velocity_min=trigger_tick_velocity_min,
    )
    score += pts
    if note:
        reasons.append(note)

    penalty, note = _anti_chase_penalty(df1, side, atr)
    score += penalty
    if note:
        reasons.append(note)

    vol_pts, vol_notes, warning = _volatility_filter_score(df1, regime, vol_note)
    score += vol_pts
    reasons.extend(vol_notes)

    micro_ok, micro_note = _micro_break_confirmed(df1, side, lookback=micro_break_lookback)

    if micro_ok:
        score += 12.0
        reasons.append(micro_note)
    else:
        # Do not punish PREP too hard, but prevent early TRIGGER naming.
        reasons.append(micro_note)

    score = float(max(0.0, min(100.0, score)))

    entry_low, entry_high, stop_loss, tp1, tp2, rr = _build_levels(
        side=side,
        last_close=last_close,
        atr=atr,
        entry_zone_atr_multiplier=entry_zone_atr_multiplier,
        stop_loss_atr_multiplier=stop_loss_atr_multiplier,
        take_profit_1_atr_multiplier=take_profit_1_atr_multiplier,
        take_profit_2_atr_multiplier=take_profit_2_atr_multiplier,
    )

    if rr < float(min_risk_reward):
        return None

    trigger_confirmed = (
        trigger_alerts_enabled
        and score >= float(trigger_min_signal_score)
        and spike_pressure_confirmed
        and (micro_ok or not require_micro_break_for_trigger)
    )

    if trigger_confirmed:
        stage: AlertStage = "TRIGGER"
    elif preparation_alerts_enabled and score >= float(min_score):
        stage = "PREP"
    else:
        return None

    # Keep Telegram reasons readable and focused.
    cleaned_reasons: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if not reason or reason in seen:
            continue
        cleaned_reasons.append(reason)
        seen.add(reason)

    ema20 = _safe_float(row.get("ema_20"))
    ema50 = _safe_float(row.get("ema_50"))
    ema200 = _safe_float(row.get("ema_200"))
    macd_hist = _safe_float(row.get("macd_hist"))
    bb_lower = _safe_float(row.get("bb_lower"))
    bb_upper = _safe_float(row.get("bb_upper"))
    bb_mid = _safe_float(row.get("bb_mid"))
    close_minus_ema20_atr = (last_close - ema20) / atr if atr else 0.0
    close_minus_ema50_atr = (last_close - ema50) / atr if atr else 0.0
    close_minus_ema200_atr = (last_close - ema200) / atr if atr else 0.0
    bb_position = 0.5
    if bb_upper > bb_lower:
        bb_position = (last_close - bb_lower) / max(bb_upper - bb_lower, 1e-9)

    support = zones.get("support")
    resistance = zones.get("resistance")
    support_distance_atr = None if support is None else (last_close - float(support)) / atr
    resistance_distance_atr = None if resistance is None else (float(resistance) - last_close) / atr

    features: dict[str, Any] = {
        "symbol": symbol,
        "symbol_type": "BOOM" if _is_boom(symbol) else "CRASH",
        "side": side,
        "alert_stage": stage,
        "score": score,
        "rsi_14": rsi,
        "macd_hist": macd_hist,
        "atr_14": atr,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "ema20_gt_ema50": 1 if ema20 > ema50 else 0,
        "ema50_gt_ema200": 1 if ema50 > ema200 else 0,
        "close_minus_ema20_atr": close_minus_ema20_atr,
        "close_minus_ema50_atr": close_minus_ema50_atr,
        "close_minus_ema200_atr": close_minus_ema200_atr,
        "bb_position": float(bb_position),
        "bb_mid": bb_mid,
        "bb_lower": bb_lower,
        "bb_upper": bb_upper,
        "support_distance_atr": support_distance_atr,
        "resistance_distance_atr": resistance_distance_atr,
        "spike_direction": spike_ctx.spike_direction,
        "spike_strength": float(spike_ctx.spike_strength),
        "tick_velocity": float(spike_ctx.tick_velocity),
        "spike_pressure_confirmed": 1 if spike_pressure_confirmed else 0,
        "micro_break_confirmed": 1 if micro_ok else 0,
        "regime": regime,
        "risk_reward": rr,
        "entry_zone_atr_multiplier": float(entry_zone_atr_multiplier),
        "stop_loss_atr_multiplier": float(stop_loss_atr_multiplier),
        "take_profit_1_atr_multiplier": float(take_profit_1_atr_multiplier),
        "take_profit_2_atr_multiplier": float(take_profit_2_atr_multiplier),
        "hour_utc": int(pd.to_datetime(now_epoch, unit="s", utc=True).hour),
        "dayofweek_utc": int(pd.to_datetime(now_epoch, unit="s", utc=True).dayofweek),
    }

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
        reasons=cleaned_reasons,
        volatility_warning=warning,
        regime=regime,
        timestamp_epoch=float(now_epoch),
        alert_stage=stage,
        features=features,
    )
