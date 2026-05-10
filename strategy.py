"""
Score-based confluence strategy for Deriv Crash / Boom synthetic indices.

This is educational / research code — not investment advice.
Scores are capped at 100; default minimum to alert is 75 via config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

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


def evaluate_signal(
    symbol: str,
    df_1m: pd.DataFrame,
    spike_ctx: SpikeContext,
    min_score: float,
    now_epoch: float,
) -> Signal | None:
    """
    Run the confluence engine on the latest completed 1m history.

    `df_1m` must use a DatetimeIndex (UTC) and OHLC columns.
    """
    if df_1m.empty or len(df_1m) < 220:
        return None

    dfs_raw = build_multi_timeframe(df_1m)
    dfs = _tf_attach(dfs_raw)
    df1 = dfs["1m"]
    if df1.empty or len(df1) < 220:
        return None

    regime, vol_note = classify_regime(df1)
    zones = detect_sr_zones(df1)
    atr = float(df1["atr_14"].iloc[-1])
    last_close = float(df1["close"].iloc[-1])

    boom = symbol.upper().startswith("BOOM")

    candidates: list[tuple[Literal["BUY", "SELL"], float, list[str], str | None]] = []

    for side in ("BUY", "SELL"):
        reasons: list[str] = []
        score = 0.0

        ema20 = float(df1["ema_20"].iloc[-1])
        ema50 = float(df1["ema_50"].iloc[-1])
        ema200 = float(df1["ema_200"].iloc[-1])
        rsi = float(df1["rsi_14"].iloc[-1])

        if side == "BUY":
            if last_close > ema200:
                score += 18.0
                reasons.append("Price above EMA200 (long-term bias)")
            if ema20 > ema50:
                score += 17.0
                reasons.append("EMA20 above EMA50")
        else:
            if last_close < ema200:
                score += 18.0
                reasons.append("Price below EMA200 (long-term bias)")
            if ema20 < ema50:
                score += 17.0
                reasons.append("EMA20 below EMA50")

        r_pts, r_note = _rsi_zone_score(side, rsi)
        score += r_pts
        if r_note:
            reasons.append(r_note)

        if "macd_hist" in df1.columns:
            mh_pts, mh_note = _macd_hist_improving(df1["macd_hist"], side)
            score += mh_pts
            if mh_note:
                reasons.append(mh_note)

        sr_pts, sr_note = _support_resistance_score(side, last_close, zones, atr)
        score += sr_pts
        if sr_note:
            reasons.append(sr_note)

        bb_pts, bb_note = _bb_confluence(df1, side)
        score += bb_pts
        if bb_note:
            reasons.append(bb_note)

        rej_pts, rej_note = candle_rejection_score(side, df1)
        score += rej_pts
        if rej_note:
            reasons.append(rej_note)

        tf_pts, tf_note = _tf_alignment_bonus(dfs, side)
        score += tf_pts
        if tf_note:
            reasons.append(tf_note)

        sp_pen, sp_note = _spike_penalty(symbol, side, spike_ctx, boom=boom)
        score += sp_pen
        if sp_note:
            reasons.append(sp_note)

        # Volatility-aware cap: very hot regimes reduce certainty
        vol_warn = vol_note
        if regime.endswith("high_volatility") or (
            isinstance(vol_warn, str) and "elevated" in vol_warn.lower()
        ):
            score -= 10.0
            reasons.append("High volatility haircut applied — consider smaller size")

        score = float(max(0.0, min(100.0, score)))

        if score >= min_score:
            candidates.append((side, score, reasons, vol_warn))

    if not candidates:
        return None

    # Pick stronger side if both pass threshold
    candidates.sort(key=lambda x: x[1], reverse=True)
    side, score, reasons, vol_warn = candidates[0]

    last_close = float(df1["close"].iloc[-1])
    entry_low = last_close - atr * 0.15
    entry_high = last_close + atr * 0.15
    if side == "BUY":
        sl = entry_low - atr * 2.0
        tp1 = entry_high + atr * 2.5
        tp2 = entry_high + atr * 4.0
        risk = entry_low - sl
        reward = tp1 - entry_low
    else:
        sl = entry_high + atr * 2.0
        tp1 = entry_low - atr * 2.5
        tp2 = entry_low - atr * 4.0
        risk = sl - entry_high
        reward = entry_high - tp1
    rr = float(max(0.0, reward / max(risk, 1e-9)))

    return Signal(
        symbol=symbol,
        side=side,
        score=score,
        timeframe="1m (multi-TF context: 5m/15m)",
        entry_zone_low=float(entry_low),
        entry_zone_high=float(entry_high),
        stop_loss=float(sl),
        take_profit_1=float(tp1),
        take_profit_2=float(tp2),
        risk_reward=rr,
        reasons=reasons,
        volatility_warning=vol_warn,
        regime=regime,
        timestamp_epoch=float(now_epoch),
    )
