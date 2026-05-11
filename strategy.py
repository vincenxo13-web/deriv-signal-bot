"""
Signal-only Boom/Crash strategy.

Directional rules requested:
- BOOM symbols: BUY only, aiming to anticipate upward spike/sniper entries.
- CRASH symbols: SELL only, aiming to anticipate downward spike/sniper entries.

This is research code, not financial advice.
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
)


@dataclass
class SpikeContext:
    last_spike_epoch: float | None
    spike_direction: str  # "up" | "down" | "none"
    spike_strength: float
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


def _rsi_score(side: Literal["BUY", "SELL"], rsi: float) -> tuple[float, str | None]:
    if np.isnan(rsi):
        return 0.0, None
    if side == "BUY":
        if 38 <= rsi <= 58:
            return 14.0, "RSI is in a buy-preparation zone, not overextended"
        if 30 <= rsi < 38:
            return 10.0, "RSI is low; possible spring setup before Boom spike"
        if 58 < rsi <= 66:
            return 5.0, "RSI positive but entry needs patience"
    else:
        if 42 <= rsi <= 62:
            return 14.0, "RSI is in a sell-preparation zone, not oversold"
        if 62 < rsi <= 72:
            return 10.0, "RSI is high; possible exhaustion before Crash drop"
        if 34 <= rsi < 42:
            return 5.0, "RSI weak but entry needs patience"
    return 0.0, None


def _macd_score(series: pd.Series, side: Literal["BUY", "SELL"]) -> tuple[float, str | None]:
    s = series.dropna()
    if len(s) < 6:
        return 0.0, None
    last, prev, older = float(s.iloc[-1]), float(s.iloc[-2]), float(s.iloc[-6])
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


def _bb_score(df: pd.DataFrame, side: Literal["BUY", "SELL"]) -> tuple[float, str | None]:
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

    if side == "BUY":
        if close <= lower + (mid - lower) * 0.45:
            return 12.0, "Price is near lower Bollinger area: sniper buy zone"
        if close > mid and last_w > med_w:
            return 7.0, "Price reclaimed mid-BB with volatility expanding"
    else:
        if close >= upper - (upper - mid) * 0.45:
            return 12.0, "Price is near upper Bollinger area: sniper sell zone"
        if close < mid and last_w > med_w:
            return 7.0, "Price lost mid-BB with volatility expanding"
    return 0.0, None


def _support_resistance_score(
    side: Literal["BUY", "SELL"],
    price: float,
    zones: dict[str, float | None],
    atr: float,
) -> tuple[float, str | None]:
    tol = max(atr * 0.45, 1e-9)
    if side == "BUY" and zones.get("support") is not None:
        support = float(zones["support"])
        if abs(price - support) <= tol * 2.5:
            return 15.0, "Price is testing/holding support before potential Boom spike"
        if price > support and (price - support) <= tol * 5:
            return 9.0, "Price is close above support"
    if side == "SELL" and zones.get("resistance") is not None:
        resistance = float(zones["resistance"])
        if abs(price - resistance) <= tol * 2.5:
            return 15.0, "Price is testing/rejecting resistance before potential Crash drop"
        if price < resistance and (resistance - price) <= tol * 5:
            return 9.0, "Price is close below resistance"
    return 0.0, None


def _trend_score(df: pd.DataFrame, side: Literal["BUY", "SELL"]) -> tuple[float, list[str]]:
    row = df.iloc[-1]
    close = float(row["close"])
    ema20 = float(row["ema_20"])
    ema50 = float(row["ema_50"])
    ema200 = float(row["ema_200"])
    pts = 0.0
    notes: list[str] = []

    if side == "BUY":
        # Boom sniper: prefer pullback while structure is not fully broken.
        if close > ema200:
            pts += 14.0
            notes.append("Price is above EMA200, so buy setup has structural support")
        if ema20 >= ema50:
            pts += 10.0
            notes.append("EMA20 is holding above/near EMA50")
        elif close > ema50:
            pts += 6.0
            notes.append("Pullback is still holding around EMA50")
    else:
        # Crash sniper: prefer rejection while structure is not fully broken.
        if close < ema200:
            pts += 14.0
            notes.append("Price is below EMA200, so sell setup has structural pressure")
        if ema20 <= ema50:
            pts += 10.0
            notes.append("EMA20 is holding below/near EMA50")
        elif close < ema50:
            pts += 6.0
            notes.append("Pullback is still failing around EMA50")
    return pts, notes


def _higher_tf_score(dfs: dict[str, pd.DataFrame], side: Literal["BUY", "SELL"]) -> tuple[float, str | None]:
    pts = 0.0
    aligned: list[str] = []
    for tf in ("5m", "15m"):
        df = dfs.get(tf)
        if df is None or len(df) < 5 or not {"ema_20", "ema_50"}.issubset(df.columns):
            continue
        row = df.iloc[-1]
        ema20, ema50 = row.get("ema_20"), row.get("ema_50")
        if pd.isna(ema20) or pd.isna(ema50):
            continue
        if side == "BUY" and float(ema20) >= float(ema50):
            pts += 5.0
            aligned.append(f"{tf} bullish")
        elif side == "SELL" and float(ema20) <= float(ema50):
            pts += 5.0
            aligned.append(f"{tf} bearish")
    if not aligned:
        return 0.0, None
    return min(10.0, pts), "Higher timeframe context agrees: " + ", ".join(aligned)


def _spike_context_score(side: Literal["BUY", "SELL"], ctx: SpikeContext) -> tuple[float, str | None]:
    # We want to enter before the move, not after the spike has already exploded.
    if ctx.last_spike_epoch is None or ctx.spike_direction == "none":
        if ctx.tick_velocity > 0:
            return 3.0, "Tick activity is present; watching for spike trigger"
        return 0.0, None

    if side == "BUY" and ctx.spike_direction == "up" and ctx.spike_strength > 1.0:
        return -18.0, "Recent Boom up-spike already fired; avoid chasing"
    if side == "SELL" and ctx.spike_direction == "down" and ctx.spike_strength > 1.0:
        return -18.0, "Recent Crash down-spike already fired; avoid chasing"
    if ctx.spike_strength > 0.55:
        return -8.0, "Fresh spike volatility detected; signal reduced for safety"
    return 0.0, None


def allowed_side_for_symbol(symbol: str) -> Literal["BUY", "SELL"] | None:
    if _is_boom(symbol):
        return "BUY"
    if _is_crash(symbol):
        return "SELL"
    return None


def evaluate_signal(
    symbol: str,
    df_1m: pd.DataFrame,
    spike_ctx: SpikeContext,
    min_score: float,
    now_epoch: float,
    warmup_bars: int = 120,
) -> Signal | None:
    """Evaluate one symbol. Boom returns BUY only; Crash returns SELL only."""
    side = allowed_side_for_symbol(symbol)
    if side is None:
        return None
    if df_1m.empty or len(df_1m) < warmup_bars:
        return None

    dfs = _attach_all(build_multi_timeframe(df_1m))
    df1 = dfs["1m"]
    if df1.empty or len(df1) < warmup_bars:
        return None

    required = {"ema_20", "ema_50", "ema_200", "rsi_14", "atr_14"}
    if not required.issubset(df1.columns):
        return None

    row = df1.iloc[-1]
    last_close = float(row["close"])
    atr = float(row["atr_14"])
    rsi = float(row["rsi_14"])
    zones = detect_sr_zones(df1)
    regime, vol_note = classify_regime(df1)

    score = 0.0
    reasons: list[str] = []

    pts, notes = _trend_score(df1, side)
    score += pts
    reasons.extend(notes)

    for pts, note in (
        _rsi_score(side, rsi),
        _macd_score(df1["macd_hist"], side) if "macd_hist" in df1.columns else (0.0, None),
        _support_resistance_score(side, last_close, zones, atr),
        _bb_score(df1, side),
        candle_rejection_score(side, df1),
        _higher_tf_score(dfs, side),
        _spike_context_score(side, spike_ctx),
    ):
        score += pts
        if note:
            reasons.append(note)

    if regime.endswith("high_volatility") or (vol_note and "elevated" in vol_note.lower()):
        score -= 8.0
        reasons.append("High volatility haircut applied; wait for clean entry confirmation")

    score = float(max(0.0, min(100.0, score)))
    if score < min_score:
        return None

    entry_low = last_close - atr * 0.18
    entry_high = last_close + atr * 0.18
    if side == "BUY":
        stop_loss = entry_low - atr * 1.8
        take_profit_1 = entry_high + atr * 2.2
        take_profit_2 = entry_high + atr * 3.6
        risk = entry_low - stop_loss
        reward = take_profit_1 - entry_low
    else:
        stop_loss = entry_high + atr * 1.8
        take_profit_1 = entry_low - atr * 2.2
        take_profit_2 = entry_low - atr * 3.6
        risk = stop_loss - entry_high
        reward = entry_high - take_profit_1

    return Signal(
        symbol=symbol,
        side=side,
        score=score,
        timeframe="1m setup with 5m/15m context",
        entry_zone_low=float(entry_low),
        entry_zone_high=float(entry_high),
        stop_loss=float(stop_loss),
        take_profit_1=float(take_profit_1),
        take_profit_2=float(take_profit_2),
        risk_reward=float(max(0.0, reward / max(risk, 1e-9))),
        reasons=reasons,
        volatility_warning=vol_note,
        regime=regime,
        timestamp_epoch=float(now_epoch),
    )
