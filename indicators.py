"""
Technical indicators for OHLC dataframes implemented with pandas/numpy only.

Avoids heavyweight optional deps so installs stay painless on newer Python builds.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    cols = {"open", "high", "low", "close"}
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing OHLC columns: {missing}")
    out = df.copy()
    out = out.sort_index()
    return out


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0)).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _atr_wilder(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr_parts = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_parts.max(axis=1)
    atr = true_range.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    return atr



def _stochastic_oscillator(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smoothing: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Slow stochastic oscillator (%K/%D).

    %K = 100 * (close - lowest_low) / (highest_high - lowest_low).
    A small smoothing is applied to make it usable for Boom/Crash pullback timing.
    """
    k_period = max(3, int(k_period))
    d_period = max(1, int(d_period))
    smoothing = max(1, int(smoothing))

    low_min = df["low"].rolling(k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(k_period, min_periods=k_period).max()
    raw_k = 100.0 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    slow_k = raw_k.rolling(smoothing, min_periods=smoothing).mean()
    slow_d = slow_k.rolling(d_period, min_periods=d_period).mean()
    return slow_k.clip(0, 100), slow_d.clip(0, 100)


def attach_core_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds EMA(20/50/200), RSI(14), MACD histogram, Bollinger Bands, ATR(14).

    Expects DatetimeIndex and OHLC columns.
    """
    ohlc = ensure_ohlc(df)
    close = ohlc["close"]

    ohlc["ema_20"] = _ema(close, 20)
    ohlc["ema_50"] = _ema(close, 50)
    ohlc["ema_200"] = _ema(close, 200)

    ohlc["rsi_14"] = _rsi_wilder(close, 14)

    stoch_k, stoch_d = _stochastic_oscillator(ohlc, 14, 3, 3)
    ohlc["stoch_k"] = stoch_k
    ohlc["stoch_d"] = stoch_d

    macd_line = _ema(close, 12) - _ema(close, 26)
    signal_line = _ema(macd_line, 9)
    ohlc["macd"] = macd_line
    ohlc["macd_signal"] = signal_line
    ohlc["macd_hist"] = macd_line - signal_line

    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    ohlc["bb_mid"] = mid
    ohlc["bb_upper"] = mid + 2.0 * std
    ohlc["bb_lower"] = mid - 2.0 * std

    ohlc["atr_14"] = _atr_wilder(ohlc[["high", "low", "close"]], 14)

    return ohlc


def bollinger_bandwidth(df: pd.DataFrame) -> pd.Series:
    """Normalized BB width — squeeze detection helper."""
    if not {"bb_lower", "bb_upper", "bb_mid"}.issubset(df.columns):
        return pd.Series(np.nan, index=df.index)
    mid = df["bb_mid"].replace(0, np.nan)
    width = (df["bb_upper"] - df["bb_lower"]) / mid
    return width


def detect_sr_zones(
    df: pd.DataFrame,
    lookback: int = 80,
    touch_tolerance_atr_mult: float = 0.25,
) -> dict[str, float | None]:
    """
    Lightweight support / resistance notion:

    - support ~ recent swing low cluster near latest price
    - resistance ~ recent swing high cluster near latest price
    """
    out: dict[str, float | None] = {"support": None, "resistance": None}
    if len(df) < max(lookback, 25):
        return out

    window = df.iloc[-lookback:]
    atr = float(window["atr_14"].iloc[-1]) if "atr_14" in window.columns else None
    if atr is None or np.isnan(atr):
        atr = float((window["high"] - window["low"]).tail(14).mean())

    tol = atr * touch_tolerance_atr_mult
    last = float(window["close"].iloc[-1])

    lows = window["low"].rolling(5).min().dropna()
    highs = window["high"].rolling(5).max().dropna()
    if lows.empty or highs.empty:
        return out

    candidates = lows[lows < last]
    support = float(candidates.iloc[-1]) if not candidates.empty else None

    candidates_r = highs[highs > last]
    resistance = float(candidates_r.iloc[-1]) if not candidates_r.empty else None

    if support is not None and abs(last - support) > tol * 8:
        support = None
    if resistance is not None and abs(resistance - last) > tol * 8:
        resistance = None

    out["support"] = support
    out["resistance"] = resistance
    return out


def classify_regime(df: pd.DataFrame) -> tuple[str, str | None]:
    """
    Returns (regime_label, note).

      - uptrend / downtrend from EMA stack & EMA50 slope
      - ranging if EMA20/50 distance is small vs ATR
      - volatility from ATR vs medium average
    """
    if len(df) < 220:
        return "insufficient_data", "Need more candles for stable EMA200/regime."

    row = df.iloc[-1]
    ema20, ema50, ema200 = row.get("ema_20"), row.get("ema_50"), row.get("ema_200")
    atr = row.get("atr_14")
    if any(pd.isna(v) for v in (ema20, ema50, ema200, atr)):
        return "insufficient_data", "Indicators not fully formed yet."

    ema_stack_up = ema20 > ema50 > ema200
    ema_stack_down = ema20 < ema50 < ema200

    ema50_slope = float(df["ema_50"].iloc[-1] - df["ema_50"].iloc[-6])

    atr_series = df["atr_14"].dropna()
    atr_ma = float(atr_series.iloc[-50:].mean()) if len(atr_series) >= 50 else float(
        atr_series.mean()
    )
    hi_vol = atr > atr_ma * 1.35
    lo_vol = atr < atr_ma * 0.75

    ema_sep = abs(ema20 - ema50)
    ranging = ema_sep < atr * 0.35

    vol_note = None
    if hi_vol:
        vol_note = "ATR elevated vs recent average (high volatility)."
    elif lo_vol:
        vol_note = "ATR compressed vs recent average (low volatility)."

    if ranging and not hi_vol:
        return "ranging", vol_note
    if ema_stack_up and ema50_slope >= 0:
        label = "uptrend"
    elif ema_stack_down and ema50_slope <= 0:
        label = "downtrend"
    elif ema_stack_up:
        label = "uptrend"
    elif ema_stack_down:
        label = "downtrend"
    else:
        label = "transition"

    label2 = label
    if hi_vol:
        label2 = f"{label}_high_volatility"
    elif lo_vol:
        label2 = f"{label}_low_volatility"

    return label2, vol_note


def last_bar_spike_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Body / wick stats for the most recent closed bar, normalized by ATR."""
    row = df.iloc[-1]
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    atr = float(row["atr_14"]) if "atr_14" in row and not pd.isna(row["atr_14"]) else max(
        h - l, 1e-9
    )
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    direction: Literal[-1, 0, 1]
    if c > o:
        direction = 1
    elif c < o:
        direction = -1
    else:
        direction = 0

    return {
        "body": body,
        "body_atr": body / max(atr, 1e-9),
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "wick_sum_atr": (upper_wick + lower_wick) / max(atr, 1e-9),
        "direction": float(direction),
    }


def candle_rejection_score(side: Literal["BUY", "SELL"], df: pd.DataFrame) -> tuple[float, str | None]:
    """Simple hammer / shooting-star style heuristic on the closing bar."""
    m = last_bar_spike_metrics(df)
    note = None
    if side == "BUY":
        strength = min(
            10.0,
            m["lower_wick"] / max(1e-9, abs(m["body"]) + 1e-9) * 3.0,
        )
        if m["direction"] > 0:
            strength += 2.0
        if m["lower_wick"] > m["upper_wick"] * 1.4:
            note = "Bullish rejection wick vs body"
        return float(min(10.0, strength)), note
    strength = min(
        10.0,
        m["upper_wick"] / max(1e-9, abs(m["body"]) + 1e-9) * 3.0,
    )
    if m["direction"] < 0:
        strength += 2.0
    if m["upper_wick"] > m["lower_wick"] * 1.4:
        note = "Bearish rejection wick vs body"
    return float(min(10.0, strength)), note
