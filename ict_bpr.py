"""
ICT Balanced Price Range (BPR) helpers.

This module detects simple Fair Value Gaps (FVGs), then finds Balanced Price
Ranges as overlaps between opposite-direction FVG zones. It is intentionally
conservative and lightweight for dashboard/signal context, not a standalone
trading system.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import pandas as pd

BprBias = Literal["bullish", "bearish", "none"]


@dataclass(frozen=True)
class BprZone:
    bias: BprBias
    low: float
    high: float
    start_epoch: float
    end_epoch: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BprContext:
    status: str
    bias: BprBias
    aligned: bool
    zone_low: float | None
    zone_high: float | None
    distance_atr: float | None
    location: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resample_to_h4(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Build H4 OHLC candles from 1m candles. Expects a UTC DatetimeIndex."""
    if df_1m.empty:
        return pd.DataFrame()
    out = (
        df_1m[["open", "high", "low", "close"]]
        .resample("4h", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    return out


def detect_fvg_zones(df: pd.DataFrame, lookback: int = 120) -> list[BprZone]:
    """
    Detect simple 3-candle FVG zones.

    Bullish FVG: candle[i].low > candle[i-2].high, gap is [i-2 high, i low].
    Bearish FVG: candle[i].high < candle[i-2].low, gap is [i high, i-2 low].
    """
    if df.empty or len(df) < 3:
        return []

    recent = df.tail(max(3, int(lookback))).copy()
    zones: list[BprZone] = []

    for i in range(2, len(recent)):
        c0 = recent.iloc[i - 2]
        c2 = recent.iloc[i]
        ts = recent.index[i]
        prev_ts = recent.index[i - 2]

        c0_high = float(c0["high"])
        c0_low = float(c0["low"])
        c2_high = float(c2["high"])
        c2_low = float(c2["low"])

        if c2_low > c0_high:
            zones.append(
                BprZone(
                    bias="bullish",
                    low=c0_high,
                    high=c2_low,
                    start_epoch=float(prev_ts.timestamp()),
                    end_epoch=float(ts.timestamp()),
                    source="H4 bullish FVG",
                )
            )

        if c2_high < c0_low:
            zones.append(
                BprZone(
                    bias="bearish",
                    low=c2_high,
                    high=c0_low,
                    start_epoch=float(prev_ts.timestamp()),
                    end_epoch=float(ts.timestamp()),
                    source="H4 bearish FVG",
                )
            )

    return zones


def detect_bpr_zones(df_h4: pd.DataFrame, lookback: int = 120) -> list[BprZone]:
    """
    Detect BPR zones as overlaps between opposing FVGs.

    Bias is assigned to the newer FVG in the overlap. This makes the zone useful
    as current context: a newer bullish FVG overlapping an earlier bearish FVG is
    treated as bullish BPR, and vice versa.
    """
    fvgs = detect_fvg_zones(df_h4, lookback=lookback)
    zones: list[BprZone] = []

    for idx, newer in enumerate(fvgs):
        for older in fvgs[:idx]:
            if newer.bias == older.bias:
                continue
            low = max(float(newer.low), float(older.low))
            high = min(float(newer.high), float(older.high))
            if high <= low:
                continue
            zones.append(
                BprZone(
                    bias=newer.bias,
                    low=low,
                    high=high,
                    start_epoch=min(float(newer.start_epoch), float(older.start_epoch)),
                    end_epoch=max(float(newer.end_epoch), float(older.end_epoch)),
                    source=f"H4 BPR overlap: {older.source} + {newer.source}",
                )
            )

    # Newest zones are generally more useful.
    zones.sort(key=lambda z: z.end_epoch, reverse=True)
    return zones


def bpr_context_for_signal(
    df_1m: pd.DataFrame,
    side: str,
    price: float,
    atr: float,
    *,
    enabled: bool = True,
    lookback_candles: int = 120,
    max_distance_atr: float = 1.5,
) -> BprContext:
    """Return a simple H4 BPR alignment context for the signal side."""
    if not enabled:
        return BprContext("OFF", "none", False, None, None, None, "disabled", "H4 BPR disabled")

    if df_1m.empty or len(df_1m) < 240:
        return BprContext("NO_DATA", "none", False, None, None, None, "not_enough_h4", "Not enough candles for stable H4 BPR")

    df_h4 = resample_to_h4(df_1m)
    if df_h4.empty or len(df_h4) < 8:
        return BprContext("NO_DATA", "none", False, None, None, None, "not_enough_h4", "Not enough H4 candles for BPR")

    zones = detect_bpr_zones(df_h4, lookback=lookback_candles)
    target_bias: BprBias = "bullish" if str(side).upper() == "BUY" else "bearish"
    target_zones = [z for z in zones if z.bias == target_bias]

    if not target_zones:
        return BprContext("NO_ZONE", target_bias, False, None, None, None, "no_zone", f"No clean {target_bias} H4 BPR nearby")

    atr = max(float(atr), 1e-9)
    price = float(price)

    def distance_to_zone(z: BprZone) -> float:
        if z.low <= price <= z.high:
            return 0.0
        if price < z.low:
            return z.low - price
        return price - z.high

    zone = min(target_zones, key=distance_to_zone)
    dist_price = distance_to_zone(zone)
    dist_atr = dist_price / atr

    if zone.low <= price <= zone.high:
        return BprContext("ALIGNED", target_bias, True, zone.low, zone.high, 0.0, "inside", f"Price is inside {target_bias} H4 BPR")

    # For Boom BUY, slightly above a bullish BPR can still be a valid support reaction.
    if target_bias == "bullish" and price > zone.high and dist_atr <= max_distance_atr:
        return BprContext("ALIGNED", target_bias, True, zone.low, zone.high, dist_atr, "above_near", f"Price is {dist_atr:.2f} ATR above bullish H4 BPR")

    # For Crash SELL, slightly below a bearish BPR can still be a valid resistance reaction.
    if target_bias == "bearish" and price < zone.low and dist_atr <= max_distance_atr:
        return BprContext("ALIGNED", target_bias, True, zone.low, zone.high, dist_atr, "below_near", f"Price is {dist_atr:.2f} ATR below bearish H4 BPR")

    if dist_atr <= max_distance_atr:
        return BprContext("NEAR", target_bias, False, zone.low, zone.high, dist_atr, "near_wrong_side", f"Near {target_bias} H4 BPR but not in ideal reaction side")

    return BprContext("FAR", target_bias, False, zone.low, zone.high, dist_atr, "far", f"Price is {dist_atr:.2f} ATR away from nearest {target_bias} H4 BPR")


def bpr_zones_for_dashboard(df_1m: pd.DataFrame, lookback_candles: int = 120) -> list[dict[str, Any]]:
    """Return H4 BPR zones as dictionaries for dashboard plotting."""
    if df_1m.empty:
        return []
    df_h4 = resample_to_h4(df_1m)
    if df_h4.empty:
        return []
    return [z.to_dict() for z in detect_bpr_zones(df_h4, lookback=lookback_candles)]
