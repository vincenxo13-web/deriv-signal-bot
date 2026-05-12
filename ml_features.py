"""
ML feature helpers for the Deriv Boom/Crash signal bot.

Stage 1 is deliberately safe: we only log ML-ready feature snapshots.
No signal is blocked or changed by ML yet.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


def _clean_value(value: Any) -> Any:
    """Make feature values JSON/CSV friendly."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return str(value)


def normalise_features(features: Mapping[str, Any]) -> dict[str, Any]:
    """Return a cleaned dictionary suitable for SQLite JSON storage."""
    return {str(key): _clean_value(value) for key, value in dict(features).items()}


def outcome_to_success(outcome_status: str) -> int | None:
    """Convert stored outcome to ML binary target.

    Returns:
      1 for wins, 0 for losses/expired, None for open/watch-only/unknown.
    """
    status = str(outcome_status or "").upper()
    if status.startswith("WIN"):
        return 1
    if status.startswith("LOSS") or status == "EXPIRED":
        return 0
    return None
