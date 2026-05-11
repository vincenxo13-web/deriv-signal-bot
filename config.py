"""
Runtime settings for the Deriv Crash/Boom signal-only bot.

Secrets should live in Railway Variables or local .env, never in GitHub.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path:
    return Path(__file__).resolve().parent


load_dotenv(_project_root() / ".env")


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _split_symbols(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def get_data_dir() -> Path:
    """Persistent folder. Railway volume should mount to /app/data."""
    return Path(os.getenv("RAILWAY_VOLUME_MOUNT_PATH", os.getenv("DATA_DIR", "data")))


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded once."""

    # Deriv public market stream. Signal-only mode does not need a Deriv token.
    deriv_app_id: int
    symbols: tuple[str, ...]
    mode: str

    # Telegram delivery.
    telegram_bot_token: str | None
    telegram_chat_id: str | None
    telegram_parse_mode: str
    notify_on_start: bool

    # Signal engine.
    min_signal_score: float
    signal_warmup_bars: int
    signal_timezone: str

    # Stream/storage.
    tick_sample_seconds: int
    data_db_path: Path
    dashboard_refresh_seconds: int

    # Alert throttles / risk filters.
    max_signals_per_symbol_per_hour: int
    min_minutes_between_signals_same_symbol: int
    max_spread_points_estimate: float
    estimated_spread_points: float
    extreme_atr_ratio_threshold: float
    cooldown_after_spike_seconds: int

    # Spike detection thresholds.
    spike_body_atr_multiplier_boom: float
    spike_body_atr_multiplier_crash: float
    spike_tick_velocity_threshold_boom: float
    spike_tick_velocity_threshold_crash: float

    log_level: str


_SETTINGS_INSTANCE: Settings | None = None


def get_settings() -> Settings:
    """Singleton-style settings accessor."""
    global _SETTINGS_INSTANCE
    if _SETTINGS_INSTANCE is None:
        data_dir = get_data_dir()
        db_path = data_dir / "deriv_signals.db"

        symbols = tuple(_split_symbols(os.getenv("SYMBOLS")))
        if not symbols:
            symbols = (
                "BOOM300N",
                "BOOM1000",
                "CRASH300N",
                "CRASH1000",
            )

        _SETTINGS_INSTANCE = Settings(
            deriv_app_id=_int_env("DERIV_APP_ID", 1089),
            symbols=symbols,
            mode="signal_only",
            telegram_bot_token=(os.getenv("TELEGRAM_BOT_TOKEN") or "").strip() or None,
            telegram_chat_id=(os.getenv("TELEGRAM_CHAT_ID") or "").strip() or None,
            telegram_parse_mode=(os.getenv("TELEGRAM_PARSE_MODE") or "HTML").strip(),
            notify_on_start=_bool_env("NOTIFY_ON_START", True),
            min_signal_score=_float_env("MIN_SIGNAL_SCORE", 72.0),
            signal_warmup_bars=max(60, _int_env("SIGNAL_WARMUP_BARS", 120)),
            signal_timezone=(os.getenv("SIGNAL_TIMEZONE") or "Asia/Kuala_Lumpur").strip(),
            tick_sample_seconds=max(0, _int_env("TICK_SAMPLE_SECONDS", 2)),
            data_db_path=db_path,
            dashboard_refresh_seconds=max(1, _int_env("DASHBOARD_REFRESH_SECONDS", 2)),
            max_signals_per_symbol_per_hour=max(
                1, _int_env("MAX_SIGNALS_PER_SYMBOL_PER_HOUR", 6)
            ),
            min_minutes_between_signals_same_symbol=max(
                0, _int_env("MIN_MINUTES_BETWEEN_SIGNALS_SAME_SYMBOL", 10)
            ),
            max_spread_points_estimate=_float_env("MAX_SPREAD_POINTS_ESTIMATE", 15.0),
            estimated_spread_points=_float_env("ESTIMATED_SPREAD_POINTS", 4.0),
            extreme_atr_ratio_threshold=_float_env("EXTREME_ATR_RATIO_THRESHOLD", 3.2),
            cooldown_after_spike_seconds=max(
                0, _int_env("COOLDOWN_AFTER_SPIKE_SECONDS", 45)
            ),
            spike_body_atr_multiplier_boom=_float_env(
                "SPIKE_BODY_ATR_MULTIPLIER_BOOM", 2.5
            ),
            spike_body_atr_multiplier_crash=_float_env(
                "SPIKE_BODY_ATR_MULTIPLIER_CRASH", 2.5
            ),
            spike_tick_velocity_threshold_boom=_float_env(
                "SPIKE_TICK_VELOCITY_THRESHOLD_BOOM", 0.0015
            ),
            spike_tick_velocity_threshold_crash=_float_env(
                "SPIKE_TICK_VELOCITY_THRESHOLD_CRASH", 0.0015
            ),
            log_level=(os.getenv("LOG_LEVEL") or "INFO").strip().upper(),
        )

    return _SETTINGS_INSTANCE
