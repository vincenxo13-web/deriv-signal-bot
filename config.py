"""
Load application settings from environment variables (.env via python-dotenv).

All secrets live in .env — never commit real tokens.
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
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


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
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def data_dir() -> Path:
    """
    Persistent data directory.

    Railway:
      - add a Volume mounted to /app/data
      - set DATA_DIR=/app/data

    Local:
      - defaults to ./data inside the project
    """
    return Path(os.getenv("RAILWAY_VOLUME_MOUNT_PATH", os.getenv("DATA_DIR", _project_root() / "data")))


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded once."""

    deriv_app_id: int
    deriv_api_token: str | None
    telegram_bot_token: str | None
    telegram_chat_id: str | None

    mode: str
    symbols: tuple[str, ...]

    min_signal_score: float
    risk_percent: float

    enable_real_trading: bool
    deriv_real_trading_confirm: str | None

    tick_sample_seconds: int

    max_signals_per_symbol_per_hour: int
    min_minutes_between_signals_same_symbol: int
    max_spread_points_estimate: float
    estimated_spread_points: float
    extreme_atr_ratio_threshold: float
    cooldown_after_spike_seconds: int

    spike_body_atr_multiplier_boom: float
    spike_body_atr_multiplier_crash: float
    spike_tick_velocity_threshold_boom: float
    spike_tick_velocity_threshold_crash: float

    daily_loss_limit_percent: float
    max_open_trades: int

    dashboard_refresh_seconds: int

    log_level: str

    data_db_path: Path

    trade_approval_required: bool
    trade_stake: float
    trade_currency: str
    trade_duration: int
    trade_duration_unit: str
    trade_basis: str
    telegram_poll_seconds: int
    trade_max_price: float


_SETTINGS_INSTANCE: Settings | None = None


def get_settings() -> Settings:
    """Singleton-style settings accessor (helps tests reload by clearing module)."""
    global _SETTINGS_INSTANCE
    if _SETTINGS_INSTANCE is None:
        db_path = data_dir() / "deriv_signals.db"

        symbols = tuple(_split_symbols(os.getenv("SYMBOLS")))
        if not symbols:
            symbols = (
                "BOOM1000",
                "BOOM500",
                "BOOM300",
                "CRASH1000",
                "CRASH500",
                "CRASH300",
            )

        confirm = os.getenv("DERIV_REAL_TRADING_CONFIRM")
        confirm_norm = confirm.strip() if confirm else None
        if confirm_norm == "":
            confirm_norm = None

        trade_stake = max(0.35, _float_env("TRADE_STAKE", 1.0))
        trade_max_price = max(trade_stake, _float_env("TRADE_MAX_PRICE", trade_stake))

        _SETTINGS_INSTANCE = Settings(
            deriv_app_id=_int_env("DERIV_APP_ID", 1089),
            deriv_api_token=(os.getenv("DERIV_API_TOKEN") or "").strip() or None,
            telegram_bot_token=(os.getenv("TELEGRAM_BOT_TOKEN") or "").strip() or None,
            telegram_chat_id=(os.getenv("TELEGRAM_CHAT_ID") or "").strip() or None,
            mode=(os.getenv("MODE") or "signal_only").strip(),
            symbols=symbols,
            min_signal_score=_float_env("MIN_SIGNAL_SCORE", 75.0),
            risk_percent=_float_env("RISK_PERCENT", 1.0),
            enable_real_trading=_bool_env("ENABLE_REAL_TRADING", False),
            deriv_real_trading_confirm=confirm_norm,
            tick_sample_seconds=max(0, _int_env("TICK_SAMPLE_SECONDS", 2)),
            max_signals_per_symbol_per_hour=max(
                1, _int_env("MAX_SIGNALS_PER_SYMBOL_PER_HOUR", 8)
            ),
            min_minutes_between_signals_same_symbol=max(
                0, _int_env("MIN_MINUTES_BETWEEN_SIGNALS_SAME_SYMBOL", 15)
            ),
            max_spread_points_estimate=_float_env("MAX_SPREAD_POINTS_ESTIMATE", 15.0),
            estimated_spread_points=_float_env("ESTIMATED_SPREAD_POINTS", 4.0),
            extreme_atr_ratio_threshold=_float_env(
                "EXTREME_ATR_RATIO_THRESHOLD", 3.0
            ),
            cooldown_after_spike_seconds=max(
                0, _int_env("COOLDOWN_AFTER_SPIKE_SECONDS", 120)
            ),
            spike_body_atr_multiplier_boom=_float_env(
                "SPIKE_BODY_ATR_MULTIPLIER_BOOM", 2.8
            ),
            spike_body_atr_multiplier_crash=_float_env(
                "SPIKE_BODY_ATR_MULTIPLIER_CRASH", 2.8
            ),
            spike_tick_velocity_threshold_boom=_float_env(
                "SPIKE_TICK_VELOCITY_THRESHOLD_BOOM", 0.0015
            ),
            spike_tick_velocity_threshold_crash=_float_env(
                "SPIKE_TICK_VELOCITY_THRESHOLD_CRASH", 0.0015
            ),
            daily_loss_limit_percent=_float_env("DAILY_LOSS_LIMIT_PERCENT", 5.0),
            max_open_trades=max(1, _int_env("MAX_OPEN_TRADES", 3)),
            dashboard_refresh_seconds=max(1, _int_env("DASHBOARD_REFRESH_SECONDS", 2)),
            log_level=(os.getenv("LOG_LEVEL") or "INFO").strip().upper(),
            data_db_path=db_path,
            trade_approval_required=_bool_env("TRADE_APPROVAL_REQUIRED", True),
            trade_stake=trade_stake,
            trade_currency=(os.getenv("TRADE_CURRENCY") or "USD").strip().upper(),
            trade_duration=max(1, _int_env("TRADE_DURATION", 1)),
            trade_duration_unit=(os.getenv("TRADE_DURATION_UNIT") or "m").strip(),
            trade_basis=(os.getenv("TRADE_BASIS") or "stake").strip(),
            telegram_poll_seconds=max(1, _int_env("TELEGRAM_POLL_SECONDS", 2)),
            trade_max_price=trade_max_price,
        )

    return _SETTINGS_INSTANCE


def execution_allowed(settings: Settings) -> tuple[bool, str]:
    """
    Deriv proposal/buy/sell must stay blocked unless operator explicitly opts in.

    Demo tokens and real tokens can both place contracts, so this guard is kept even
    when you plan to use a demo account.
    """
    if settings.mode.strip().lower() == "signal_only":
        return False, "MODE is signal_only (signals only)."
    if settings.mode.strip().lower() not in {"approval_trade", "auto_trade"}:
        return False, "MODE must be approval_trade or auto_trade for execution."
    if not settings.enable_real_trading:
        return False, "ENABLE_REAL_TRADING is false."
    expected = "I_UNDERSTAND_REAL_TRADING_RISK"
    if settings.deriv_real_trading_confirm != expected:
        return False, (
            "DERIV_REAL_TRADING_CONFIRM must exactly match the documented phrase "
            "when enabling execution."
        )
    if not settings.deriv_api_token:
        return False, "DERIV_API_TOKEN is missing."
    return True, "Execution checks passed."
