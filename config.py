"""Runtime settings loaded from environment variables."""
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
    return [p.strip() for p in raw.split(",") if p.strip()]


@dataclass(frozen=True)
class Settings:
    deriv_app_id: int
    deriv_api_token: str | None
    telegram_bot_token: str | None
    telegram_chat_id: str | None
    telegram_parse_mode: str
    notify_on_start: bool
    mode: str
    symbols: tuple[str, ...]

    min_signal_score: float
    signal_warmup_bars: int
    signal_timezone: str
    risk_percent: float
    tick_sample_seconds: int

    enable_real_trading: bool
    deriv_real_trading_confirm: str | None

    preparation_alerts_enabled: bool
    trigger_alerts_enabled: bool
    trigger_min_signal_score: float
    trigger_spike_strength: float
    trigger_tick_velocity_min: float
    require_micro_break_for_trigger: bool
    micro_break_lookback: int

    entry_zone_atr_multiplier: float
    stop_loss_atr_multiplier: float
    take_profit_1_atr_multiplier: float
    take_profit_2_atr_multiplier: float
    min_risk_reward: float

    trend_following_spike_mode: bool
    require_trend_alignment: bool
    require_regime_alignment: bool
    allow_counter_regime_reversal: bool
    regime_conflict_penalty: float
    require_price_action_confirmation_in_high_vol: bool

    stoch_enabled: bool
    require_stoch_for_trigger: bool
    stoch_k_period: int
    stoch_d_period: int
    stoch_smoothing: int
    stoch_oversold: float
    stoch_overbought: float

    outcome_tracking_enabled: bool
    signal_expiry_minutes: int
    notify_signal_outcomes: bool

    ml_feature_logging_enabled: bool
    ml_training_min_samples: int

    ict_bpr_enabled: bool
    ict_bpr_lookback_candles: int
    ict_bpr_score_bonus: float
    ict_bpr_require_for_trigger: bool
    ict_bpr_max_distance_atr: float

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


_SETTINGS_INSTANCE: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS_INSTANCE
    if _SETTINGS_INSTANCE is None:
        data_dir = Path(os.getenv("DATA_DIR") or (_project_root() / "data"))
        db_path = data_dir / "deriv_signals.db"

        symbols = tuple(_split_symbols(os.getenv("SYMBOLS")))
        if not symbols:
            symbols = (
                "BOOM300N", "BOOM500", "BOOM600", "BOOM900", "BOOM1000",
                "CRASH300N", "CRASH500", "CRASH600", "CRASH900", "CRASH1000",
            )

        confirm = (os.getenv("DERIV_REAL_TRADING_CONFIRM") or "").strip() or None

        _SETTINGS_INSTANCE = Settings(
            deriv_app_id=_int_env("DERIV_APP_ID", 1089),
            deriv_api_token=(os.getenv("DERIV_API_TOKEN") or "").strip() or None,
            telegram_bot_token=(os.getenv("TELEGRAM_BOT_TOKEN") or "").strip() or None,
            telegram_chat_id=(os.getenv("TELEGRAM_CHAT_ID") or "").strip() or None,
            telegram_parse_mode=(os.getenv("TELEGRAM_PARSE_MODE") or "HTML").strip(),
            notify_on_start=_bool_env("NOTIFY_ON_START", True),
            mode=(os.getenv("MODE") or "signal_only").strip(),
            symbols=symbols,
            min_signal_score=_float_env("MIN_SIGNAL_SCORE", 72.0),
            signal_warmup_bars=max(60, _int_env("SIGNAL_WARMUP_BARS", 180)),
            signal_timezone=(os.getenv("SIGNAL_TIMEZONE") or "Asia/Kuala_Lumpur").strip(),
            risk_percent=_float_env("RISK_PERCENT", 1.0),
            tick_sample_seconds=max(0, _int_env("TICK_SAMPLE_SECONDS", 2)),
            enable_real_trading=_bool_env("ENABLE_REAL_TRADING", False),
            deriv_real_trading_confirm=confirm,
            preparation_alerts_enabled=_bool_env("PREPARATION_ALERTS_ENABLED", False),
            trigger_alerts_enabled=_bool_env("TRIGGER_ALERTS_ENABLED", True),
            trigger_min_signal_score=_float_env("TRIGGER_MIN_SIGNAL_SCORE", 82.0),
            trigger_spike_strength=_float_env("TRIGGER_SPIKE_STRENGTH", 0.8),
            trigger_tick_velocity_min=_float_env("TRIGGER_TICK_VELOCITY_MIN", 0.01),
            require_micro_break_for_trigger=_bool_env("REQUIRE_MICRO_BREAK_FOR_TRIGGER", True),
            micro_break_lookback=max(2, _int_env("MICRO_BREAK_LOOKBACK", 3)),
            entry_zone_atr_multiplier=max(0.01, _float_env("ENTRY_ZONE_ATR_MULTIPLIER", 0.08)),
            stop_loss_atr_multiplier=max(0.5, _float_env("STOP_LOSS_ATR_MULTIPLIER", 2.8)),
            take_profit_1_atr_multiplier=max(0.5, _float_env("TAKE_PROFIT_1_ATR_MULTIPLIER", 3.5)),
            take_profit_2_atr_multiplier=max(0.5, _float_env("TAKE_PROFIT_2_ATR_MULTIPLIER", 6.0)),
            min_risk_reward=max(0.1, _float_env("MIN_RISK_REWARD", 1.2)),
            trend_following_spike_mode=_bool_env("TREND_FOLLOWING_SPIKE_MODE", True),
            require_trend_alignment=_bool_env("REQUIRE_TREND_ALIGNMENT", True),
            require_regime_alignment=_bool_env("REQUIRE_REGIME_ALIGNMENT", True),
            allow_counter_regime_reversal=_bool_env("ALLOW_COUNTER_REGIME_REVERSAL", False),
            regime_conflict_penalty=max(0.0, _float_env("REGIME_CONFLICT_PENALTY", 35.0)),
            require_price_action_confirmation_in_high_vol=_bool_env("REQUIRE_PRICE_ACTION_CONFIRMATION_IN_HIGH_VOL", True),
            stoch_enabled=_bool_env("STOCH_ENABLED", True),
            require_stoch_for_trigger=_bool_env("REQUIRE_STOCH_FOR_TRIGGER", True),
            stoch_k_period=max(3, _int_env("STOCH_K_PERIOD", 14)),
            stoch_d_period=max(1, _int_env("STOCH_D_PERIOD", 3)),
            stoch_smoothing=max(1, _int_env("STOCH_SMOOTHING", 3)),
            stoch_oversold=_float_env("STOCH_OVERSOLD", 20.0),
            stoch_overbought=_float_env("STOCH_OVERBOUGHT", 80.0),
            outcome_tracking_enabled=_bool_env("OUTCOME_TRACKING_ENABLED", True),
            signal_expiry_minutes=max(5, _int_env("SIGNAL_EXPIRY_MINUTES", 180)),
            notify_signal_outcomes=_bool_env("NOTIFY_SIGNAL_OUTCOMES", True),
            ml_feature_logging_enabled=_bool_env("ML_FEATURE_LOGGING_ENABLED", True),
            ml_training_min_samples=max(20, _int_env("ML_TRAINING_MIN_SAMPLES", 200)),
            ict_bpr_enabled=_bool_env("ICT_BPR_ENABLED", True),
            ict_bpr_lookback_candles=max(20, _int_env("ICT_BPR_LOOKBACK_CANDLES", 120)),
            ict_bpr_score_bonus=max(0.0, _float_env("ICT_BPR_SCORE_BONUS", 5.0)),
            ict_bpr_require_for_trigger=_bool_env("ICT_BPR_REQUIRE_FOR_TRIGGER", False),
            ict_bpr_max_distance_atr=max(0.1, _float_env("ICT_BPR_MAX_DISTANCE_ATR", 2.0)),
            max_signals_per_symbol_per_hour=max(1, _int_env("MAX_SIGNALS_PER_SYMBOL_PER_HOUR", 4)),
            min_minutes_between_signals_same_symbol=max(0, _int_env("MIN_MINUTES_BETWEEN_SIGNALS_SAME_SYMBOL", 10)),
            max_spread_points_estimate=_float_env("MAX_SPREAD_POINTS_ESTIMATE", 15.0),
            estimated_spread_points=_float_env("ESTIMATED_SPREAD_POINTS", 4.0),
            extreme_atr_ratio_threshold=_float_env("EXTREME_ATR_RATIO_THRESHOLD", 3.2),
            cooldown_after_spike_seconds=max(0, _int_env("COOLDOWN_AFTER_SPIKE_SECONDS", 5)),
            spike_body_atr_multiplier_boom=_float_env("SPIKE_BODY_ATR_MULTIPLIER_BOOM", 3.5),
            spike_body_atr_multiplier_crash=_float_env("SPIKE_BODY_ATR_MULTIPLIER_CRASH", 3.5),
            spike_tick_velocity_threshold_boom=_float_env("SPIKE_TICK_VELOCITY_THRESHOLD_BOOM", 0.02),
            spike_tick_velocity_threshold_crash=_float_env("SPIKE_TICK_VELOCITY_THRESHOLD_CRASH", 0.02),
            daily_loss_limit_percent=_float_env("DAILY_LOSS_LIMIT_PERCENT", 5.0),
            max_open_trades=max(1, _int_env("MAX_OPEN_TRADES", 3)),
            dashboard_refresh_seconds=max(1, _int_env("DASHBOARD_REFRESH_SECONDS", 2)),
            log_level=(os.getenv("LOG_LEVEL") or "INFO").strip().upper(),
            data_db_path=db_path,
        )
    return _SETTINGS_INSTANCE


def execution_allowed(settings: Settings) -> tuple[bool, str]:
    if settings.mode.strip().lower() == "signal_only":
        return False, "MODE is signal_only."
    if not settings.enable_real_trading:
        return False, "ENABLE_REAL_TRADING is false."
    expected = "I_UNDERSTAND_REAL_TRADING_RISK"
    if settings.deriv_real_trading_confirm != expected:
        return False, "DERIV_REAL_TRADING_CONFIRM missing."
    return True, "Execution checks passed."

def get_data_dir() -> Path:
    """Return the persistent data directory used for SQLite + logs.

    Railway should set DATA_DIR=/app/data. Locally, this falls back to
    the project data/ folder.
    """
    raw = os.getenv("DATA_DIR")
    if raw and raw.strip():
        return Path(raw).expanduser().resolve()
    return (_project_root() / "data").resolve()

