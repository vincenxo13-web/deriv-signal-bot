# Deriv Crash/Boom Telegram Signal Bot

Signal-only Python bot for Deriv synthetic Crash/Boom indices.

It streams public Deriv ticks, builds 1m / 5m / 15m candles, detects sniper-style setups, saves history to SQLite, and sends clean Telegram alerts.

**No auto-trading code is active in this build.**

## Direction rules

- **Boom symbols:** BUY signals only, focused on anticipating upward spike setups.
- **Crash symbols:** SELL signals only, focused on anticipating downward spike setups.

## What the bot sends to Telegram

Each Telegram alert includes:

- Symbol
- BUY/SELL direction
- Signal score
- Local timestamp and UTC timestamp
- Sniper entry zone
- SL idea
- TP1 / TP2 ideas
- R:R estimate
- Reasons the signal fired
- Regime / volatility note

## Railway variables

Set these in Railway → Service → Variables:

```env
DERIV_APP_ID=1089
SYMBOLS=BOOM300N,BOOM1000,CRASH300N,CRASH1000

TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_numeric_chat_id_here
TELEGRAM_PARSE_MODE=HTML
NOTIFY_ON_START=true

DATA_DIR=/app/data

MIN_SIGNAL_SCORE=72
SIGNAL_WARMUP_BARS=120
SIGNAL_TIMEZONE=Asia/Kuala_Lumpur
TICK_SAMPLE_SECONDS=2

# Outcome tracking
OUTCOME_TRACKING_ENABLED=true
SIGNAL_EXPIRY_MINUTES=180
NOTIFY_SIGNAL_OUTCOMES=true

MAX_SIGNALS_PER_SYMBOL_PER_HOUR=6
MIN_MINUTES_BETWEEN_SIGNALS_SAME_SYMBOL=10
COOLDOWN_AFTER_SPIKE_SECONDS=45

ESTIMATED_SPREAD_POINTS=4
MAX_SPREAD_POINTS_ESTIMATE=15
EXTREME_ATR_RATIO_THRESHOLD=3.2

SPIKE_BODY_ATR_MULTIPLIER_BOOM=2.5
SPIKE_BODY_ATR_MULTIPLIER_CRASH=2.5
SPIKE_TICK_VELOCITY_THRESHOLD_BOOM=0.0015
SPIKE_TICK_VELOCITY_THRESHOLD_CRASH=0.0015

DASHBOARD_REFRESH_SECONDS=2
LOG_LEVEL=INFO
```

You do **not** need `DERIV_API_TOKEN`, `DERIV_ACCOUNT_ID`, `ENABLE_REAL_TRADING`, or MT5 details for this signal-only version.

## Railway volume

Attach a Railway Volume to the service and mount it at:

```txt
/app/data
```

The SQLite database will be saved at:

```txt
/app/data/deriv_signals.db
```

Logs will be saved at:

```txt
/app/data/logs/bot.log
```

## Start command

Use this Railway start command:

```bash
bash start_all.sh
```

If you only want the dashboard:

```bash
bash start_dashboard.sh
```

## Local run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```

Dashboard:

```bash
streamlit run dashboard.py
```

## Telegram setup

1. Create a Telegram bot using `@BotFather`.
2. Copy the bot token to `TELEGRAM_BOT_TOKEN`.
3. Get your numeric chat ID using a bot like `@userinfobot`.
4. Put it in `TELEGRAM_CHAT_ID`.
5. Redeploy.
6. You should receive a startup message if `NOTIFY_ON_START=true`.

If you do not receive the startup message, the Telegram token/chat ID is wrong or the bot has not been started by you in Telegram.

## Signal outcome tracking

The bot now checks every open signal after it is sent. It marks each signal as:

- `WIN_TP1` when TP1 is reached before SL
- `WIN_TP2` when TP2 is reached before SL
- `LOSS_SL` when the invalidation/SL is reached first
- `LOSS_SL_AMBIGUOUS` when TP and SL are both inside the same 1-minute candle
- `EXPIRED` when no TP/SL is hit before `SIGNAL_EXPIRY_MINUTES`
- `OPEN` while the signal is still being tracked

Outcome updates are saved to SQLite and shown on the dashboard. If `NOTIFY_SIGNAL_OUTCOMES=true`, Telegram also receives a result message when a signal resolves.

The tracker uses 1-minute OHLC candles. If TP and SL are both touched in the same candle, the exact order is unknown, so it is marked conservatively as an ambiguous loss.

## Tuning

Faster signals:

```env
SIGNAL_WARMUP_BARS=90
MIN_SIGNAL_SCORE=68
MIN_MINUTES_BETWEEN_SIGNALS_SAME_SYMBOL=5
```

More selective signals:

```env
SIGNAL_WARMUP_BARS=160
MIN_SIGNAL_SCORE=78
MIN_MINUTES_BETWEEN_SIGNALS_SAME_SYMBOL=15
```

A good balanced starting point:

```env
SIGNAL_WARMUP_BARS=120
MIN_SIGNAL_SCORE=72
MIN_MINUTES_BETWEEN_SIGNALS_SAME_SYMBOL=10
```

## Risk note

Signals are research alerts, not guaranteed entries. Synthetic indices can spike hard and reverse quickly. Use your own confirmation before entering any trade.


## Two-stage alerts

The bot now sends two kinds of Telegram alerts:

- **Stage 1 / PREP**: the setup is forming near a useful zone. Open the chart and watch.
- **Stage 2 / TRIGGER**: spike/drop confirmation is active. This is the stronger alert, but still signal-only.

Railway variables:

```env
PREPARATION_ALERTS_ENABLED=false
TRIGGER_ALERTS_ENABLED=true
MIN_SIGNAL_SCORE=72
TRIGGER_MIN_SIGNAL_SCORE=80
TRIGGER_SPIKE_STRENGTH=0.9
TRIGGER_TICK_VELOCITY_MIN=0.015
```

Boom symbols still produce BUY-only alerts. Crash symbols still produce SELL-only alerts.

## Latest TP/SL and outcome-stat update

This build separates watchlist alerts from trade-quality alerts:

- `PREP` alerts are saved as `WATCH_ONLY`. They still appear in Telegram/dashboard, but they are **not** counted in win/loss stats.
- `TRIGGER` alerts are saved as `OPEN` and are the only alerts tracked for TP/SL/expiry outcomes.

The TP/SL model is now configurable with Railway variables:

```env
ENTRY_ZONE_ATR_MULTIPLIER=0.12
STOP_LOSS_ATR_MULTIPLIER=2.2
TAKE_PROFIT_1_ATR_MULTIPLIER=3.0
TAKE_PROFIT_2_ATR_MULTIPLIER=5.0
MIN_RISK_REWARD=1.2
```

Compared with the earlier closer targets, these defaults push TP1/TP2 further away from entry and make outcome stats focus on confirmed `TRIGGER` alerts only.

## ML feature logging, stage 1

This build can log ML-ready feature snapshots for every signal. It does **not** let ML control or block signals yet. The goal is to collect clean data first.

Railway variables:

```env
ML_FEATURE_LOGGING_ENABLED=true
ML_TRAINING_MIN_SAMPLES=200
```

Feature rows are saved in SQLite table `signal_features`. PREP alerts are saved as `WATCH_ONLY`. Only resolved `TRIGGER` rows should be used for future training.

To inspect/export the dataset locally after downloading the SQLite DB:

```bash
python ml_trainer.py --summary
python ml_trainer.py --export data/ml_training_rows.csv
```

Suggested rollout:

- 0-50 resolved TRIGGER rows: collect only.
- 50-200 resolved TRIGGER rows: inspect statistics only.
- 200+ resolved TRIGGER rows: begin offline ML experiments.
- 500+ resolved TRIGGER rows: consider enabling an ML probability filter in a later build.

## ICT H4 BPR context filter

This build adds a lightweight ICT Balanced Price Range (BPR) context layer.
The bot resamples stored 1-minute candles into H4 candles, detects simple Fair
Value Gaps, finds overlaps between opposing FVGs, and reports whether the signal
is aligned with the correct H4 BPR context.

Telegram signals now include a short H4 BPR line:

- `H4 BPR: ✅ ALIGNED` means price is inside or near the correct H4 BPR side.
- `H4 BPR: ⚠️ NEAR/NO_ZONE` means the context is not perfect but not a hard block.
- `H4 BPR: ❌ FAR` means price is too far from the nearest correct BPR zone.

Dashboard charts also shade detected H4 BPR zones:

- Green shaded zone = bullish H4 BPR, useful context for Boom BUY setups.
- Red shaded zone = bearish H4 BPR, useful context for Crash SELL setups.

Railway variables:

```env
ICT_BPR_ENABLED=true
ICT_BPR_LOOKBACK_CANDLES=120
ICT_BPR_SCORE_BONUS=5
ICT_BPR_REQUIRE_FOR_TRIGGER=false
ICT_BPR_MAX_DISTANCE_ATR=2.0
```

Recommended rollout:

1. Keep `ICT_BPR_REQUIRE_FOR_TRIGGER=false` first.
2. Let outcome tracking and ML feature logging collect data with BPR fields.
3. Later, compare win rate for `bpr_aligned=true` vs `bpr_aligned=false`.
4. Only make BPR mandatory if your data proves it improves results.


## Trend-following stochastic spike model

This build adds a stricter spike-catching model:

- **Crash:** SELL only. Bias must be downtrend, then wait for a pull-up/retracement. Stochastic should be overbought or rolling down from the overbought area before a SELL trigger.
- **Boom:** BUY only. Bias must be uptrend, then wait for a pull-down/retracement. Stochastic should be oversold or recovering from the oversold area before a BUY trigger.

Railway variables:

```env
TREND_FOLLOWING_SPIKE_MODE=true
REQUIRE_TREND_ALIGNMENT=true
STOCH_ENABLED=true
REQUIRE_STOCH_FOR_TRIGGER=true
STOCH_K_PERIOD=14
STOCH_D_PERIOD=3
STOCH_SMOOTHING=3
STOCH_OVERSOLD=20
STOCH_OVERBOUGHT=80
```

This reduces weak alerts that happen while Boom is still drifting down or Crash is still drifting up. Stochastic is used as a timing filter, not as a standalone signal.
