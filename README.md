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
