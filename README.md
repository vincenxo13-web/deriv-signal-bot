# Deriv Crash / Boom Signal Bot (Python)

Educational and research-oriented asyncio bot that streams **Deriv synthetic Crash / Boom indices**, aggregates **1m / 5m / 15m** candles locally, scores **BUY / SELL confluence** setups, logs everything, optionally alerts via **Telegram / macOS banners**, and exposes a **Streamlit** viewer. **Signals are not guarantees** — synthetic indices gap and spike violently.

> **Risk warning:** You can lose your entire account balance quickly. This project defaults to **signal-only** mode (`MODE=signal_only`, `ENABLE_REAL_TRADING=false`). Nothing here promises profit or fitness for live trading.

---

## 1. What the bot does

- Connects to the official **Deriv WebSocket** endpoint `wss://ws.binaryws.com/websockets/v3?app_id=…`
- Optionally authorizes if you provide `DERIV_API_TOKEN` (useful for future features)
- Subscribes to every symbol listed in `SYMBOLS` concurrently on one socket
- Stores sampled ticks + OHLC candles inside `data/deriv_signals.db` (SQLite WAL)
- Calculates **EMA(20/50/200), RSI (Wilder smoothing), MACD, Bollinger Bands, ATR (Wilder TR)** entirely with **pandas/numpy** (no brittle technical-analysis wheels required)
- Labels **regime** (`uptrend`, `downtrend`, `ranging`, `…_high_volatility`, etc.)
- Applies **Boom vs Crash** spike heuristics using ATR-scaled bodies, wicks, and tick velocity
- Emits rich **BUY/SELL** alerts including entry zone, TP ladder, SL placement, score, and rationale
- Ships a **backtester** for offline CSV / SQLite candle replays
- Includes **risk management throttles** (per-hour caps, spread estimate, post-spike cooldown, extreme ATR suppression)
- Blocks any `proposal` / `buy` helper unless every execution guard in `.env` is deliberately satisfied

---

## 2. macOS installation

```bash
cd deriv-signal-bot
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Ensure **Python 3.11+**:

```bash
python3 --version
```

---

## 3. Creating your `.env`

1. Register or locate a **Deriv Application ID** (`DERIV_APP_ID`) from the Deriv developers portal.
2. Copy `.env.example` → `.env`.
3. Fill in **placeholders only** — never commit `.env`.
4. **Symbol names matter.** Use the exact short codes from Deriv Trader / API (examples people use include `BOOM1000`, `CRASH500`, etc.). If ticks do not arrive, run `python main.py --validate-symbols` after populating `SYMBOLS`.

Key fields:

| Variable | Purpose |
| --- | --- |
| `DERIV_APP_ID` | Required WebSocket `app_id` query parameter |
| `DERIV_API_TOKEN` | Optional authorize token (keep secret) |
| `SYMBOLS` | Comma-separated list to stream |
| `MIN_SIGNAL_SCORE` | Minimum confluence score (0–100 scale) |
| `MODE` | `signal_only` (default) keeps you in research territory |
| `ENABLE_REAL_TRADING` | MUST stay `false` until you knowingly wire execution |
| `DERIV_REAL_TRADING_CONFIRM` | Must equal `I_UNDERSTAND_REAL_TRADING_RISK` if ever enabling bots to trade |

---

## 4. Running the streaming bot

```bash
source .venv/bin/activate
python main.py
```

Outputs:

- Structured logs under `logs/bot.log`
- Console INFO lines for transparency
- Optional Telegram notifications if credentials exist
- Optional macOS `osascript` banner when `osascript` is available

Press `Ctrl+C` to stop cleanly.

Validate configured symbols quickly:

```bash
python main.py --validate-symbols
```

---

## 5. Streamlit dashboard

In a **second terminal** (while `main.py` runs):

```bash
source .venv/bin/activate
streamlit run dashboard.py
```

The dashboard reads SQLite meta snapshots refreshed by the bot (prices, indicator hints, regimes, newest alerts).

---

## 6. Backtesting

Provide either a CSV dataset or SQLite candles harvested by the live bot.

**CSV format:**

```csv
datetime,symbol,open,high,low,close
2024-01-01 00:00:00,CRASH500,10050.2,10055.0,10048.1,10052.4
```

```bash
python backtester.py --csv data/your_history.csv --symbol CRASH500 --export data/backtest_trades.csv
```

**Replay from the bot database:**

```bash
python backtester.py --sqlite data/deriv_signals.db --symbol BOOM500
```

The script prints JSON metrics (win rate, profit factor, drawdown in **price points**, average R:R, counts) and exports detailed trades.

> Backtests are **idealized**. They assume TP/SL touch logic on 1-minute OHLC and ignore spread, commissions, liquidity, latency, rejected orders, contract rules, session pauses, and psychological factors.

---

## 7. Connecting Telegram alerts

1. Talk to `@BotFather` → create bot → grab **token**.
2. DM `@userinfobot` (or equivalent) → copy your **numeric chat ID**.
3. Update `.env`:

```
TELEGRAM_BOT_TOKEN=paste_here
TELEGRAM_CHAT_ID=paste_here
```

4. Restart `main.py`. The bot asynchronously POSTs formatted alerts to Telegram’s HTTPS API (`httpx`).

---

## 8. Deploying to a VPS later

High-level checklist:

1. **Ubuntu 22.04+**, non-root sudo user.
2. Install Python 3.11 (`deadsnakes` PPA if needed).
3. Clone repo, create venv, `pip install -r requirements.txt`.
4. Provide `.env` via **Secrets / dotenv**, never plaintext in repos.
5. Run `main.py` under **systemd** with `Restart=always`, `WorkingDirectory=/path/to/deriv-signal-bot`.
6. **Streamlit** (if desired) sits behind reverse proxy (`nginx`) with HTTPS + auth — do **not** expose DB files publicly.
7. Harden server (UFW firewall, unattended upgrades, SSH keys only).
8. Keep `ENABLE_REAL_TRADING=false` unless you consciously accept total loss scenarios.

Consult current Deriv docs for API limits, uptime policies, tokens, residency rules, etc.

---

## 9. Project layout

```
deriv-signal-bot/
├── README.md
├── requirements.txt
├── .env.example
├── main.py
├── config.py
├── deriv_client.py
├── market_stream.py
├── indicators.py
├── strategy.py
├── risk_manager.py
├── signal_engine.py
├── notifier.py
├── backtester.py
├── storage.py
├── dashboard.py
├── logs/
└── data/
```

---

## 10. Safety & ethics

- Treat outputs as **decision-support research**, not financial advice.
- Always start on **demo** wallets when experimenting.
- **Never** enable `ENABLE_REAL_TRADING` until execution code paths are audited, funded responsibly, every order is logged, `DAILY_LOSS_LIMIT_PERCENT` / `MAX_OPEN_TRADES` are enforced, and you typed the acknowledgment phrase verbatim.
- If you augment the bot to trade, obey local regulations — many jurisdictions classify retail derivatives strictly.

Stay curious, stay sceptical of any “edge,” and prioritise surviving the learning curve intact.

## Regime conflict protection update

This build fixes the issue where a BOOM BUY could still score highly while the regime label was `downtrend`, or a CRASH SELL could score highly while the regime label was `uptrend_high_volatility`.

New behaviour:

- BOOM BUY during a downtrend is blocked unless you explicitly allow counter-regime reversals.
- CRASH SELL during an uptrend is blocked unless you explicitly allow counter-regime reversals.
- High-volatility regimes require hard confirmation: target-direction spike pressure plus micro-break or strong rejection.
- Micro-break confirmation receives higher weighting than indicator-only reasons.
- Stochastic and EMA can support a signal, but they should not create a trigger alone.

Recommended Railway variables:

```env
PREPARATION_ALERTS_ENABLED=false
TRIGGER_ALERTS_ENABLED=true
MIN_SIGNAL_SCORE=72
TRIGGER_MIN_SIGNAL_SCORE=82
TRIGGER_SPIKE_STRENGTH=0.8
TRIGGER_TICK_VELOCITY_MIN=0.01

REQUIRE_TREND_ALIGNMENT=true
REQUIRE_REGIME_ALIGNMENT=true
ALLOW_COUNTER_REGIME_REVERSAL=false
REGIME_CONFLICT_PENALTY=35
REQUIRE_PRICE_ACTION_CONFIRMATION_IN_HIGH_VOL=true

STOCH_ENABLED=true
REQUIRE_STOCH_FOR_TRIGGER=true
STOCH_K_PERIOD=14
STOCH_D_PERIOD=3
STOCH_SMOOTHING=3
STOCH_OVERSOLD=20
STOCH_OVERBOUGHT=80
```

If triggers become too rare, lower only `TRIGGER_MIN_SIGNAL_SCORE` to `78` first. Do not set `ALLOW_COUNTER_REGIME_REVERSAL=true` until the outcome stats prove it helps.
