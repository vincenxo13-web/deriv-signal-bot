"""
Streamlit dashboard for reading live snapshots + historical candles from SQLite.

Run (from project root):

  streamlit run dashboard.py
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from config import get_settings
from indicators import attach_core_indicators


def load_meta(db_path: Path, key: str) -> dict | None:
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    conn.close()
    if not row:
        return None
    return json.loads(row[0])


def load_recent_signals(db_path: Path, limit: int = 50) -> list[dict]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT symbol, side, score, timeframe, payload_json, created_epoch,
               outcome_status, outcome_epoch, outcome_price, outcome_reason
        FROM signals
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    out: list[dict] = []
    for row in rows:
        sym, side, score, tf, payload, created, status, outcome_epoch, outcome_price, outcome_reason = row
        base = {
            "symbol": sym,
            "side": side,
            "score": score,
            "timeframe": tf,
            "epoch": created,
            "outcome_status": status,
            "outcome_epoch": outcome_epoch,
            "outcome_price": outcome_price,
            "outcome_reason": outcome_reason,
        }
        extra = json.loads(payload)
        base.update(extra)
        out.append(base)
    return out


def load_outcome_stats(db_path: Path, limit_days: int = 30) -> tuple[dict, pd.DataFrame]:
    cutoff = time.time() - max(1, limit_days) * 86400.0
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT symbol, side, outcome_status, COUNT(*) AS count
        FROM signals
        WHERE created_epoch >= ?
        GROUP BY symbol, side, outcome_status
        ORDER BY symbol ASC, outcome_status ASC
        """,
        (cutoff,),
    ).fetchall()
    conn.close()

    totals = {"signals": 0, "wins": 0, "losses": 0, "open": 0, "expired": 0}
    by_symbol: dict[str, dict] = {}

    for sym, side, status, count in rows:
        rec = by_symbol.setdefault(
            sym,
            {"symbol": sym, "signals": 0, "wins": 0, "losses": 0, "open": 0, "expired": 0},
        )
        count = int(count)
        rec["signals"] += count
        totals["signals"] += count

        if str(status).startswith("WIN"):
            rec["wins"] += count
            totals["wins"] += count
        elif str(status).startswith("LOSS"):
            rec["losses"] += count
            totals["losses"] += count
        elif status == "OPEN":
            rec["open"] += count
            totals["open"] += count
        elif status == "EXPIRED":
            rec["expired"] += count
            totals["expired"] += count

    rows_out = []
    for rec in by_symbol.values():
        closed = rec["wins"] + rec["losses"] + rec["expired"]
        rec["win_rate_closed"] = rec["wins"] / max(1, closed)
        rows_out.append(rec)

    closed_total = totals["wins"] + totals["losses"] + totals["expired"]
    totals["win_rate_closed"] = totals["wins"] / max(1, closed_total)
    return totals, pd.DataFrame(rows_out)


def load_candles(
    db_path: Path,
    symbol: str,
    timeframe: str,
    max_bars: int,
) -> pd.DataFrame:
    """Load OHLC candles oldest-first (for charts)."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT bucket_epoch, open, high, low, close
        FROM candles
        WHERE UPPER(symbol) = UPPER(?) AND timeframe = ?
        ORDER BY bucket_epoch DESC
        LIMIT ?
        """,
        (symbol, timeframe, max_bars),
    ).fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["bucket_epoch", "open", "high", "low", "close"])
    df["datetime"] = pd.to_datetime(df["bucket_epoch"], unit="s", utc=True)
    df = df.sort_values("datetime").set_index("datetime")
    return df[["open", "high", "low", "close"]]


def indicator_status(rsi: float | None, macd_hist: float | None) -> str:
    parts = []
    if rsi is None:
        parts.append("RSI warming up")
    elif rsi >= 65:
        parts.append("RSI stretched higher")
    elif rsi <= 35:
        parts.append("RSI stretched lower")
    else:
        parts.append("RSI mid-zone")

    if macd_hist is None:
        parts.append("MACD warming up")
    elif macd_hist >= 0:
        parts.append("MACD histogram >= 0")
    else:
        parts.append("MACD histogram < 0")
    return " · ".join(parts)


def ema_status(em20: float | None, em50: float | None, em200: float | None) -> str:
    if em20 is None or em50 is None or em200 is None:
        return "EMAs still forming"
    if em20 > em50 > em200:
        return "EMA stack bullish (20>50>200)"
    if em20 < em50 < em200:
        return "EMA stack bearish (20<50<200)"
    return "EMA stack mixed / transition"


def render_symbol_charts(db_path: Path, symbol: str, timeframe: str, max_bars: int) -> None:
    df = load_candles(db_path, symbol, timeframe, max_bars)
    if df.empty or len(df) < 5:
        st.info(f"No candle data yet for **{symbol}** · **{timeframe}** — wait for the bot to accumulate bars.")
        return

    feat = attach_core_indicators(df)
    last = feat.iloc[-1]

    price_block = pd.DataFrame(
        {
            "close": feat["close"],
            "ema_20": feat["ema_20"],
            "ema_50": feat["ema_50"],
            "ema_200": feat["ema_200"],
        }
    )
    st.markdown("**Price + EMAs**")
    st.line_chart(price_block, height=320)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**RSI (14)**")
        rsi_series = feat["rsi_14"].dropna()
        if not rsi_series.empty:
            st.line_chart(rsi_series.to_frame(name="RSI"), height=220)
        else:
            st.caption("RSI not ready yet.")

    with c2:
        st.markdown("**MACD histogram**")
        if "macd_hist" in feat.columns:
            mh = feat["macd_hist"].dropna()
            if not mh.empty:
                st.bar_chart(mh.to_frame(name="hist"), height=220)
            else:
                st.caption("MACD not ready yet.")
        else:
            st.caption("MACD not available.")

    st.markdown("**ATR (14)**")
    if "atr_14" in feat.columns:
        atr = feat["atr_14"].dropna()
        if not atr.empty:
            st.line_chart(atr.to_frame(name="ATR"), height=180)
    else:
        st.caption("ATR not ready.")

    snap = (
        f"**Latest bar** · O {last['open']:.5f} H {last['high']:.5f} "
        f"L {last['low']:.5f} C {last['close']:.5f}"
    )
    if pd.notna(last.get("rsi_14")):
        snap += f" · RSI {float(last['rsi_14']):.1f}"
    st.caption(snap)


def render_sparklines(db_path: Path, symbols: tuple[str, ...], bars: int, tf: str) -> None:
    """Compact last-N closes for every symbol (quick overview)."""
    st.markdown("**Quick price trail (all symbols)**")
    cols = st.columns(min(3, len(symbols)) or 1)
    for i, sym in enumerate(symbols):
        df = load_candles(db_path, sym, tf, bars)
        col = cols[i % len(cols)]
        with col:
            if df.empty:
                st.caption(f"{sym}: no data")
                continue
            trail = df["close"].iloc[-bars:]
            st.caption(sym)
            st.line_chart(trail.to_frame(name="close"), height=140)


def main() -> None:
    settings = get_settings()
    st.set_page_config(page_title="Deriv Signal Dashboard", layout="wide")
    st.title("Deriv Crash / Boom Signal Monitor")
    st.caption("Live research view — refresh pulls from local SQLite snapshots.")

    db_path = Path(settings.data_db_path)
    if not db_path.exists():
        st.error("SQLite file not found — run main.py briefly to initialise storage.")
        st.stop()

    refresh = settings.dashboard_refresh_seconds

    snapshot = load_meta(db_path, "dashboard_snapshot") or {}
    heartbeat = load_meta(db_path, "bot_heartbeat") or {}
    status = load_meta(db_path, "bot_status") or {}

    cols = st.columns(3)
    cols[0].metric("Tracked symbols", len(settings.symbols))
    hb = heartbeat.get("epoch")
    cols[1].metric(
        "Last heartbeat age (s)",
        f"{max(0.0, time.time() - float(hb)):.1f}" if hb else "n/a",
    )
    cols[2].metric("MODE", settings.mode)

    conn_state = (
        "connected"
        if status.get("last_msg_epoch")
        and (time.time() - float(status["last_msg_epoch"])) < 60
        else "idle / awaiting ticks"
    )
    st.success(f"WebSocket ingest: **{conn_state}**")

    with st.sidebar:
        st.header("Charts")
        chart_sym = st.selectbox("Symbol", list(settings.symbols), key="chart_sym")
        chart_tf = st.selectbox("Timeframe", ("1m", "5m", "15m"), index=0, key="chart_tf")
        chart_bars = st.slider("Bars to load", min_value=50, max_value=2000, value=400, step=50)
        spark_bars = st.slider("Sparkline bars", 20, 200, 60, key="spark")
        spark_tf = st.selectbox("Sparkline TF", ("1m", "5m", "15m"), index=0, key="spark_tf")
        st.caption("Charts read from `candles` in SQLite; run `main.py` to populate.")

    st.subheader("Charts (price & indicators)")
    render_symbol_charts(db_path, chart_sym, chart_tf, chart_bars)

    render_sparklines(db_path, settings.symbols, spark_bars, spark_tf)

    st.subheader("Per-symbol telemetry")
    for sym in settings.symbols:
        data = snapshot.get(sym, {})
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"**{sym}**")
        price = data.get("price")
        c2.metric("Latest price / last close", f"{price:.5f}" if price is not None else "…")
        last_sig = data.get("last_signal")
        if last_sig:
            c3.markdown(
                f"Last alert: `{last_sig.get('stage', 'SIGNAL')}` `{last_sig.get('side')}` "
                f"@ {last_sig.get('score', 0):.1f} pts — {last_sig.get('summary', '')}",
            )
        else:
            c3.markdown("_No qualifying alert since restart / bar close._")

        rsi = data.get("rsi")
        macdh = data.get("macd_hist")
        c4.markdown(f"{indicator_status(rsi, macdh)}")

        st.caption(
            f"Trend: `{data.get('regime','n/a')}` — {ema_status(data.get('ema20'), data.get('ema50'), data.get('ema200'))}"
            f"{(' · ' + str(data['regime_note'])) if data.get('regime_note') else ''}"
        )
        note = data.get("risk_note")
        if note:
            st.warning(f"Risk filter note for {sym}: {note}")

        st.divider()

    st.subheader("Signal outcome tracking")
    totals, symbol_stats = load_outcome_stats(db_path)
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Signals tracked", totals.get("signals", 0))
    s2.metric("Wins", totals.get("wins", 0))
    s3.metric("Losses", totals.get("losses", 0))
    s4.metric("Open", totals.get("open", 0))
    s5.metric("Closed win rate", f"{totals.get('win_rate_closed', 0) * 100:.1f}%")
    if not symbol_stats.empty:
        st.dataframe(symbol_stats, use_container_width=True)
    else:
        st.caption("No resolved signal outcomes yet. The tracker will update after TP/SL/expiry.")

    st.subheader("Recent stored signals")
    st.dataframe(load_recent_signals(db_path), use_container_width=True)

    time.sleep(refresh)
    st.rerun()


if __name__ == "__main__":
    main()
