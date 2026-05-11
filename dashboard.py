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
        SELECT symbol, side, score, timeframe, payload_json, created_epoch
        FROM signals
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    out: list[dict] = []
    for sym, side, score, tf, payload, created in rows:
        base = {"symbol": sym, "side": side, "score": score, "timeframe": tf, "epoch": created}
        extra = json.loads(payload)
        base.update(extra)
        out.append(base)
    return out


def load_trade_approvals(db_path: Path, limit: int = 50) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT approval_id, status, symbol, side, score, stake, currency,
                   duration, duration_unit, created_epoch, updated_epoch, note
            FROM trade_approvals
            ORDER BY created_epoch DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows,
        columns=[
            "approval_id", "status", "symbol", "side", "score", "stake",
            "currency", "duration", "duration_unit", "created_epoch",
            "updated_epoch", "note",
        ],
    )
    df["created_time"] = pd.to_datetime(df["created_epoch"], unit="s", utc=True)
    return df


def load_trade_executions(db_path: Path, limit: int = 50) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT approval_id, status, request_json, response_json, created_epoch
            FROM trade_executions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    if not rows:
        return pd.DataFrame()
    out = []
    for approval_id, status, req_json, resp_json, created_epoch in rows:
        req = json.loads(req_json)
        resp = json.loads(resp_json)
        buy_resp = resp.get("buy_response") or {}
        buy = buy_resp.get("buy") or {}
        error = buy_resp.get("error") or (resp.get("proposal_response") or {}).get("error") or {}
        out.append(
            {
                "approval_id": approval_id,
                "status": status,
                "symbol": req.get("symbol"),
                "side": req.get("side"),
                "stake": req.get("stake"),
                "contract_id": buy.get("contract_id"),
                "buy_price": buy.get("buy_price"),
                "error": error.get("message"),
                "created_time": pd.to_datetime(created_epoch, unit="s", utc=True),
            }
        )
    return pd.DataFrame(out)


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
                f"Last alert: `{last_sig.get('side')}` @ {last_sig.get('score', 0):.1f} pts "
                f"— {last_sig.get('summary', '')}",
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

    st.subheader("Recent stored signals")
    st.dataframe(load_recent_signals(db_path))

    st.subheader("Telegram trade approvals")
    approvals_df = load_trade_approvals(db_path)
    if approvals_df.empty:
        st.caption("No trade approvals yet.")
    else:
        st.dataframe(approvals_df)

    st.subheader("Trade executions")
    executions_df = load_trade_executions(db_path)
    if executions_df.empty:
        st.caption("No trade executions yet.")
    else:
        st.dataframe(executions_df)

    time.sleep(refresh)
    st.rerun()


if __name__ == "__main__":
    main()
