"""
Streamlit dashboard for Deriv Crash / Boom signal monitoring.

Features:
- Dark MT5-style layout
- Candlestick chart with EMA overlays
- RSI panel
- Signal markers on chart
- Outcome tracking stats
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from config import get_settings
from indicators import attach_core_indicators


# -----------------------------
# Database helpers
# -----------------------------

def load_meta(db_path: Path, key: str) -> dict | None:
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    conn.close()

    if not row:
        return None

    return json.loads(row[0])


def _table_columns(db_path: Path, table: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    conn.close()
    return {str(row[1]) for row in rows}


def load_candles(
    db_path: Path,
    symbol: str,
    timeframe: str,
    max_bars: int,
) -> pd.DataFrame:
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

    df = pd.DataFrame(
        rows,
        columns=["bucket_epoch", "open", "high", "low", "close"],
    )

    df["datetime"] = pd.to_datetime(df["bucket_epoch"], unit="s", utc=True)
    df = df.sort_values("datetime").set_index("datetime")

    return df[["open", "high", "low", "close"]]


def load_recent_signals(db_path: Path, limit: int = 100) -> list[dict[str, Any]]:
    cols = _table_columns(db_path, "signals")

    outcome_reason_col = None
    if "outcome_reason" in cols:
        outcome_reason_col = "outcome_reason"
    elif "outcome_note" in cols:
        outcome_reason_col = "outcome_note"

    select_cols = [
        "id",
        "symbol",
        "side",
        "score",
        "timeframe",
        "payload_json",
        "created_epoch",
    ]

    optional_cols = [
        "outcome_status",
        "outcome_epoch",
        "outcome_price",
    ]

    for col in optional_cols:
        if col in cols:
            select_cols.append(col)

    if outcome_reason_col:
        select_cols.append(f"{outcome_reason_col} AS outcome_reason")

    query = f"""
        SELECT {", ".join(select_cols)}
        FROM signals
        ORDER BY id DESC
        LIMIT ?
    """

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(query, (limit,)).fetchall()
    conn.close()

    out: list[dict[str, Any]] = []

    for row in rows:
        base = dict(row)

        payload = base.pop("payload_json", "{}")
        try:
            extra = json.loads(payload) if payload else {}
        except json.JSONDecodeError:
            extra = {}

        base.update(extra)
        base.setdefault("alert_stage", base.get("stage", "SIGNAL"))
        base.setdefault("outcome_status", "OPEN")
        out.append(base)

    return out


def load_chart_signals(
    db_path: Path,
    symbol: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.DataFrame:
    all_signals = load_recent_signals(db_path, limit=500)

    rows: list[dict[str, Any]] = []

    start_epoch = start_dt.timestamp()
    end_epoch = end_dt.timestamp()

    for sig in all_signals:
        if str(sig.get("symbol", "")).upper() != symbol.upper():
            continue

        created = float(sig.get("created_epoch") or sig.get("epoch") or 0)

        if created < start_epoch or created > end_epoch:
            continue

        entry_low = sig.get("entry_zone_low")
        entry_high = sig.get("entry_zone_high")

        if entry_low is not None and entry_high is not None:
            marker_price = (float(entry_low) + float(entry_high)) / 2
        elif sig.get("outcome_price") is not None:
            marker_price = float(sig["outcome_price"])
        else:
            marker_price = None

        if marker_price is None:
            continue

        stage = str(sig.get("alert_stage", sig.get("stage", "SIGNAL"))).upper()
        side = str(sig.get("side", "")).upper()
        score = float(sig.get("score") or 0)
        outcome = str(sig.get("outcome_status", "OPEN")).upper()

        if stage == "TRIGGER":
            marker_label = f"{side} TRIGGER {score:.0f}"
        elif stage == "PREP":
            marker_label = f"{side} PREP {score:.0f}"
        else:
            marker_label = f"{side} {score:.0f}"

        if outcome.startswith("WIN"):
            marker_status = "WIN"
        elif outcome.startswith("LOSS"):
            marker_status = "LOSS"
        elif outcome == "EXPIRED":
            marker_status = "EXPIRED"
        elif outcome == "WATCH_ONLY":
            marker_status = "WATCH_ONLY"
        else:
            marker_status = "OPEN"

        rows.append(
            {
                "datetime": pd.to_datetime(created, unit="s", utc=True),
                "price": marker_price,
                "symbol": sig.get("symbol"),
                "side": side,
                "stage": stage,
                "score": score,
                "outcome": outcome,
                "marker_status": marker_status,
                "label": marker_label,
                "entry_zone_low": sig.get("entry_zone_low"),
                "entry_zone_high": sig.get("entry_zone_high"),
                "stop_loss": sig.get("stop_loss"),
                "take_profit_1": sig.get("take_profit_1"),
                "take_profit_2": sig.get("take_profit_2"),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def load_outcome_stats(db_path: Path, limit_days: int = 30) -> tuple[dict[str, Any], pd.DataFrame]:
    cols = _table_columns(db_path, "signals")

    if "outcome_status" not in cols:
        return (
            {
                "trigger_signals": 0,
                "wins": 0,
                "losses": 0,
                "open": 0,
                "expired": 0,
                "watch_only": 0,
                "win_rate_closed": 0.0,
            },
            pd.DataFrame(),
        )

    cutoff = time.time() - max(1, limit_days) * 86400.0

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT symbol, side, outcome_status, payload_json, COUNT(*) AS count
        FROM signals
        WHERE created_epoch >= ?
        GROUP BY symbol, side, outcome_status, payload_json
        ORDER BY symbol ASC, outcome_status ASC
        """,
        (cutoff,),
    ).fetchall()
    conn.close()

    totals = {
        "trigger_signals": 0,
        "wins": 0,
        "losses": 0,
        "open": 0,
        "expired": 0,
        "watch_only": 0,
    }

    by_symbol: dict[str, dict[str, Any]] = {}

    for sym, side, status, payload_json, count in rows:
        count = int(count)
        status = str(status or "OPEN").upper()

        try:
            payload = json.loads(payload_json or "{}")
        except json.JSONDecodeError:
            payload = {}

        stage = str(payload.get("alert_stage", payload.get("stage", "SIGNAL"))).upper()

        if stage == "PREP" or status == "WATCH_ONLY":
            totals["watch_only"] += count
            continue

        rec = by_symbol.setdefault(
            sym,
            {
                "symbol": sym,
                "trigger_signals": 0,
                "wins": 0,
                "losses": 0,
                "open": 0,
                "expired": 0,
            },
        )

        rec["trigger_signals"] += count
        totals["trigger_signals"] += count

        if status.startswith("WIN"):
            rec["wins"] += count
            totals["wins"] += count
        elif status.startswith("LOSS"):
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



def load_ml_dataset_stats(db_path: Path, limit_days: int = 30) -> tuple[dict[str, Any], pd.DataFrame]:
    """Read ML feature logging stats from SQLite for dashboard visibility."""
    try:
        cols = _table_columns(db_path, "signal_features")
    except Exception:
        return {
            "total": 0,
            "trigger_total": 0,
            "trigger_resolved": 0,
            "trigger_wins": 0,
            "trigger_losses": 0,
            "trigger_open": 0,
            "trigger_win_rate": 0.0,
            "prep_watch_only": 0,
        }, pd.DataFrame()

    if not cols:
        return {
            "total": 0,
            "trigger_total": 0,
            "trigger_resolved": 0,
            "trigger_wins": 0,
            "trigger_losses": 0,
            "trigger_open": 0,
            "trigger_win_rate": 0.0,
            "prep_watch_only": 0,
        }, pd.DataFrame()

    cutoff = time.time() - max(1, limit_days) * 86400.0

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT symbol, alert_stage, outcome_status, COUNT(*) AS count
        FROM signal_features
        WHERE created_epoch >= ?
        GROUP BY symbol, alert_stage, outcome_status
        ORDER BY symbol ASC, alert_stage ASC, outcome_status ASC
        """,
        (cutoff,),
    ).fetchall()
    conn.close()

    totals = {
        "total": 0,
        "trigger_total": 0,
        "trigger_resolved": 0,
        "trigger_wins": 0,
        "trigger_losses": 0,
        "trigger_open": 0,
        "trigger_win_rate": 0.0,
        "prep_watch_only": 0,
    }
    by_symbol: dict[str, dict[str, Any]] = {}

    for sym, stage, status, count in rows:
        sym = str(sym)
        stage = str(stage or "SIGNAL").upper()
        status = str(status or "OPEN").upper()
        count = int(count)
        totals["total"] += count

        rec = by_symbol.setdefault(
            sym,
            {
                "symbol": sym,
                "total_features": 0,
                "trigger_total": 0,
                "trigger_resolved": 0,
                "trigger_wins": 0,
                "trigger_losses": 0,
                "trigger_open": 0,
                "prep_watch_only": 0,
            },
        )
        rec["total_features"] += count

        if stage == "TRIGGER":
            totals["trigger_total"] += count
            rec["trigger_total"] += count
            if status == "OPEN":
                totals["trigger_open"] += count
                rec["trigger_open"] += count
            elif status.startswith("WIN"):
                totals["trigger_wins"] += count
                totals["trigger_resolved"] += count
                rec["trigger_wins"] += count
                rec["trigger_resolved"] += count
            elif status.startswith("LOSS") or status == "EXPIRED":
                totals["trigger_losses"] += count
                totals["trigger_resolved"] += count
                rec["trigger_losses"] += count
                rec["trigger_resolved"] += count
        elif status == "WATCH_ONLY":
            totals["prep_watch_only"] += count
            rec["prep_watch_only"] += count

    totals["trigger_win_rate"] = totals["trigger_wins"] / max(1, totals["trigger_resolved"])
    for rec in by_symbol.values():
        rec["trigger_win_rate"] = rec["trigger_wins"] / max(1, rec["trigger_resolved"])

    return totals, pd.DataFrame(list(by_symbol.values()))

# -----------------------------
# Styling
# -----------------------------

def apply_dark_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #05070d;
            color: #f5f7fb;
        }

        [data-testid="stSidebar"] {
            background: #090d16;
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }

        div[data-testid="stMetric"] {
            background: #0d1422;
            border: 1px solid #1f2a3d;
            padding: 14px;
            border-radius: 14px;
        }

        div[data-testid="stMetricLabel"] {
            color: #8ea0bb;
        }

        div[data-testid="stMetricValue"] {
            color: #f8fafc;
        }

        .market-card {
            background: #0d1422;
            border: 1px solid #1f2a3d;
            border-radius: 14px;
            padding: 14px 16px;
            margin-bottom: 10px;
        }

        .signal-good {
            color: #30d158;
            font-weight: 700;
        }

        .signal-warn {
            color: #ffd60a;
            font-weight: 700;
        }

        .signal-bad {
            color: #ff453a;
            font-weight: 700;
        }

        .small-muted {
            color: #8ea0bb;
            font-size: 0.90rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Chart helpers
# -----------------------------

def build_candle_chart(df: pd.DataFrame, signal_df: pd.DataFrame) -> alt.Chart:
    chart_df = df.copy()
    chart_df = attach_core_indicators(chart_df)
    chart_df = chart_df.reset_index()

    chart_df["direction"] = chart_df.apply(
        lambda row: "up" if row["close"] >= row["open"] else "down",
        axis=1,
    )

    base = alt.Chart(chart_df).encode(
        x=alt.X(
            "datetime:T",
            axis=alt.Axis(
                title=None,
                labelColor="#a8b3c7",
                gridColor="#1b2434",
            ),
        )
    )

    rule = base.mark_rule().encode(
        y=alt.Y(
            "low:Q",
            title="Price",
            axis=alt.Axis(
                labelColor="#a8b3c7",
                titleColor="#a8b3c7",
                gridColor="#1b2434",
            ),
            scale=alt.Scale(zero=False),
        ),
        y2="high:Q",
        color=alt.Color(
            "direction:N",
            scale=alt.Scale(
                domain=["up", "down"],
                range=["#00c853", "#ff5252"],
            ),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("datetime:T", title="Time"),
            alt.Tooltip("open:Q", title="Open", format=".5f"),
            alt.Tooltip("high:Q", title="High", format=".5f"),
            alt.Tooltip("low:Q", title="Low", format=".5f"),
            alt.Tooltip("close:Q", title="Close", format=".5f"),
        ],
    )

    candle = base.mark_bar(size=5).encode(
        y="open:Q",
        y2="close:Q",
        color=alt.Color(
            "direction:N",
            scale=alt.Scale(
                domain=["up", "down"],
                range=["#00c853", "#ff5252"],
            ),
            legend=None,
        ),
    )

    ema_data = chart_df.melt(
        id_vars=["datetime"],
        value_vars=["ema_20", "ema_50", "ema_200"],
        var_name="EMA",
        value_name="value",
    ).dropna()

    ema = alt.Chart(ema_data).mark_line(strokeWidth=1.5).encode(
        x="datetime:T",
        y="value:Q",
        color=alt.Color(
            "EMA:N",
            scale=alt.Scale(
                domain=["ema_20", "ema_50", "ema_200"],
                range=["#4da3ff", "#ffd60a", "#b388ff"],
            ),
            legend=alt.Legend(labelColor="#a8b3c7", titleColor="#a8b3c7"),
        ),
    )

    layers = [rule, candle, ema]

    if not signal_df.empty:
        marker_colors = alt.Scale(
            domain=["WIN", "LOSS", "EXPIRED", "OPEN", "WATCH_ONLY"],
            range=["#30d158", "#ff453a", "#8ea0bb", "#ffd60a", "#fbbf24"],
        )

        signal_points = (
            alt.Chart(signal_df)
            .mark_point(
                filled=True,
                size=160,
                stroke="#ffffff",
                strokeWidth=1.2,
            )
            .encode(
                x="datetime:T",
                y="price:Q",
                shape=alt.Shape(
                    "stage:N",
                    scale=alt.Scale(
                        domain=["PREP", "TRIGGER", "SIGNAL"],
                        range=["triangle-up", "diamond", "circle"],
                    ),
                    legend=alt.Legend(labelColor="#a8b3c7", titleColor="#a8b3c7"),
                ),
                color=alt.Color(
                    "marker_status:N",
                    scale=marker_colors,
                    legend=alt.Legend(labelColor="#a8b3c7", titleColor="#a8b3c7"),
                ),
                tooltip=[
                    alt.Tooltip("datetime:T", title="Signal time"),
                    alt.Tooltip("symbol:N", title="Symbol"),
                    alt.Tooltip("stage:N", title="Stage"),
                    alt.Tooltip("side:N", title="Side"),
                    alt.Tooltip("score:Q", title="Score", format=".1f"),
                    alt.Tooltip("price:Q", title="Marker price", format=".5f"),
                    alt.Tooltip("entry_zone_low:Q", title="Entry low", format=".5f"),
                    alt.Tooltip("entry_zone_high:Q", title="Entry high", format=".5f"),
                    alt.Tooltip("stop_loss:Q", title="SL", format=".5f"),
                    alt.Tooltip("take_profit_1:Q", title="TP1", format=".5f"),
                    alt.Tooltip("take_profit_2:Q", title="TP2", format=".5f"),
                    alt.Tooltip("outcome:N", title="Outcome"),
                ],
            )
        )

        signal_labels = (
            alt.Chart(signal_df)
            .mark_text(
                align="left",
                dx=8,
                dy=-10,
                fontSize=11,
                fontWeight="bold",
                color="#f8fafc",
            )
            .encode(
                x="datetime:T",
                y="price:Q",
                text="label:N",
            )
        )

        layers.extend([signal_points, signal_labels])

    chart = (
        alt.layer(*layers)
        .properties(height=520)
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#a8b3c7",
            titleColor="#a8b3c7",
            gridColor="#1b2434",
        )
        .configure(background="#05070d")
    )

    return chart


def build_rsi_chart(df: pd.DataFrame) -> alt.Chart:
    feat = attach_core_indicators(df)
    rsi_df = feat.reset_index()[["datetime", "rsi_14"]].dropna()

    if rsi_df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line()

    base = alt.Chart(rsi_df).encode(
        x=alt.X(
            "datetime:T",
            axis=alt.Axis(title=None, labelColor="#a8b3c7", gridColor="#1b2434"),
        ),
        y=alt.Y(
            "rsi_14:Q",
            title="RSI",
            scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(labelColor="#a8b3c7", titleColor="#a8b3c7", gridColor="#1b2434"),
        ),
    )

    rsi_line = base.mark_line(color="#4da3ff", strokeWidth=2)

    levels = pd.DataFrame({"level": [15, 50, 85]})

    level_lines = (
        alt.Chart(levels)
        .mark_rule(strokeDash=[5, 5], color="#64748b")
        .encode(y="level:Q")
    )

    return (
        (rsi_line + level_lines)
        .properties(height=180)
        .configure_view(strokeWidth=0)
        .configure(background="#05070d")
    )


def render_market_cards(snapshot: dict[str, Any], symbols: tuple[str, ...]) -> None:
    st.subheader("Market telemetry")

    cols = st.columns(2)

    for i, sym in enumerate(symbols):
        data = snapshot.get(sym, {})
        price = data.get("price")
        last_sig = data.get("last_signal")

        if str(sym).upper().startswith("BOOM"):
            bias = "BUY-only spike watch"
            bias_class = "signal-good"
        else:
            bias = "SELL-only drop watch"
            bias_class = "signal-bad"

        html = f"""
        <div class="market-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div style="font-size:1.05rem;font-weight:800;">{sym}</div>
                    <div class="{bias_class}">{bias}</div>
                </div>
                <div style="text-align:right;">
                    <div class="small-muted">Last price</div>
                    <div style="font-size:1.15rem;font-weight:800;">
                        {f"{price:.5f}" if price is not None else "…"}
                    </div>
                </div>
            </div>
        """

        if last_sig:
            html += f"""
            <hr style="border-color:#1f2a3d;">
            <div class="small-muted">Latest alert</div>
            <div>
                <b>{last_sig.get("stage", "SIGNAL")}</b>
                {last_sig.get("side", "")}
                @ {float(last_sig.get("score", 0)):.1f}
            </div>
            <div class="small-muted">{last_sig.get("summary", "")}</div>
            """

        note = data.get("risk_note")
        if note:
            html += f"""
            <hr style="border-color:#1f2a3d;">
            <div class="signal-warn">Risk note: {note}</div>
            """

        html += "</div>"

        with cols[i % 2]:
            st.markdown(html, unsafe_allow_html=True)


# -----------------------------
# Main app
# -----------------------------

def main() -> None:
    settings = get_settings()

    st.set_page_config(
        page_title="Deriv Signal Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_dark_style()

    st.title("Deriv Boom / Crash Signal Monitor")
    st.caption("Signal-only dashboard · Boom BUY setups · Crash SELL setups")

    db_path = Path(settings.data_db_path)

    if not db_path.exists():
        st.error("SQLite file not found — run main.py briefly to initialise storage.")
        st.stop()

    refresh = settings.dashboard_refresh_seconds

    snapshot = load_meta(db_path, "dashboard_snapshot") or {}
    heartbeat = load_meta(db_path, "bot_heartbeat") or {}
    status = load_meta(db_path, "bot_status") or {}

    hb = heartbeat.get("epoch")
    last_msg = status.get("last_msg_epoch")

    conn_state = (
        "connected"
        if last_msg and (time.time() - float(last_msg)) < 60
        else "idle / reconnecting"
    )

    top1, top2, top3, top4 = st.columns(4)

    top1.metric("Tracked markets", len(settings.symbols))
    top2.metric("WebSocket", conn_state)
    top3.metric(
        "Heartbeat age",
        f"{max(0.0, time.time() - float(hb)):.1f}s" if hb else "n/a",
    )
    top4.metric("Mode", settings.mode)

    with st.sidebar:
        st.header("Chart controls")

        chart_sym = st.selectbox(
            "Symbol",
            list(settings.symbols),
            key="chart_sym",
        )

        chart_tf = st.selectbox(
            "Timeframe",
            ("1m", "5m", "15m"),
            index=0,
            key="chart_tf",
        )

        chart_bars = st.slider(
            "Bars to load",
            min_value=50,
            max_value=2000,
            value=400,
            step=50,
        )

        outcome_days = st.slider(
            "Outcome stats days",
            min_value=1,
            max_value=90,
            value=30,
            step=1,
        )

        st.caption("Signal markers are plotted from SQLite signal history.")

    df = load_candles(db_path, chart_sym, chart_tf, chart_bars)

    st.subheader(f"{chart_sym} · {chart_tf} chart")

    if df.empty or len(df) < 5:
        st.info(
            f"No candle data yet for **{chart_sym}** · **{chart_tf}**. "
            "Wait for the bot to accumulate bars."
        )
    else:
        signal_df = load_chart_signals(
            db_path=db_path,
            symbol=chart_sym,
            start_dt=df.index.min(),
            end_dt=df.index.max(),
        )

        st.altair_chart(
            build_candle_chart(df, signal_df),
            width="stretch",
        )

        st.markdown("**RSI (14)**")
        st.altair_chart(
            build_rsi_chart(df),
            width="stretch",
        )

        if not signal_df.empty:
            st.caption(f"Showing {len(signal_df)} signal marker(s) on this chart.")
        else:
            st.caption("No signals found inside the currently loaded chart range.")

    st.divider()

    totals, symbol_stats = load_outcome_stats(db_path, limit_days=outcome_days)

    st.subheader("Trigger signal outcome tracking")

    s1, s2, s3, s4, s5, s6 = st.columns(6)

    s1.metric("Trigger signals", totals.get("trigger_signals", 0))
    s2.metric("Wins", totals.get("wins", 0))
    s3.metric("Losses", totals.get("losses", 0))
    s4.metric("Open", totals.get("open", 0))
    s5.metric("Watch-only PREP", totals.get("watch_only", 0))
    s6.metric("Closed win rate", f"{totals.get('win_rate_closed', 0) * 100:.1f}%")

    if not symbol_stats.empty:
        st.dataframe(symbol_stats, width="stretch")
    else:
        st.caption("No trigger outcome stats yet. PREP alerts are watch-only and ignored here.")

    st.subheader("ML learning dataset")
    ml_totals, ml_by_symbol = load_ml_dataset_stats(db_path, limit_days=outcome_days)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Feature rows", ml_totals.get("total", 0))
    m2.metric("Resolved triggers", ml_totals.get("trigger_resolved", 0))
    m3.metric("Trigger wins", ml_totals.get("trigger_wins", 0))
    m4.metric("Trigger losses", ml_totals.get("trigger_losses", 0))
    m5.metric("Training win rate", f"{ml_totals.get('trigger_win_rate', 0) * 100:.1f}%")

    if not ml_by_symbol.empty:
        st.dataframe(ml_by_symbol, width="stretch")
    else:
        st.caption("No ML feature rows yet. New signals will start filling this dataset.")

    st.divider()

    render_market_cards(snapshot, settings.symbols)

    st.divider()

    st.subheader("Recent stored signals")

    recent = load_recent_signals(db_path, limit=100)

    if recent:
        recent_df = pd.DataFrame(recent)

        preferred_cols = [
            "id",
            "symbol",
            "side",
            "alert_stage",
            "score",
            "created_epoch",
            "entry_zone_low",
            "entry_zone_high",
            "stop_loss",
            "take_profit_1",
            "take_profit_2",
            "outcome_status",
            "outcome_price",
            "outcome_reason",
        ]

        cols = [c for c in preferred_cols if c in recent_df.columns]
        remaining = [c for c in recent_df.columns if c not in cols]

        st.dataframe(recent_df[cols + remaining], width="stretch")
    else:
        st.caption("No signals stored yet.")

    time.sleep(refresh)
    st.rerun()


if __name__ == "__main__":
    main()
