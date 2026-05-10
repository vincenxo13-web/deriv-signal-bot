"""
Offline research helper — replay candles through the scoring engine.

Usage examples:

  python backtester.py --csv data/history.csv --symbol CRASH500 --min-score 75
  python backtester.py --sqlite data/deriv_signals.db --symbol BOOM300

CSV expects columns:

  datetime,symbol,open,high,low,close(,volume optional)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import get_settings
from strategy import SpikeContext, evaluate_signal


def load_from_csv(path: Path, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        raise ValueError("CSV must include a datetime column (UTC timestamps).")

    dt = pd.to_datetime(df["datetime"], utc=True)
    df = df.copy()
    df["datetime"] = dt
    if "symbol" in df.columns:
        df = df[df["symbol"].str.upper() == symbol.upper()].copy()

    df = df.sort_values("datetime")
    df.set_index("datetime", inplace=True)
    return df[["open", "high", "low", "close"]]


def load_from_sqlite(db_path: Path, symbol: str) -> pd.DataFrame:
    import sqlite3

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT bucket_epoch, open, high, low, close
        FROM candles
        WHERE UPPER(symbol) = UPPER(?) AND timeframe = '1m'
        ORDER BY bucket_epoch ASC
        """,
        (symbol,),
    ).fetchall()
    conn.close()
    if not rows:
        raise ValueError(f"No 1m candles stored yet for {symbol} in database.")
    df = pd.DataFrame(
        rows,
        columns=["bucket_epoch", "open", "high", "low", "close"],
    )
    df["datetime"] = pd.to_datetime(df["bucket_epoch"], unit="s", utc=True)
    df.set_index("datetime", inplace=True)
    return df[["open", "high", "low", "close"]]


def simulate_trade(sig, future: pd.DataFrame) -> str:
    """
    Very simple OHLC traversal — assumes intra-bar extremes can hit TP/SL (conservative ambiguity).

    Returns 'win','loss','open' status string.
    """
    sl = sig.stop_loss
    tp = sig.take_profit_1
    for _, row in future.iloc[:120].iterrows():  # up to ~2h forward on 1m bars
        if sig.side == "BUY":
            if row.low <= sl:
                return "loss"
            if row.high >= tp:
                return "win"
        else:
            if row.high >= sl:
                return "loss"
            if row.low <= tp:
                return "win"
    return "open"


def run_backtest(df: pd.DataFrame, symbol: str, min_score: float, export: Path | None) -> dict:
    results: list[dict] = []
    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    start = 220
    neutral_spike = SpikeContext(None, "none", 0.0, 0.0)

    for i in range(start, len(df) - 2):
        window = df.iloc[: i + 1]
        sig = evaluate_signal(
            symbol=symbol,
            df_1m=window,
            spike_ctx=neutral_spike,
            min_score=min_score,
            now_epoch=float(window.index[-1].timestamp()),
        )
        if sig is None:
            continue

        future = df.iloc[i + 1 :]
        outcome = simulate_trade(sig, future)
        pnl = 0.0
        if outcome == "win":
            pnl = abs(sig.take_profit_1 - window.close.iloc[-1])
        elif outcome == "loss":
            pnl = -abs(window.close.iloc[-1] - sig.stop_loss)

        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

        results.append(
            {
                "timestamp": window.index[-1].isoformat(),
                "side": sig.side,
                "score": sig.score,
                "outcome": outcome,
                "pnl_points": pnl,
                "rr": sig.risk_reward,
            }
        )

    wins = [r for r in results if r["outcome"] == "win"]
    losses = [r for r in results if r["outcome"] == "loss"]
    gross_win = sum(r["pnl_points"] for r in wins)
    gross_loss = abs(sum(r["pnl_points"] for r in losses))
    profit_factor = (
        gross_win / gross_loss if gross_loss > 1e-9 else float("inf") if gross_win > 0 else 0.0
    )

    metrics = {
        "signals": len(results),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / max(len(results), 1),
        "profit_factor": float(profit_factor),
        "max_drawdown_points": float(max_dd),
        "avg_rr": float(np.mean([r["rr"] for r in results])) if results else 0.0,
        "gross_win_points": float(gross_win),
        "gross_loss_points": float(gross_loss),
    }

    if export and results:
        pd.DataFrame(results).to_csv(export, index=False)

    return {"metrics": metrics, "trades": results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest Deriv signal engine on saved candles")
    parser.add_argument("--symbol", required=True, help="Symbol such as CRASH500")
    parser.add_argument("--csv", type=Path, help="CSV with OHLC history")
    parser.add_argument("--sqlite", type=Path, help="SQLite DB written by the live bot")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--export", type=Path, default=Path("data/backtest_trades.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    min_score = args.min_score or settings.min_signal_score

    if args.csv:
        df = load_from_csv(args.csv, args.symbol)
    elif args.sqlite:
        df = load_from_sqlite(args.sqlite, args.symbol)
    else:
        raise SystemExit("Provide --csv or --sqlite")

    report = run_backtest(df, args.symbol.upper(), min_score, args.export)
    print(json.dumps(report["metrics"], indent=2))
    if report["trades"]:
        print(f"Exported trades to {args.export}")


if __name__ == "__main__":
    main()
