"""
Prepare an ML training dataset from logged signal features.

Stage 1 does not train or deploy a model yet. It exports clean resolved
TRIGGER examples so we can inspect whether the dataset is large enough.

Examples:
  python ml_trainer.py --summary
  python ml_trainer.py --export data/ml_training_rows.csv
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from config import get_settings
from ml_features import outcome_to_success


RESOLVED_STATUSES = (
    "WIN_TP1",
    "WIN_TP2",
    "LOSS_SL",
    "LOSS_SL_AMBIGUOUS",
    "EXPIRED",
)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def load_training_rows(db_path: Path, include_expired: bool = True) -> pd.DataFrame:
    statuses = list(RESOLVED_STATUSES)
    if not include_expired:
        statuses.remove("EXPIRED")

    placeholders = ",".join("?" for _ in statuses)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if not _table_exists(conn, "signal_features"):
        conn.close()
        return pd.DataFrame()

    rows = conn.execute(
        f"""
        SELECT
            signal_id, symbol, side, alert_stage, score,
            features_json, created_epoch, outcome_status,
            outcome_epoch, outcome_price, outcome_reason
        FROM signal_features
        WHERE alert_stage = 'TRIGGER'
          AND outcome_status IN ({placeholders})
        ORDER BY created_epoch ASC
        """,
        statuses,
    ).fetchall()
    conn.close()

    out: list[dict[str, Any]] = []

    for row in rows:
        base = dict(row)
        try:
            features = json.loads(base.pop("features_json") or "{}")
        except json.JSONDecodeError:
            features = {}

        success = outcome_to_success(str(base.get("outcome_status")))
        if success is None:
            continue

        item: dict[str, Any] = {}
        item.update(features)
        item.update(base)
        item["success"] = success
        out.append(item)

    return pd.DataFrame(out)


def print_summary(df: pd.DataFrame, min_samples: int) -> None:
    if df.empty:
        print("No resolved TRIGGER feature rows found yet.")
        print("Let the bot run until several TRIGGER signals resolve to TP/SL/EXPIRED.")
        return

    total = len(df)
    wins = int(df["success"].sum())
    losses = total - wins
    win_rate = wins / max(1, total)

    print("ML dataset summary")
    print("==================")
    print(f"Rows: {total}")
    print(f"Wins: {wins}")
    print(f"Losses/expired: {losses}")
    print(f"Win rate: {win_rate:.1%}")
    print(f"Minimum suggested samples before ML filtering: {min_samples}")

    if total < min_samples:
        print("Status: collect more data before training a live filter.")
    else:
        print("Status: enough rows to begin offline model experiments.")

    if "symbol" in df.columns:
        print("\nBy symbol:")
        grouped = df.groupby("symbol")["success"].agg(["count", "sum", "mean"])
        grouped = grouped.rename(columns={"sum": "wins", "mean": "win_rate"})
        print(grouped.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ML dataset from signal features")
    parser.add_argument("--db", type=Path, default=None, help="Path to SQLite DB")
    parser.add_argument("--export", type=Path, default=None, help="Optional CSV export path")
    parser.add_argument("--summary", action="store_true", help="Print dataset summary")
    parser.add_argument("--exclude-expired", action="store_true", help="Exclude EXPIRED labels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    db_path = args.db or settings.data_db_path

    df = load_training_rows(db_path, include_expired=not args.exclude_expired)

    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.export, index=False)
        print(f"Exported {len(df)} rows to {args.export}")

    if args.summary or not args.export:
        print_summary(df, settings.ml_training_min_samples)


if __name__ == "__main__":
    main()
