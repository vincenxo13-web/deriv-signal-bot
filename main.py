"""
Entry point for the Deriv Crash / Boom **signal-only** bot.

This program streams ticks, aggregates candles, scores setups, persists state,
and sends alerts — it does **not** place trades unless you later integrate
execution flows with every safety guard engaged.

Trading synthetic indices is extremely risky; past performance does not predict
future results. Use a Demo account while learning.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from config import get_settings
from deriv_client import DerivWebSocketClient, fetch_active_symbols_brief
from market_stream import MarketStreamRouter
from notifier import Notifier
from risk_manager import RiskManager
from signal_engine import SignalEngine
from storage import Storage


def setup_logging(level: str) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "bot.log", encoding="utf-8")
    fh.setFormatter(formatter)
    root.addHandler(fh)


def banner() -> None:
    logging.warning("%s", "=" * 76)
    logging.warning(
        "Deriv Signal Bot starting — SIGNAL / RESEARCH USE ONLY.",
    )
    logging.warning(
        "No outcome is guaranteed. Synthetic indices move sharply; you can lose "
        "your entire stake rapidly.",
    )
    logging.warning(
        "Automated BUY/SELL via API stays DISABLED unless you deliberately satisfy "
        "every execution guard in README + config.",
    )
    logging.warning("%s", "=" * 76)


async def validate_symbols(settings) -> None:
    """Optional helper — Deriv symbol names vary; validate against brief list."""
    try:
        avail = await fetch_active_symbols_brief(settings)
        want = {s.upper(): s for s in settings.symbols}
        avail_set = {a.upper() for a in avail}
        missing = [want[k] for k in want.keys() if k not in avail_set]
        if missing:
            logging.warning(
                "The following SYMBOLS entries were not found in active_symbols brief: %s "
                "(this endpoint may truncate — double-check Trader UI naming).",
                ", ".join(missing),
            )
        else:
            logging.info("Symbol brief check matched all configured symbols.")
    except Exception:
        logging.exception("Symbol validation skipped due to connectivity error")


async def runner(validate_only: bool) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    banner()

    storage = Storage(settings.data_db_path)
    await storage.set_meta(
        "bot_status",
        {
            "state": "starting",
            "ts": time.time(),
            "mode": settings.mode,
        },
    )

    if validate_only:
        await validate_symbols(settings)
        return

    asyncio.create_task(heartbeat(storage, settings.symbols))

    risk = RiskManager(settings)
    notifier = Notifier(settings)
    engine = SignalEngine(storage, risk, notifier, settings)
    router = MarketStreamRouter(
        storage,
        settings,
        on_bar_closed=engine.on_bar_closed,
    )

    client = DerivWebSocketClient(settings)

    async def handle_message(msg: dict) -> None:
        await router.handle_deriv_message(msg)
        await storage.set_meta(
            "bot_status",
            {
                "state": "running",
                "last_msg_epoch": time.time(),
                "symbols": list(settings.symbols),
            },
        )

    await client.stream_ticks(list(settings.symbols), handle_message)


async def heartbeat(storage: Storage, symbols: tuple[str, ...]) -> None:
    """Lightweight periodic writer so the dashboard shows liveness even if ticks pause."""
    while True:
        await asyncio.sleep(30)
        await storage.set_meta(
            "bot_heartbeat",
            {"epoch": time.time(), "symbols": list(symbols)},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deriv Crash/Boom signal bot")
    parser.add_argument(
        "--validate-symbols",
        action="store_true",
        help="Fetch active_symbols brief from Deriv and compare with .env SYMBOLS",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(runner(validate_only=args.validate_symbols))
    except KeyboardInterrupt:
        print("\nStopped by user — goodbye.")


if __name__ == "__main__":
    main()
