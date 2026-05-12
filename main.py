"""
Entry point for the Deriv Crash/Boom signal-only Telegram bot.

The bot streams public Deriv ticks, builds candles, detects sniper-style setups,
persists state, and sends clean Telegram alerts. It does not place trades.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from config import get_data_dir, get_settings
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

    log_dir = get_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "bot.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def banner() -> None:
    logging.warning("%s", "=" * 76)
    logging.warning("Deriv Signal Bot starting — TELEGRAM SIGNALS ONLY.")
    logging.warning(
        "Boom symbols will only generate BUY alerts. Crash symbols will only generate SELL alerts."
    )
    logging.warning("No auto-trading code is active in this build.")
    logging.warning("%s", "=" * 76)


async def validate_symbols(settings) -> None:
    """Optional helper — Deriv symbol names vary; validate against brief list."""
    try:
        available = await fetch_active_symbols_brief(settings)
        wanted = {symbol.upper(): symbol for symbol in settings.symbols}
        available_set = {symbol.upper() for symbol in available}
        missing = [wanted[key] for key in wanted if key not in available_set]
        if missing:
            logging.warning(
                "These SYMBOLS entries were not found: %s. Double-check Deriv symbol names.",
                ", ".join(missing),
            )
        else:
            logging.info("Symbol check matched all configured symbols.")
    except Exception:
        logging.exception("Symbol validation skipped due to connectivity error")


async def runner(validate_only: bool) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    banner()

    storage = Storage(settings.data_db_path)
    notifier = Notifier(settings)

    await storage.set_meta(
        "bot_status",
        {
            "state": "starting",
            "ts": time.time(),
            "mode": "signal_only",
            "symbols": list(settings.symbols),
        },
    )

    if validate_only:
        await validate_symbols(settings)
        return

    await notifier.send_startup_message()
    asyncio.create_task(heartbeat(storage, settings.symbols))

    risk = RiskManager(settings)
    engine = SignalEngine(storage, risk, notifier, settings)
    router = MarketStreamRouter(
        storage,
        settings,
        on_bar_closed=engine.on_bar_closed,
    )

    await router.warm_start_from_storage()

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
    """Periodic liveness writer for dashboard."""
    while True:
        await asyncio.sleep(30)
        await storage.set_meta(
            "bot_heartbeat",
            {"epoch": time.time(), "symbols": list(symbols)},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deriv Crash/Boom signal-only bot")
    parser.add_argument(
        "--validate-symbols",
        action="store_true",
        help="Fetch active_symbols brief from Deriv and compare with SYMBOLS",
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
