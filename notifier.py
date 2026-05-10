"""
Alert delivery: console, optional Telegram (HTTPS), optional macOS banners.

Signal-only by default — these are research notifications, not trade instructions.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import shutil
import subprocess

import httpx

from strategy import Signal

logger = logging.getLogger(__name__)


def format_signal_message(sig: Signal, risk_label: str = "Medium") -> str:
    reasons = "; ".join(sig.reasons[:6])
    warn = ""
    if sig.volatility_warning:
        warn = f"\nVolatility note: {sig.volatility_warning}"

    return (
        f"{sig.side} signal on {sig.symbol} — Score: {sig.score:.1f}/100\n"
        f"Reasons: {reasons}{warn}\n"
        f"Entry zone: {sig.entry_zone_low:.5f}–{sig.entry_zone_high:.5f}\n"
        f"SL: {sig.stop_loss:.5f}\n"
        f"TP1: {sig.take_profit_1:.5f}\n"
        f"TP2: {sig.take_profit_2:.5f}\n"
        f"Approx. R-multiple (TP1 vs SL distance): {sig.risk_reward:.2f}:1\n"
        f"Regime: {sig.regime}\n"
        f"Risk (qualitative): {risk_label}\n"
        f"Timeframe: {sig.timeframe}"
    )


def risk_bucket_from_score(score: float) -> str:
    if score >= 88:
        return "Medium–High (still not a guarantee)"
    if score >= 80:
        return "Medium"
    return "Elevated noise risk"


class Notifier:
    def __init__(self, settings) -> None:
        self.settings = settings

    async def broadcast(self, sig: Signal) -> None:
        text = format_signal_message(sig, risk_bucket_from_score(sig.score))
        logger.info("SIGNAL\n%s", text)

        if self.settings.telegram_bot_token and self.settings.telegram_chat_id:
            asyncio.create_task(self._send_telegram(text))

        self._maybe_desktop(text, sig)

    async def _send_telegram(self, text: str) -> None:
        token = self.settings.telegram_bot_token
        chat = self.settings.telegram_chat_id
        if not token or not chat:
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat,
            "text": text,
            "disable_web_page_preview": True,
        }
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
        except Exception:
            logger.exception("Telegram notification failed")

    def _maybe_desktop(self, text: str, sig: Signal) -> None:
        if platform.system() != "Darwin":
            return
        if shutil.which("osascript") is None:
            return

        short = f"{sig.side} {sig.symbol} · {sig.score:.0f}/100"
        script = f'display notification "{short}" with title "Deriv Signal Bot"'
        try:
            subprocess.run(
                ["osascript", "-e", script],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            logger.debug("Desktop notification failed — ignoring", exc_info=True)
