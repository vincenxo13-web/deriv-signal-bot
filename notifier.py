"""
Alert delivery: console, optional Telegram (HTTPS), optional macOS banners.

Signal-only by default — these are research notifications, not trade instructions.
"""

from __future__ import annotations

import asyncio
import html
import logging
import platform
import shutil
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from strategy import Signal

logger = logging.getLogger(__name__)


def _esc(value) -> str:
    return html.escape(str(value), quote=False)


def _fmt_price(value: float | None) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.5f}"
    except Exception:
        return "n/a"


def _sent_time(sig: Signal, tz_name: str = "Asia/Kuala_Lumpur") -> str:
    try:
        dt = datetime.fromtimestamp(float(sig.timestamp_epoch), ZoneInfo(tz_name))
        return dt.strftime("%Y-%m-%d %H:%M:%S %z")
    except Exception:
        return "n/a"


def _bpr_line(sig: Signal) -> str:
    ctx = getattr(sig, "bpr_context", None) or {}
    status = str(ctx.get("status", "NO_DATA")).upper()
    if status in {"ALIGNED", "NEAR"}:
        icon = "✅" if status == "ALIGNED" else "⚠️"
    elif status in {"FAR", "NO_ZONE", "NO_DATA"}:
        icon = "❌" if status in {"FAR", "NO_DATA"} else "⚠️"
    else:
        icon = "ℹ️"
    note = ctx.get("note") or "n/a"
    return f"{icon} H4 BPR: <b>{_esc(status)}</b> — {_esc(note)}"


def risk_bucket_from_score(score: float) -> str:
    if score >= 88:
        return "Medium–High"
    if score >= 80:
        return "Medium"
    return "Elevated noise risk"


def format_signal_message(sig: Signal, risk_label: str = "Medium") -> str:
    """Compact Telegram message for fast trading decisions."""
    stage = str(getattr(sig, "alert_stage", "TRIGGER")).upper()
    symbol = str(sig.symbol).upper()
    side = str(sig.side).upper()
    icon = "🟢" if side == "BUY" else "🔴" if side == "SELL" else "⚪"

    confirmation = getattr(sig, "confirmation_summary", None) or "confirm on chart"
    regime = str(getattr(sig, "regime", "unknown"))
    warning = ""
    if "high_volatility" in regime.lower():
        warning = "\n⚠️ High volatility — use smaller size / wait for cleaner retest"

    bpr = getattr(sig, "bpr_context", None) or {}
    bpr_status = str(bpr.get("status", "NO_DATA")).upper()

    return (
        f"{icon} <b>{_esc(symbol)} {side} {stage}</b> | <b>{sig.score:.0f}/100</b>\n\n"
        f"Entry: <code>{_fmt_price(sig.entry_zone_low)}–{_fmt_price(sig.entry_zone_high)}</code>\n"
        f"SL: <code>{_fmt_price(sig.stop_loss)}</code>\n"
        f"TP1: <code>{_fmt_price(sig.take_profit_1)}</code> | TP2: <code>{_fmt_price(sig.take_profit_2)}</code>\n"
        f"R:R: <b>{sig.risk_reward:.2f}:1</b>\n\n"
        f"Rule: {_esc(getattr(sig, 'entry_rule', None) or ('hold/reclaim zone' if side == 'BUY' else 'reject/hold below zone'))}\n"
        f"Confirm: <b>{_esc(confirmation)}</b>\n"
        f"Regime: <code>{_esc(regime)}</code>\n"
        f"BPR: {_esc(bpr_status)}"
        f"{warning}\n\n"
        f"Sent: {_esc(_sent_time(sig))}\n"
        f"Signal only."
    )

class Notifier:
    def __init__(self, settings) -> None:
        self.settings = settings

    async def send_text(self, text: str) -> None:
        logger.info("NOTIFY\n%s", text)
        if self.settings.telegram_bot_token and self.settings.telegram_chat_id:
            await self._send_telegram(text)

    async def send_startup_message(self) -> None:
        symbols = ", ".join(getattr(self.settings, "symbols", ()) or ())
        text = (
            "✅ <b>Deriv Signal Bot started</b>\n\n"
            "Mode: TELEGRAM SIGNALS ONLY\n"
            "Boom = BUY only\n"
            "Crash = SELL only"
        )
        if symbols:
            text += f"\n\nTracking: {_esc(symbols)}"
        await self.send_text(text)

    async def broadcast(self, sig: Signal) -> None:
        text = format_signal_message(sig, risk_bucket_from_score(sig.score))
        logger.info("SIGNAL\n%s", text)
        if self.settings.telegram_bot_token and self.settings.telegram_chat_id:
            asyncio.create_task(self._send_telegram(text))
        self._maybe_desktop(text, sig)

    async def broadcast_outcomes(self, events: list[dict]) -> None:
        """Send result notifications. Safe to call only when outcome notifications are enabled."""
        if not events:
            return
        for event in events:
            try:
                symbol = str(event.get("symbol", "UNKNOWN")).upper()
                side = str(event.get("side", "")).upper()
                outcome = str(event.get("outcome_status", event.get("status", "RESOLVED"))).upper()
                price = event.get("outcome_price")
                reason = event.get("outcome_reason") or event.get("outcome_note") or ""
                icon = "✅" if outcome.startswith("WIN") else "❌" if outcome.startswith("LOSS") else "⌛" if outcome == "EXPIRED" else "ℹ️"
                msg = f"{icon} <b>{_esc(symbol)} {side} RESULT</b>\n\nOutcome: <b>{_esc(outcome)}</b>"
                if price is not None:
                    msg += f"\nPrice: <code>{_fmt_price(price)}</code>"
                if reason:
                    msg += f"\nNote: {_esc(reason)}"
                await self.send_text(msg)
            except Exception:
                logger.exception("Failed to broadcast outcome event: %s", event)

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
        parse_mode = getattr(self.settings, "telegram_parse_mode", None)
        if parse_mode:
            payload["parse_mode"] = parse_mode
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
            subprocess.run(["osascript", "-e", script], check=False, capture_output=True, text=True, timeout=5)
        except Exception:
            logger.debug("Desktop notification failed — ignoring", exc_info=True)
