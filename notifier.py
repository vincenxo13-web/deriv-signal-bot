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
    stage = str(getattr(sig, "alert_stage", "TRIGGER")).upper()
    symbol = str(sig.symbol).upper()
    side = str(sig.side).upper()

    if symbol.startswith("BOOM"):
        icon = "🟢"
        market_hint = "Boom BUY setup"
    elif symbol.startswith("CRASH"):
        icon = "🔴"
        market_hint = "Crash SELL setup"
    else:
        icon = "⚪"
        market_hint = "Signal setup"

    title = f"{icon} <b>{_esc(symbol)} {side} {stage}</b> | <b>{sig.score:.0f}/100</b>"

    entry_rule = getattr(sig, "entry_rule", None) or (
        "Wait for price to hold/reclaim the entry zone before entering."
        if side == "BUY"
        else "Wait for price to reject/hold below the entry zone before entering."
    )
    entry_validity = getattr(sig, "entry_validity", None) or "Use the SL/invalidation level if the setup fails."
    confirmation = getattr(sig, "confirmation_summary", None) or "Confirm with chart before entering."

    reasons = getattr(sig, "reasons", []) or []
    reason_lines = "\n".join(f"• {_esc(reason)}" for reason in reasons[:5])
    if not reason_lines:
        reason_lines = "• No extra reason text available"

    warning_lines: list[str] = []
    if getattr(sig, "volatility_warning", None):
        warning_lines.append(f"Volatility: {_esc(sig.volatility_warning)}")
    if "high_volatility" in str(sig.regime).lower():
        warning_lines.append("High-volatility regime — wait for cleaner confirmation or use smaller size.")
    if "downtrend" in str(sig.regime).lower() and side == "BUY":
        warning_lines.append("Regime conflict: Boom BUY while regime says downtrend.")
    if "uptrend" in str(sig.regime).lower() and side == "SELL":
        warning_lines.append("Regime conflict: Crash SELL while regime says uptrend.")

    warning_block = ""
    if warning_lines:
        warning_block = "\n\n⚠️ <b>Warnings</b>\n" + "\n".join(f"• {_esc(w)}" for w in warning_lines)

    return (
        f"{title}\n\n"
        f"📍 <b>Entry zone</b>\n"
        f"<code>{_fmt_price(sig.entry_zone_low)} – {_fmt_price(sig.entry_zone_high)}</code>\n\n"
        f"✅ <b>Entry rule</b>\n"
        f"{_esc(entry_rule)}\n\n"
        f"🧭 <b>Entry validity</b>\n"
        f"{_esc(entry_validity)}\n\n"
        f"🛑 <b>SL</b>: <code>{_fmt_price(sig.stop_loss)}</code>\n"
        f"✅ <b>TP1</b>: <code>{_fmt_price(sig.take_profit_1)}</code>\n"
        f"✅ <b>TP2</b>: <code>{_fmt_price(sig.take_profit_2)}</code>\n"
        f"📐 <b>R:R</b>: {sig.risk_reward:.2f}:1\n\n"
        f"🧩 <b>Confirmation</b>\n"
        f"{_esc(confirmation)}\n\n"
        f"{_bpr_line(sig)}\n\n"
        f"📌 <b>Why</b>\n"
        f"{reason_lines}\n\n"
        f"📊 <b>Regime</b>: <code>{_esc(sig.regime)}</code>\n"
        f"⚖️ <b>Risk</b>: {_esc(risk_label)}\n"
        f"⏱ <b>Sent</b>: {_esc(_sent_time(sig))}"
        f"{warning_block}\n\n"
        f"⚠️ Signal only — confirm with chart before entering."
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
