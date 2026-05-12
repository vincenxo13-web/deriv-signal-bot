"""
Telegram-first notification layer for signal-only alerts.
"""

from __future__ import annotations

import html
import logging
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx

from strategy import Signal

logger = logging.getLogger(__name__)


def _fmt_price(value: float) -> str:
    return f"{value:.5f}"


def _fmt_time(epoch: float, tz_name: str) -> tuple[str, str]:
    utc_dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    try:
        local_dt = utc_dt.astimezone(ZoneInfo(tz_name))
    except Exception:
        local_dt = utc_dt
    return local_dt.strftime("%Y-%m-%d %H:%M:%S %Z"), utc_dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def risk_bucket_from_score(score: float) -> str:
    if score >= 88:
        return "A+ / very selective"
    if score >= 80:
        return "A / strong"
    if score >= 72:
        return "B / valid but be patient"
    return "C / early/noisy"


def format_signal_message(sig: Signal, tz_name: str = "Asia/Kuala_Lumpur") -> str:
    """Short, fast-to-read Telegram signal message."""
    local_time, _ = _fmt_time(sig.timestamp_epoch, tz_name)
    stage = str(getattr(sig, "alert_stage", "SIGNAL")).upper()
    side = str(sig.side).upper()
    symbol = str(sig.symbol).upper()

    if symbol.startswith("BOOM"):
        icon = "🟢" if stage == "TRIGGER" else "🟡"
    elif symbol.startswith("CRASH"):
        icon = "🔴" if stage == "TRIGGER" else "🟡"
    else:
        icon = "⚪"

    if stage == "PREP":
        title = f"{icon} {symbol} {side} PREP / WATCH"
        note = "Watch only — wait for trigger confirmation."
    elif stage == "TRIGGER":
        title = f"{icon} {symbol} {side} TRIGGER"
        note = "Signal only — confirm before entering."
    else:
        title = f"{icon} {symbol} {side} SIGNAL"
        note = "Signal only — confirm before entering."

    reasons = sig.reasons[:4]
    reason_lines = "\n".join(f"• {html.escape(reason)}" for reason in reasons)

    volatility = ""
    if sig.volatility_warning:
        volatility = f"\nVolatility: {html.escape(sig.volatility_warning)}"

    msg = (
        f"<b>{html.escape(title)}</b> | <b>{sig.score:.0f}/100</b>\n\n"
        f"Entry: <code>{_fmt_price(sig.entry_zone_low)} - {_fmt_price(sig.entry_zone_high)}</code>\n"
        f"SL: <code>{_fmt_price(sig.stop_loss)}</code>\n"
        f"TP1: <code>{_fmt_price(sig.take_profit_1)}</code> | "
        f"TP2: <code>{_fmt_price(sig.take_profit_2)}</code>\n"
        f"R:R: <b>{sig.risk_reward:.2f}:1</b>\n"
    )

    bpr_line = ""
    bpr = getattr(sig, "bpr_context", None) or {}
    if isinstance(bpr, dict) and bpr:
        status = str(bpr.get("status", "UNKNOWN")).upper()
        if bool(bpr.get("aligned")):
            bpr_icon = "✅"
        elif status in {"NEAR", "NO_ZONE"}:
            bpr_icon = "⚠️"
        elif status == "OFF":
            bpr_icon = "➖"
        else:
            bpr_icon = "❌"
        zone_low = bpr.get("zone_low")
        zone_high = bpr.get("zone_high")
        distance = bpr.get("distance_atr")
        zone_text = ""
        if zone_low is not None and zone_high is not None:
            try:
                zone_text = f" | Zone: <code>{_fmt_price(float(zone_low))}-{_fmt_price(float(zone_high))}</code>"
            except (TypeError, ValueError):
                zone_text = ""
        dist_text = ""
        if distance is not None:
            try:
                dist_text = f" | Dist: {float(distance):.2f} ATR"
            except (TypeError, ValueError):
                dist_text = ""
        bpr_line = f"\nH4 BPR: {bpr_icon} <b>{html.escape(status)}</b>{zone_text}{dist_text}"

    if reason_lines:
        msg += f"\n<b>Why:</b>\n{reason_lines}\n"

    msg += (
        f"{bpr_line}"
        f"\nRegime: <code>{html.escape(str(sig.regime))}</code>"
        f"{volatility}"
        f"\nSent: <b>{html.escape(local_time)}</b>"
        f"\n\n⚠️ {html.escape(note)}"
    )

    return msg

def format_outcome_message(event: dict, tz_name: str = "Asia/Kuala_Lumpur") -> str:
    status = str(event.get("outcome_status", "UNKNOWN"))
    symbol = html.escape(str(event.get("symbol", "?")))
    side = html.escape(str(event.get("side", "?")))
    score = float(event.get("score", 0) or 0)
    sent_epoch = float(event.get("created_epoch", 0) or 0)
    outcome_epoch = float(event.get("outcome_epoch", 0) or 0)
    outcome_price = event.get("outcome_price")
    reason = html.escape(str(event.get("outcome_reason", "")))

    sent_local, sent_utc = _fmt_time(sent_epoch, tz_name)
    resolved_local, resolved_utc = _fmt_time(outcome_epoch, tz_name)

    if status == "WIN_TP2":
        icon = "🏆"
        title = "TP2 HIT"
    elif status == "WIN_TP1":
        icon = "✅"
        title = "TP1 HIT"
    elif status == "LOSS_SL":
        icon = "❌"
        title = "SL / INVALIDATION HIT"
    elif status == "LOSS_SL_AMBIGUOUS":
        icon = "⚠️"
        title = "AMBIGUOUS CANDLE — MARKED LOSS"
    elif status == "EXPIRED":
        icon = "⌛"
        title = "SIGNAL EXPIRED"
    else:
        icon = "ℹ️"
        title = status

    price_line = ""
    if outcome_price is not None:
        try:
            price_line = f"\n<b>Outcome price:</b> <code>{_fmt_price(float(outcome_price))}</code>"
        except (TypeError, ValueError):
            price_line = ""

    return (
        f"{icon} <b>Signal outcome: {html.escape(title)}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Symbol:</b> <code>{symbol}</code>\n"
        f"<b>Original action:</b> <b>{side}</b>\n"
        f"<b>Original score:</b> {score:.1f}/100\n"
        f"<b>Sent:</b> {html.escape(sent_local)}\n"
        f"<b>Resolved:</b> {html.escape(resolved_local)}\n"
        f"<b>UTC:</b> {html.escape(resolved_utc)}"
        f"{price_line}\n"
        f"<b>Reason:</b> {reason}\n\n"
        f"This is tracking only — use it to review signal quality."
    )


class Notifier:
    def __init__(self, settings) -> None:
        self.settings = settings

    async def send_text(self, text: str) -> bool:
        token = self.settings.telegram_bot_token
        chat = self.settings.telegram_chat_id
        if not token or not chat:
            logger.warning("Telegram credentials missing; set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
            return False

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat,
            "text": text,
            "parse_mode": self.settings.telegram_parse_mode,
            "disable_web_page_preview": True,
        }
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            return True
        except Exception:
            logger.exception("Telegram notification failed")
            return False

    async def send_startup_message(self) -> None:
        if not self.settings.notify_on_start:
            return
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        symbols = ", ".join(self.settings.symbols)
        text = (
            "✅ <b>Deriv signal bot started</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Mode:</b> Signal-only\n"
            f"<b>Symbols:</b> <code>{html.escape(symbols)}</code>\n"
            f"<b>Warmup:</b> {self.settings.signal_warmup_bars} x 1m candles\n"
            f"<b>Started:</b> {html.escape(now)}"
        )
        await self.send_text(text)

    async def broadcast(self, sig: Signal) -> None:
        text = format_signal_message(sig, self.settings.signal_timezone)
        logger.info("SIGNAL\n%s", text)
        await self.send_text(text)
        self._maybe_desktop(sig)


    async def broadcast_outcomes(self, events: list[dict]) -> None:
        if not events or not self.settings.notify_signal_outcomes:
            return
        for event in events:
            text = format_outcome_message(event, self.settings.signal_timezone)
            logger.info("SIGNAL OUTCOME\n%s", text)
            await self.send_text(text)

    def _maybe_desktop(self, sig: Signal) -> None:
        if platform.system() != "Darwin" or shutil.which("osascript") is None:
            return
        short = f"{sig.side} {sig.symbol} · {sig.score:.0f}/100"
        script = f'display notification "{short}" with title "Deriv Signal Bot"'
        try:
            subprocess.run(["osascript", "-e", script], check=False, capture_output=True, text=True, timeout=5)
        except Exception:
            logger.debug("Desktop notification failed — ignoring", exc_info=True)
