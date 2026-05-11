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
    local_time, utc_time = _fmt_time(sig.timestamp_epoch, tz_name)
    stage = getattr(sig, "alert_stage", "PREP")

    if stage == "TRIGGER":
        arrow = "🟢" if sig.side == "BUY" else "🔴"
        title = "BOOM SPIKE TRIGGER" if sig.side == "BUY" else "CRASH DROP TRIGGER"
        stage_line = "Stage 2 / TRIGGER — spike confirmation is active"
        action_hint = "Act only if your chart confirms entry. This can move fast."
    else:
        arrow = "🟡"
        title = "BOOM BUY PREPARATION" if sig.side == "BUY" else "CRASH SELL PREPARATION"
        stage_line = "Stage 1 / PREPARATION — setup is forming"
        action_hint = "Open the chart and wait for confirmation before entering."

    reasons = sig.reasons[:7]
    reason_lines = "\n".join(f"• {html.escape(reason)}" for reason in reasons)
    volatility = (
        f"\n⚠️ <b>Volatility note:</b> {html.escape(sig.volatility_warning)}"
        if sig.volatility_warning
        else ""
    )

    return (
        f"{arrow} <b>{html.escape(title)}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Symbol:</b> <code>{html.escape(sig.symbol)}</code>\n"
        f"<b>Action:</b> <b>{html.escape(sig.side)}</b> only\n"
        f"<b>Score:</b> <b>{sig.score:.1f}/100</b> ({html.escape(risk_bucket_from_score(sig.score))})\n"
        f"<b>Sent:</b> {html.escape(local_time)}\n"
        f"<b>UTC:</b> {html.escape(utc_time)}\n\n"
        f"🎯 <b>Sniper entry zone</b>\n"
        f"<code>{_fmt_price(sig.entry_zone_low)} - {_fmt_price(sig.entry_zone_high)}</code>\n\n"
        f"🛑 <b>Invalidation / SL idea:</b> <code>{_fmt_price(sig.stop_loss)}</code>\n"
        f"✅ <b>TP1:</b> <code>{_fmt_price(sig.take_profit_1)}</code>\n"
        f"✅ <b>TP2:</b> <code>{_fmt_price(sig.take_profit_2)}</code>\n"
        f"📐 <b>Approx R:R:</b> {sig.risk_reward:.2f}:1\n\n"
        f"📌 <b>Why this alert fired</b>\n"
        f"{reason_lines}"
        f"{volatility}\n\n"
        f"📊 <b>Regime:</b> <code>{html.escape(sig.regime)}</code>\n"
        f"⏱ <b>Timeframe:</b> {html.escape(sig.timeframe)}\n\n"
        f"⚠️ Signal only — wait for your own confirmation before entering."
    )


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
