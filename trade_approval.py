"""
Telegram approval flow for demo trade execution.

A qualifying signal is saved as a pending approval and sent to Telegram with
Approve / Reject buttons. Only an approval from TELEGRAM_CHAT_ID will trigger a
Deriv proposal + buy request.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import asdict
from typing import Any

import httpx

from config import Settings, execution_allowed, get_settings
from deriv_client import DerivWebSocketClient
from storage import Storage
from strategy import Signal

logger = logging.getLogger(__name__)


class TelegramApprovalTrader:
    def __init__(
        self,
        storage: Storage,
        settings: Settings | None = None,
        client: DerivWebSocketClient | None = None,
    ) -> None:
        self.storage = storage
        self.settings = settings or get_settings()
        self.client = client or DerivWebSocketClient(self.settings)
        self._offset: int | None = None
        self._running = False

    def execution_mode_enabled(self) -> bool:
        ok, _ = execution_allowed(self.settings)
        return ok and self.settings.mode.strip().lower() in {"approval_trade", "auto_trade"}

    async def submit_signal(self, sig: Signal) -> None:
        """Create a pending trade approval for a qualifying signal."""
        if not self.execution_mode_enabled():
            return

        approval_id = uuid.uuid4().hex[:12]
        signal_payload = asdict(sig)
        await self.storage.create_trade_approval(
            {
                "approval_id": approval_id,
                "status": "pending" if self.settings.trade_approval_required else "auto_queued",
                "symbol": sig.symbol,
                "side": sig.side,
                "score": sig.score,
                "stake": self.settings.trade_stake,
                "currency": self.settings.trade_currency,
                "duration": self.settings.trade_duration,
                "duration_unit": self.settings.trade_duration_unit,
                "signal": signal_payload,
                "created_epoch": time.time(),
                "updated_epoch": time.time(),
            }
        )

        if self.settings.trade_approval_required:
            await self._send_approval_message(approval_id, sig)
        else:
            await self._execute_approval(approval_id, source="auto")

    async def poll_loop(self) -> None:
        """Long-running Telegram polling loop for approval button callbacks."""
        if not self.execution_mode_enabled():
            logger.info("Telegram trade approval disabled: execution guard is not enabled.")
            return
        if not self.settings.trade_approval_required:
            logger.info("Telegram approval polling disabled: TRADE_APPROVAL_REQUIRED=false.")
            return
        if not self.settings.telegram_bot_token or not self.settings.telegram_chat_id:
            logger.warning("Telegram approval disabled: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing.")
            return

        self._running = True
        logger.info("Telegram approval polling started.")
        while self._running:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Telegram approval polling error")
                await asyncio.sleep(5)
            await asyncio.sleep(self.settings.telegram_poll_seconds)

    def stop(self) -> None:
        self._running = False

    async def _poll_once(self) -> None:
        token = self.settings.telegram_bot_token
        if not token:
            return

        url = f"https://api.telegram.org/bot{token}/getUpdates"
        params: dict[str, Any] = {
            "timeout": 10,
            "allowed_updates": ["callback_query"],
        }
        if self._offset is not None:
            params["offset"] = self._offset

        async with httpx.AsyncClient(timeout=20.0) as http:
            resp = await http.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()

        if not payload.get("ok"):
            logger.warning("Telegram getUpdates returned non-ok payload: %s", payload)
            return

        for update in payload.get("result", []):
            update_id = int(update.get("update_id", 0))
            self._offset = max(self._offset or 0, update_id + 1)
            callback = update.get("callback_query")
            if callback:
                await self._handle_callback(callback)

    async def _handle_callback(self, callback: dict[str, Any]) -> None:
        callback_id = str(callback.get("id", ""))
        data = str(callback.get("data", ""))
        message = callback.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = str(chat.get("id", ""))

        if chat_id != str(self.settings.telegram_chat_id):
            await self._answer_callback(callback_id, "Not authorised for this bot.")
            logger.warning("Rejected Telegram approval from unauthorised chat_id=%s", chat_id)
            return

        if ":" not in data:
            await self._answer_callback(callback_id, "Invalid approval button.")
            return

        action, approval_id = data.split(":", 1)
        approval = await self.storage.get_trade_approval(approval_id)
        if approval is None:
            await self._answer_callback(callback_id, "Approval not found or expired.")
            return
        if approval.get("status") not in {"pending", "auto_queued"}:
            await self._answer_callback(callback_id, f"Already {approval.get('status')}.")
            return

        if action == "reject":
            await self.storage.update_trade_approval(approval_id, "rejected", note="Rejected in Telegram")
            await self._answer_callback(callback_id, "Trade rejected.")
            await self._send_text(f"Rejected trade approval `{approval_id}` for {approval['side']} {approval['symbol']}.")
            return

        if action != "approve":
            await self._answer_callback(callback_id, "Unknown action.")
            return

        await self._answer_callback(callback_id, "Approved. Sending demo order...")
        await self._execute_approval(approval_id, source="telegram")

    async def _execute_approval(self, approval_id: str, source: str) -> None:
        approval = await self.storage.get_trade_approval(approval_id)
        if approval is None:
            logger.warning("Approval %s not found for execution", approval_id)
            return
        if approval.get("status") not in {"pending", "auto_queued"}:
            logger.info("Approval %s already processed with status=%s", approval_id, approval.get("status"))
            return

        await self.storage.update_trade_approval(approval_id, "executing", note=f"Approved via {source}")
        request_summary = {
            "symbol": approval["symbol"],
            "side": approval["side"],
            "stake": approval["stake"],
            "currency": approval["currency"],
            "duration": approval["duration"],
            "duration_unit": approval["duration_unit"],
            "basis": self.settings.trade_basis,
            "max_price": self.settings.trade_max_price,
        }

        try:
            result = await self.client.execute_rise_fall_contract(
                symbol=approval["symbol"],
                side=approval["side"],
                stake=float(approval["stake"]),
                currency=str(approval["currency"]),
                duration=int(approval["duration"]),
                duration_unit=str(approval["duration_unit"]),
                basis=self.settings.trade_basis,
                max_price=self.settings.trade_max_price,
            )
            buy_response = result.get("buy_response") or {}
            proposal_response = result.get("proposal_response") or {}
            has_error = bool(buy_response.get("error") or proposal_response.get("error"))
            success = bool(buy_response.get("buy")) and not has_error
            status = "executed" if success else "failed"
            note = "Deriv buy confirmed" if success else "Deriv proposal/buy failed"
            await self.storage.insert_trade_execution(
                {
                    "approval_id": approval_id,
                    "status": status,
                    "request": request_summary,
                    "response": result,
                    "created_epoch": time.time(),
                }
            )
            await self.storage.update_trade_approval(approval_id, status, note=note)
            await self._send_execution_result(approval_id, approval, result, success)
        except Exception as exc:
            logger.exception("Trade execution failed for approval %s", approval_id)
            result = {"error": {"message": str(exc)}}
            await self.storage.insert_trade_execution(
                {
                    "approval_id": approval_id,
                    "status": "failed",
                    "request": request_summary,
                    "response": result,
                    "created_epoch": time.time(),
                }
            )
            await self.storage.update_trade_approval(approval_id, "failed", note=str(exc))
            await self._send_text(f"Trade approval `{approval_id}` failed: {exc}")

    async def _send_approval_message(self, approval_id: str, sig: Signal) -> None:
        if not self.settings.telegram_bot_token or not self.settings.telegram_chat_id:
            logger.warning("Cannot send approval message: Telegram credentials missing")
            return

        text = (
            "DEMO TRADE APPROVAL REQUIRED\n"
            f"ID: `{approval_id}`\n"
            f"Signal: {sig.side} {sig.symbol}\n"
            f"Score: {sig.score:.1f}/100\n"
            f"Stake: {self.settings.trade_stake:.2f} {self.settings.trade_currency}\n"
            f"Duration: {self.settings.trade_duration}{self.settings.trade_duration_unit}\n"
            f"Entry zone: {sig.entry_zone_low:.5f}–{sig.entry_zone_high:.5f}\n"
            f"SL: {sig.stop_loss:.5f}\n"
            f"TP1: {sig.take_profit_1:.5f}\n"
            "\nTap approve only if you want the bot to place this demo contract."
        )
        reply_markup = {
            "inline_keyboard": [
                [
                    {"text": f"Approve {sig.side}", "callback_data": f"approve:{approval_id}"},
                    {"text": "Reject", "callback_data": f"reject:{approval_id}"},
                ]
            ]
        }
        response = await self._telegram_post(
            "sendMessage",
            {
                "chat_id": self.settings.telegram_chat_id,
                "text": text,
                "parse_mode": "Markdown",
                "reply_markup": reply_markup,
                "disable_web_page_preview": True,
            },
        )
        message_id = ((response.get("result") or {}).get("message_id")) if response else None
        if message_id is not None:
            await self.storage.update_trade_approval(
                approval_id,
                "pending",
                note="Approval message sent to Telegram",
                telegram_message_id=str(message_id),
            )

    async def _send_execution_result(
        self,
        approval_id: str,
        approval: dict[str, Any],
        result: dict[str, Any],
        success: bool,
    ) -> None:
        if success:
            buy = (result.get("buy_response") or {}).get("buy") or {}
            contract_id = buy.get("contract_id", "n/a")
            buy_price = buy.get("buy_price", "n/a")
            text = (
                f"Demo trade executed for `{approval_id}`\n"
                f"{approval['side']} {approval['symbol']}\n"
                f"Contract ID: `{contract_id}`\n"
                f"Buy price: {buy_price} {approval['currency']}"
            )
        else:
            error = (result.get("buy_response") or {}).get("error") or (result.get("proposal_response") or {}).get("error") or {}
            msg = error.get("message", "Unknown Deriv error")
            text = f"Demo trade failed for `{approval_id}`: {msg}"
        await self._send_text(text)

    async def _send_text(self, text: str) -> None:
        await self._telegram_post(
            "sendMessage",
            {
                "chat_id": self.settings.telegram_chat_id,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
        )

    async def _answer_callback(self, callback_id: str, text: str) -> None:
        if not callback_id:
            return
        await self._telegram_post(
            "answerCallbackQuery",
            {"callback_query_id": callback_id, "text": text, "show_alert": False},
        )

    async def _telegram_post(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        token = self.settings.telegram_bot_token
        if not token:
            return {}
        url = f"https://api.telegram.org/bot{token}/{method}"
        async with httpx.AsyncClient(timeout=20.0) as http:
            resp = await http.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        if not data.get("ok"):
            logger.warning("Telegram %s returned non-ok payload: %s", method, data)
        return data
