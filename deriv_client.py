"""
Deriv WebSocket client: connect, subscribe to ticks, optional authorize, reconnect.

Official endpoint pattern:
  wss://ws.binaryws.com/websockets/v3?app_id=APP_ID

Tick subscription pattern:
  {"ticks": "BOOM500", "subscribe": 1, "req_id": 2}

Ping keep-alive:
  {"ping": 1}

This module also contains stub helpers for proposal / buy / sell. Those calls are
hard-blocked unless `execution_allowed()` in config.py returns True.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

try:
    from websockets.asyncio.client import connect as ws_connect
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "The installed 'websockets' package is incomplete. "
        "Fix: python -m pip uninstall -y websockets && "
        "python -m pip install --no-cache-dir 'websockets>=15.0.1,<16'"
    ) from exc

from config import Settings, execution_allowed, get_settings

logger = logging.getLogger(__name__)

MessageHandler = Callable[[dict[str, Any]], Awaitable[None]]


def build_ws_uri(app_id: int) -> str:
    return f"wss://ws.binaryws.com/websockets/v3?app_id={app_id}"


class DerivWebSocketClient:
    """Thin async websocket wrapper with resilient reconnect."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._running = False
        self._ws: Any | None = None
        self._pinger_task: asyncio.Task[None] | None = None
        self._req_id = 1

    def _next_req_id(self) -> int:
        self._req_id += 1
        return self._req_id

    async def _send_json(self, ws: Any, payload: dict[str, Any]) -> None:
        await ws.send(json.dumps(payload))

    async def _maybe_authorize(self, ws: Any) -> None:
        token = self.settings.deriv_api_token

        if not token:
            logger.info("No DERIV_API_TOKEN set — skipping authorize (ticks may still stream).")
            return

        req_id = self._next_req_id()
        await self._send_json(ws, {"authorize": token, "req_id": req_id})

        for _ in range(5):
            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            data = json.loads(raw)

            if data.get("msg_type") == "authorize":
                if data.get("error"):
                    logger.error("Authorize error from Deriv: %s", data)
                else:
                    logger.info("Deriv authorize successful.")
                return

            if data.get("error"):
                logger.warning("Unexpected pre-auth message: %s", data)
                return

    async def _subscribe_ticks(self, ws: Any, symbols: list[str]) -> None:
        for sym in symbols:
            req_id = self._next_req_id()
            await self._send_json(
                ws,
                {"ticks": sym, "subscribe": 1, "req_id": req_id},
            )
            logger.info("Requested tick subscription for %s (req_id=%s)", sym, req_id)

    async def _pinger_loop(self, ws: Any) -> None:
        try:
            while self._running:
                await asyncio.sleep(45)
                await self._send_json(ws, {"ping": 1})
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Ping loop failed")

    async def stream_ticks(
        self,
        symbols: list[str],
        on_message: MessageHandler,
    ) -> None:
        """
        Long-running loop: connects, subscribes, dispatches parsed JSON dicts.

        The callback receives all websocket messages except ping replies.
        """
        self._running = True
        backoff = 1.0
        uri = build_ws_uri(self.settings.deriv_app_id)

        while self._running:
            try:
                async with ws_connect(
                    uri,
                    ping_interval=None,
                    max_queue=512,
                ) as ws:
                    self._ws = ws
                    logger.info("Connected to Deriv WebSocket")
                    backoff = 1.0

                    await self._maybe_authorize(ws)
                    await self._subscribe_ticks(ws, symbols)

                    if self._pinger_task:
                        self._pinger_task.cancel()

                    self._pinger_task = asyncio.create_task(self._pinger_loop(ws))

                    async for raw in ws:
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            logger.warning("Non-JSON frame: %s", raw[:200])
                            continue

                        msg_type = data.get("msg_type")

                        if msg_type == "ping":
                            continue

                        await on_message(data)

            except asyncio.CancelledError:
                break

            except Exception:
                logger.exception(
                    "WebSocket connection error — reconnecting in %.1fs",
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

            finally:
                if self._pinger_task:
                    self._pinger_task.cancel()
                    self._pinger_task = None

                self._ws = None

        logger.info("Deriv tick stream stopped")

    def stop(self) -> None:
        self._running = False


class DerivExecutionGuard:
    """
    Placeholder for proposal / purchase endpoints.

    This repository ships signal-only by default.
    These methods raise unless every guard in .env is satisfied.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def ensure_execution_allowed(self) -> None:
        ok, reason = execution_allowed(self.settings)

        if not ok:
            raise PermissionError(
                f"Blocked automated execution: {reason} "
                "Trading synthetic indices carries substantial risk."
            )

    async def proposal(self, ws: Any | None, payload: dict[str, Any]) -> dict[str, Any]:
        self.ensure_execution_allowed()

        if ws is None:
            raise RuntimeError("Execution requires an active authenticated websocket session.")

        await ws.send(json.dumps(payload))
        raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
        return json.loads(raw)

    async def buy(self, ws: Any | None, payload: dict[str, Any]) -> dict[str, Any]:
        self.ensure_execution_allowed()

        if ws is None:
            raise RuntimeError("Execution requires an active authenticated websocket session.")

        await ws.send(json.dumps(payload))
        raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
        return json.loads(raw)


async def fetch_active_symbols_brief(settings: Settings | None = None) -> list[str]:
    """
    Pull the brief symbol list via websocket.

    Helpful for validating SYMBOLS=.env entries.
    """
    settings = settings or get_settings()
    uri = build_ws_uri(settings.deriv_app_id)
    symbols: list[str] = []

    async with ws_connect(uri, ping_interval=None) as ws:
        req_id = 1
        await ws.send(json.dumps({"active_symbols": "brief", "req_id": req_id}))

        raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
        data = json.loads(raw)

        if data.get("error"):
            logger.warning("active_symbols error: %s", data["error"])
            return symbols

        lst = data.get("active_symbols") or []

        for row in lst:
            sym = row.get("symbol")
            if sym:
                symbols.append(sym)

    return symbols