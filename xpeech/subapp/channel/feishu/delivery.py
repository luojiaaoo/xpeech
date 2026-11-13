from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from html import escape
from typing import Any

from lark_channel import SendResult
from loguru import logger

from ..schema import ChatEvent, ChatEventType
from .config import FEISHU_STREAM_UPDATE_INTERVAL_SECONDS
from .formatting import OUTPUT_EVENT_TYPES, ensure_markdown


@dataclass
class ToolCardProgress:
    title: str
    card_id: str | None = None
    sequence: int = 0
    header: dict[str, Any] = field(default_factory=dict)
    success: bool | None = None


@dataclass
class ToolProgress:
    cards: dict[str, ToolCardProgress] = field(default_factory=dict)


class FeishuDeliveryMixin:
    """发送普通卡片、流式卡片和可更新的飞书卡片。"""

    async def send_tool_progress(self, to: str, event: ChatEvent, progress: ToolProgress) -> None:
        """每个工具调用使用独立卡片，按调用 ID 更新结果和标题。"""
        if ChatEventType.TOOL_CALL not in OUTPUT_EVENT_TYPES:
            return
        payload = json.loads(event.context)
        if event.event == ChatEventType.TOOL_CALL:
            for entry in payload:
                call_id = entry["tool_call_id"]
                if call_id not in progress.cards:
                    await self._create_tool_progress(to, entry, progress)
            return

        call_id = payload["tool_call_id"]
        state = progress.cards.get(call_id)
        if state is None:
            return

        success = payload["success"]
        text = self._format_tool_result(payload["result"])
        message, _ = self._format_chat_event(
            ChatEvent(event=ChatEventType.TOOL_CALL, context=text), title=state.title,
        )
        panel = message["card"]["body"]["elements"][0]
        await asyncio.sleep(FEISHU_STREAM_UPDATE_INTERVAL_SECONDS)
        state.sequence += 1
        await self.channel.update_card_element_content(
            state.card_id, "main", panel["elements"][0]["content"], sequence=state.sequence,
        )
        if success != state.success:
            state.sequence += 1
            await self.channel.patch_card_element(
                state.card_id, "panel", {"header": self._tool_progress_header(state.header, success)},
                sequence=state.sequence,
            )
            state.success = success

    async def _create_tool_progress(
        self, to: str, call: dict[str, Any], progress: ToolProgress,
    ) -> None:
        state = ToolCardProgress(title=self._format_tool_title(call))
        progress.cards[call["tool_call_id"]] = state
        message, _ = self._format_chat_event(
            ChatEvent(event=ChatEventType.TOOL_CALL, context="执行中…"), title=state.title,
        )
        card = message["card"]
        panel = card["body"]["elements"][0]
        state.header = panel["header"]
        state.card_id = await self.channel.create_card_instance(card)
        send_result = await self.channel.send_card_by_reference(to, state.card_id)
        self._ensure_send_success(send_result, operation="send tool card")

    @staticmethod
    def _tool_progress_header(header: dict[str, Any], success: bool) -> dict[str, Any]:
        color = "green" if success else "red"
        result = dict(header)
        for name in ("title", "expanded_title"):
            if name in header:
                title = escape(header[name]["content"], quote=False)
                result[name] = {"tag": "markdown", "content": f'<font color="{color}">{title}</font>'}
        return result

    async def finish_tool_progress(self, progress: ToolProgress) -> None:
        """结束当前轮每张工具卡片的流式更新。"""
        for state in progress.cards.values():
            if state.card_id is not None:
                state.sequence += 1
                try:
                    await self.channel.finish_streaming_card(state.card_id, sequence=state.sequence)
                except Exception:
                    logger.exception("Failed to finish Feishu tool card card_id={}", state.card_id)

    @staticmethod
    def _ensure_send_success(
        result: SendResult,
        *,
        operation: str,
        require_message_id: bool = False,
    ) -> SendResult:
        if not result.success:
            raise RuntimeError(f"Feishu {operation} failed: {result.error}")
        if require_message_id and not result.message_id:
            raise RuntimeError(f"Feishu {operation} succeeded without a message_id")
        return result

    async def send(
        self,
        to: str,
        message: dict | AsyncIterator[str],
        iter_content: AsyncIterator[str] | None,
        opts: dict | None = None,
    ) -> SendResult:
        if not iter_content:
            return await self.channel.send(to=to, message=message, opts=opts)

        card_id = await self.channel.create_card_instance(message["card"])
        send_result = await self.channel.send_card_by_reference(
            to,
            card_id,
            **(opts or {}),
        )
        if not send_result.success:
            raise RuntimeError(send_result.error)

        seq = 0
        accumulated = ""
        last_sent = ""
        loop = asyncio.get_running_loop()
        update_interval = FEISHU_STREAM_UPDATE_INTERVAL_SECONDS
        next_update_at = loop.time() + update_interval

        async for token in iter_content:
            accumulated += token
            if loop.time() < next_update_at:
                continue
            seq += 1
            await self.channel.update_card_element_content(
                card_id,
                "main",
                ensure_markdown(accumulated),
                sequence=seq,
            )
            last_sent = accumulated
            next_update_at = loop.time() + update_interval

        if accumulated != last_sent:
            await asyncio.sleep(max(0, next_update_at - loop.time()))
            seq += 1
            await self.channel.update_card_element_content(
                card_id,
                "main",
                ensure_markdown(accumulated),
                sequence=seq,
            )

        seq += 1
        await self.channel.finish_streaming_card(card_id, sequence=seq)
        return send_result

    async def channel_send(
        self,
        to: str,
        message: dict | AsyncIterator[str],
        opts: dict | None,
        user_id: str,
        sender_name: str,
        message_type: ChatEventType,
        iter_content: AsyncIterator[str] | None,
    ) -> None:
        persistence_types = (
            ChatEventType.THINKING,
            ChatEventType.ASSISTANT,
            ChatEventType.ERROR,
            ChatEventType.COMMAND,
            ChatEventType.TOKEN_USAGE,
            ChatEventType.QUESTION,
        )
        logger.info(
            "Feishu message sending user_id={} sender_name={} message_type={}",
            user_id,
            sender_name,
            message_type,
        )
        if message_type in persistence_types:
            result = await self.send(
                to=to,
                message=message,
                iter_content=iter_content,
                opts=opts,
            )
            self._ensure_send_success(result, operation="send message")
            self.session_update_message_id.pop(user_id, None)
            return

        message_id = self.session_update_message_id.get(user_id)
        if message_id is None:
            send_result = await self.send(
                to=to,
                message=message,
                iter_content=iter_content,
                opts=opts,
            )
            self._ensure_send_success(
                send_result,
                operation="send progress card",
                require_message_id=True,
            )
            self.session_update_message_id[user_id] = send_result.message_id
            return

        if isinstance(message, AsyncIterator):
            raise TypeError("Cannot update progress card with streaming message")
        await asyncio.sleep(0.25)
        result = await self.channel.update_card(message_id, message["card"])
        self._ensure_send_success(result, operation="update progress card")
