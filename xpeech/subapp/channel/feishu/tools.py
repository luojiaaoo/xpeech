from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from html import escape
from typing import Any

from loguru import logger

from ..schema import ChatEvent, ChatEventType
from .channel import XpeechFeishuChannel
from .config import FEISHU_STREAM_UPDATE_INTERVAL_SECONDS
from .delivery import ensure_send_success
from .formatting import OUTPUT_EVENT_TYPES, FeishuEventFormatter


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


class FeishuTools:
    """按工具调用 ID 管理进度、结果和流式卡片序号。"""

    def __init__(self, channel: XpeechFeishuChannel, formatter: FeishuEventFormatter):
        self.channel = channel
        self.formatter = formatter

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
        text = self.formatter._format_tool_result(payload["result"])
        message, _ = self.formatter._format_chat_event(
            ChatEvent(event=ChatEventType.TOOL_CALL, context=text),
            title=state.title,
        )
        panel = message["card"]["body"]["elements"][0]
        await asyncio.sleep(FEISHU_STREAM_UPDATE_INTERVAL_SECONDS)
        state.sequence += 1
        await self.channel.update_card_element_content(
            state.card_id,
            "main",
            panel["elements"][0]["content"],
            sequence=state.sequence,
        )
        if success != state.success:
            state.sequence += 1
            await self.channel.patch_card_element(
                state.card_id,
                "panel",
                {"header": self._tool_progress_header(state.header, success)},
                sequence=state.sequence,
            )
            state.success = success

    async def _create_tool_progress(
        self,
        to: str,
        call: dict[str, Any],
        progress: ToolProgress,
    ) -> None:
        state = ToolCardProgress(title=self.formatter._format_tool_title(call))
        progress.cards[call["tool_call_id"]] = state
        message, _ = self.formatter._format_chat_event(
            ChatEvent(event=ChatEventType.TOOL_CALL, context="执行中…"),
            title=state.title,
        )
        card = message["card"]
        panel = card["body"]["elements"][0]
        state.header = panel["header"]
        state.card_id = await self.channel.create_card_instance(card)
        send_result = await self.channel.send_card_by_reference(to, state.card_id)
        ensure_send_success(send_result, operation="send tool card")

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
                except Exception:  # noqa: BLE001 - 继续关闭其余工具卡片。
                    logger.exception("Failed to finish Feishu tool card card_id={}", state.card_id)
