from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from lark_channel import SendResult
from loguru import logger

from ..schema import ChatEvent, ChatEventType
from .cards import build_feishu_tool_details
from .config import FEISHU_STREAM_UPDATE_INTERVAL_SECONDS
from .formatting import OUTPUT_EVENT_TYPES, color_markdown, ensure_markdown


@dataclass
class ToolProgress:
    card_id: str | None = None
    sequence: int = 0
    title: str = ""
    text: str = ""
    calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    results: dict[str, str] = field(default_factory=dict)
    failures: set[str] = field(default_factory=set)


class FeishuDeliveryMixin:
    """发送普通卡片、流式卡片和可更新的飞书卡片。"""

    async def send_tool_progress(self, to: str, event: ChatEvent, progress: ToolProgress) -> None:
        """将当前轮的工具调用和结果更新到同一张折叠卡片。"""
        output = OUTPUT_EVENT_TYPES.get(ChatEventType.TOOL_CALL)
        if output is None:
            return
        text = self._format_tool_event(event)
        if event.event == ChatEventType.TOOL_CALL:
            progress.title = self._truncate(
                " | ".join(filter(None, (progress.title, text))),
                output.get("max_title_chars"),
            )
            try:
                for entry in json.loads(event.context):
                    tool_call_id = entry["tool_call_id"]
                    progress.calls[tool_call_id] = entry
                    progress.results.setdefault(tool_call_id, "")
                    arguments = entry.get("arguments")
                    if isinstance(arguments, dict) and arguments.get("execution_description"):
                        pending = {
                            "name": entry["name"], "result": "执行中…",
                        }
                        progress.results[tool_call_id] = self._format_tool_event(
                            ChatEvent(event=ChatEventType.TOOL_CALL_RESULT, context=json.dumps(pending)), tool_call=entry,
                        )
                progress.text = self._truncate(
                    "\n\n".join(filter(None, progress.results.values())), output.get("max_chars"),
                )
            except (ValueError, TypeError, KeyError):
                pass
        else:
            try:
                payload = json.loads(event.context)
                tool_call_id = payload["tool_call_id"]
            except (ValueError, TypeError, KeyError):
                tool_call_id = None
                payload = {}
            if isinstance(tool_call_id, str):
                text = self._format_tool_event(event, tool_call=progress.calls.get(tool_call_id))
                progress.results[tool_call_id] = text
                if payload.get("success") is False:
                    progress.failures.add(tool_call_id)
                else:
                    progress.failures.discard(tool_call_id)
                progress.text = self._truncate(
                    "\n\n".join(filter(None, progress.results.values())), output.get("max_chars"),
                )
            else:
                progress.text = self._truncate(
                    "\n\n".join(filter(None, (progress.text, text))), output.get("max_chars"),
                )
        message, _ = self._format_chat_event(
            ChatEvent(event=ChatEventType.TOOL_CALL, context=progress.text),
            title=progress.title,
        )
        card = message["card"]
        if progress.failures:
            element = card["body"]["elements"][0]
            body = element["elements"][0] if element["tag"] == "collapsible_panel" else element
            content = body["content"]
            parts = []
            # 在统一转义、截断后按 ID 标色，避免截断标签或把工具输出当作标签解析。
            for call_id, result_text in progress.results.items():
                if not result_text or not content:
                    continue
                if parts:
                    parts.append(content[:2])  # 结果之间的换行。
                    content = content[2:]
                length = len(ensure_markdown(result_text))
                result_content, content = content[:length], content[length:]
                if call_id in progress.failures and result_content:
                    result_content = color_markdown(result_content, "red")
                parts.append(result_content)
            body["content"] = "".join(parts) + content
        calls_with_copy = [
            call for call in progress.calls.values()
            if isinstance(call.get("arguments"), dict)
            and (call["arguments"].get("execution_title") or call["arguments"].get("execution_description"))
        ]
        if calls_with_copy:
            card["body"]["elements"].append(build_feishu_tool_details(calls_with_copy))
        if progress.card_id is None:
            progress.card_id = await self.channel.create_card_instance(card)
            result = await self.channel.send_card_by_reference(to, progress.card_id)
            self._ensure_send_success(result, operation="send tool card")
        else:
            await asyncio.sleep(FEISHU_STREAM_UPDATE_INTERVAL_SECONDS)
            element = card["body"]["elements"][0]
            if event.event == ChatEventType.TOOL_CALL and "header" in element:
                progress.sequence += 1
                await self.channel.patch_card_element(
                    progress.card_id,
                    "panel",
                    {"header": element["header"]},
                    sequence=progress.sequence,
                )
            body = element["elements"][0] if element["tag"] == "collapsible_panel" else element
            progress.sequence += 1
            await self.channel.update_card_element_content(
                progress.card_id,
                "main",
                body["content"],
                sequence=progress.sequence,
            )

    async def finish_tool_progress(self, progress: ToolProgress) -> None:
        """结束当前轮工具调用的卡片流式更新。"""
        if progress.card_id is not None:
            progress.sequence += 1
            await self.channel.finish_streaming_card(progress.card_id, sequence=progress.sequence)

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
