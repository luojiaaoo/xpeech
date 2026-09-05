from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from lark_channel import SendResult
from loguru import logger

from ..schema import ChatEventType
from .config import FEISHU_STREAM_UPDATE_INTERVAL_SECONDS


class FeishuDeliveryMixin:
    """发送普通卡片、流式卡片和可更新的飞书卡片。"""

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
                accumulated,
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
                accumulated,
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
        session_id: str,
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
            "Feishu message sending session_id={} sender_name={} message_type={}",
            session_id,
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
            self.session_update_message_id.pop(session_id, None)
            return

        message_id = self.session_update_message_id.get(session_id)
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
            self.session_update_message_id[session_id] = send_result.message_id
            return

        if isinstance(message, AsyncIterator):
            raise TypeError("Cannot update progress card with streaming message")
        await asyncio.sleep(0.25)
        result = await self.channel.update_card(message_id, message["card"])
        self._ensure_send_success(result, operation="update progress card")
