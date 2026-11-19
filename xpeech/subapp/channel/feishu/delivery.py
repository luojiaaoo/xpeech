from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from lark_channel import SendResult
from loguru import logger

from ..schema import ChatEventType
from .channel import XpeechFeishuChannel
from .config import FEISHU_STREAM_UPDATE_INTERVAL_SECONDS
from .formatting import ensure_markdown


def ensure_send_success(
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


class FeishuDelivery:
    """直接发送独立消息和流式卡片。"""

    def __init__(self, channel: XpeechFeishuChannel):
        self.channel = channel

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
        logger.info(
            "Feishu message sending user_id={} sender_name={} message_type={}",
            user_id,
            sender_name,
            message_type,
        )
        result = await self.send(
            to=to,
            message=message,
            iter_content=iter_content,
            opts=opts,
        )
        ensure_send_success(result, operation="send message")
