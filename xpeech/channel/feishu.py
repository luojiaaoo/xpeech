from __future__ import annotations

from lark_oapi.core import LogLevel
from lark_oapi.channel.config import PolicyConfig
from lark_oapi.channel import DedupConfig, FeishuChannel, OutboundConfig, RetryConfig, SafetyConfig
import asyncio
from typing import Any


class FeishuBridge:
    """Bridge normalized Feishu messages into the Xpeech /chat SSE endpoint."""

    def __init__(self, chat_url: str, app_id: str, app_secret: str):
        self.chat_url = chat_url
        self.app_id = app_id
        self.app_secret = app_secret
        self.queues: dict[str, asyncio.Queue[Any]] = {}
        # 初始化通道
        self.channel = FeishuChannel(
            log_level=LogLevel.DEBUG,
            app_id=self.app_id,
            app_secret=self.app_secret,
            policy=PolicyConfig(
                dm_policy="open",
                group_policy="disabled",
            ),
            safety=SafetyConfig(dedup=DedupConfig(ttl_seconds=43_200)),
            outbound=OutboundConfig(retry=RetryConfig(max_attempts=5)),
        )
        self.channel.on("message", self._on_message)

    async def _on_message(self, msg):
        await self.channel.send(
            msg.chat_id,
            {"markdown": f"received: {msg.content_text}"},
            {"reply_to": msg.message_id},
        )

    def start(self) -> None:
        asyncio.run(self.channel.connect())


def run(chat_url, app_id, app_secret) -> None:
    asyncio.run(
        FeishuBridge(
            chat_url=chat_url,
            app_id=app_id,
            app_secret=app_secret,
        ).start()
    )
