from __future__ import annotations

import asyncio

from lark_channel import DedupConfig, Events, InboundMessage, OutboundConfig, PolicyConfig, RetryConfig, SafetyConfig
from loguru import logger

from ....config.settings import settings
from ..schema import Message
from .background import FeishuBackgroundMessages
from .channel import XpeechFeishuChannel
from .conversation import FeishuConversation
from .delivery import FeishuDelivery
from .formatting import FeishuEventFormatter
from .inbound import FeishuInbound, UnsupportedFeishuMessageError
from .questions import FeishuQuestions
from .tools import FeishuTools


class FeishuBridge:
    """组装飞书渠道组件，管理 SDK 消息入口和连接生命周期。"""

    def __init__(
        self,
        chat_url: str,
        app_id: str,
        app_secret: str,
        *,
        channel: XpeechFeishuChannel | None = None,
    ):
        self.chat_url = chat_url
        self.app_id = app_id
        self.app_secret = app_secret
        self.channel = (
            channel
            if channel is not None
            else XpeechFeishuChannel(
                app_id=self.app_id,
                app_secret=self.app_secret,
                policy=PolicyConfig(
                    dm_policy="open",
                    group_policy="open",
                    require_mention=True,
                ),
                safety=SafetyConfig(dedup=DedupConfig(ttl_seconds=43_200)),
                outbound=OutboundConfig(retry=RetryConfig(max_attempts=5)),
            )
        )
        self.inbound = FeishuInbound(self.channel)
        self.formatter = FeishuEventFormatter()
        self.delivery = FeishuDelivery(self.channel)
        self.tools = FeishuTools(self.channel, self.formatter)
        self.questions = FeishuQuestions(chat_url, self.channel, self.inbound)
        self.conversation = FeishuConversation(
            chat_url,
            self.channel,
            self.delivery,
            self.formatter,
            self.questions,
            self.tools,
        )
        self.background = FeishuBackgroundMessages(chat_url, self.channel)
        self.channel.on(Events.MESSAGE, self._on_message)
        self.channel.on(Events.CARD_ACTION, self.questions.handle_action)

    async def _on_message(self, inbound_message: InboundMessage) -> None:
        scope = inbound_message.chat_id or inbound_message.message_id
        self.channel.block_batch_scope(scope)
        messages: list[Message] = []
        try:
            for source in inbound_message.batched_sources or [inbound_message]:
                try:
                    messages.append(await self.inbound.parse_message(source))
                except UnsupportedFeishuMessageError as exc:
                    logger.warning(str(exc))
            if messages:
                await self.consume(messages)
        finally:
            for message in messages:
                self.delivery.session_update_message_id.pop(message.user_id, None)
            self.channel.unblock_batch_scope(scope)

    async def consume(self, messages: list[Message]) -> None:
        await self.conversation.consume(messages)

    async def start(self) -> None:
        try:
            async with asyncio.TaskGroup() as task_group:
                background_message_task = task_group.create_task(self.background.poll_background_messages())
                try:
                    await self.channel.connect()
                finally:
                    background_message_task.cancel()
        finally:
            await self.channel.disconnect()


def run(chat_url: str) -> None:
    try:
        asyncio.run(
            FeishuBridge(
                chat_url=chat_url,
                app_id=settings.feishu.app_id,
                app_secret=settings.feishu.app_secret,
            ).start()
        )
    except KeyboardInterrupt:
        print("Stopped by user")
