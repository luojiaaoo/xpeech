from __future__ import annotations

from lark_channel import (
    ChatQueueConfig,
    DedupConfig,
    Events,
    InboundMessage,
    OutboundConfig,
    PolicyConfig,
    RetryConfig,
    SafetyConfig,
    TextContent,
)
from loguru import logger

from ....utils.helper import stack_ensure_aclosing
from ..helper import stop_chat
from ..schema import ChatEvent, ChatEventType, Message
from .channel import XpeechFeishuChannel
from .config import DEFAULT_SESSION_ID
from .conversation import FeishuConversation
from .delivery import FeishuDelivery, ensure_send_success
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
                safety=SafetyConfig(
                    dedup=DedupConfig(ttl_seconds=43_200),
                    # The backend inbox accepts concurrent submissions. Keep SDK
                    # admission checks, but do not batch or serialize whole chats.
                    chat_queue=ChatQueueConfig(enabled=False),
                ),
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
        self.channel.on(Events.MESSAGE, self._on_message)
        self.channel.on(Events.CARD_ACTION, self.questions.handle_action)

    async def _on_message(self, inbound_message: InboundMessage) -> None:
        async with stack_ensure_aclosing() as resources:
            messages: list[Message] = []
            for source in inbound_message.batched_sources or [inbound_message]:
                if (
                    source.chat_type == "p2p"
                    and isinstance(source.content, TextContent)
                    and source.content.text.strip() == "/stop"
                ):
                    await self._stop(source)
                    continue
                try:
                    message = await self.inbound.parse_message(source, resources)
                except UnsupportedFeishuMessageError as exc:
                    logger.warning(str(exc))
                    continue
                if message.content:
                    messages.append(message)
            if messages:
                await self.consume(messages)

    async def _stop(self, source: InboundMessage) -> None:
        """A channel control command: never turn it into an agent message."""
        try:
            user_id, _ = await self.inbound.get_user_identity(source.sender_id)
            response = await stop_chat(user_id, self.chat_url, "feishu", session_id=DEFAULT_SESSION_ID)
            response.raise_for_status()
            stopped = response.json().get("stopped")
            if not isinstance(stopped, bool):
                raise ValueError("Backend did not acknowledge the stop request")
            text = "已停止当前会话。" if stopped else "当前没有正在运行的会话。"
        except Exception:  # noqa: BLE001 - 控制指令失败需反馈给发送者，不能误报成功。
            logger.exception("Failed to stop Feishu chat message_id={}", source.message_id)
            text = "停止失败，请稍后重试。"
        message, _ = self.formatter._format_chat_event(
            ChatEvent(event=ChatEventType.COMMAND, context=text),
        )
        result = await self.channel.send(
            to=source.chat_id,
            message=message,
            opts={"reply_to": source.message_id},
        )
        ensure_send_success(result, operation="send stop acknowledgement")

    async def consume(self, messages: list[Message]) -> None:
        await self.conversation.consume(messages)

    async def start(self) -> None:
        try:
            await self.channel.connect()
        finally:
            await self.channel.disconnect()
