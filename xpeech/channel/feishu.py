from __future__ import annotations

from lark_oapi.core import LogLevel
from lark_oapi.channel.config import PolicyConfig
from lark_oapi.channel import (
    DedupConfig,
    FeishuChannel,
    OutboundConfig,
    RetryConfig,
    SafetyConfig,
    InboundMessage,
    TextContent,
    ImageContent,
    FileContent,
)
import asyncio
from .schema import Messages, TextMessage, ImageMessage, FileMessage
import json
import lark_oapi as lark
from lark_oapi.api.im.v1 import GetImageRequest, GetImageResponse


class FeishuBridge:
    """Bridge normalized Feishu messages into the Xpeech /chat SSE endpoint."""

    def __init__(self, chat_url: str, app_id: str, app_secret: str):
        self.chat_url = chat_url
        self.app_id = app_id
        self.app_secret = app_secret
        self.receive_queues: dict[str, asyncio.Queue[InboundMessage]] = {}
        # 初始化通道
        self.channel = FeishuChannel(
            log_level=LogLevel.DEBUG,
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
        self.channel.on("message", self._on_message)
        

    async def _on_message(self, msg: InboundMessage):
        print(msg)
        # self.receive_queues[msg.chat_id].put_nowait(msg)
        # await self.channel.send(
        #     msg.chat_id,
        #     {"markdown": f"received: {msg.content_text}"},
        #     {"reply_to": msg.message_id},
        # )

    async def get_image_from_key(self, image_key: str) -> bytes:
        client = self.channel.client
        request: GetImageRequest = (
            GetImageRequest.builder().image_key(image_key).build()
        )
        response: GetImageResponse = client.im.v1.image.get(request)
        if not response.success():
            lark.logger.error(
                f"client.im.v1.image.get failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}"
            )
            return
        return response.file.read()

    def _parse_msg(self, msg: InboundMessage) -> str:
        if msg.chat_type == "p2p":
            if isinstance(msg.content, TextContent):
                Messages(
                    message_id=msg.message_id,
                    session_id=f"{msg.chat_type}_{msg.chat_id}",
                    content=[TextMessage(text=msg.content.text)],
                    timestamp=int(msg.create_time),
                    session_metadata={"sender_id": msg.sender_id, "sender_name": msg.sender_name},
                )
            elif isinstance(msg.content, ImageContent):
                Messages(
                    message_id=msg.message_id,
                    session_id=f"{msg.chat_type}_{msg.chat_id}",
                    content=[ImageMessage(image_url=msg.content.image_key)],
                    timestamp=int(msg.create_time),
                    session_metadata={"sender_id": msg.sender_id, "sender_name": msg.sender_name},
                )
            elif isinstance(msg.content, FileContent):
                Messages(
                    message_id=msg.message_id,
                    session_id=f"{msg.chat_type}_{msg.chat_id}",
                    content=[FileMessage(file_path=msg.content.file_path)],
                    timestamp=int(msg.create_time),
                    session_metadata={"sender_id": msg.sender_id, "sender_name": msg.sender_name},
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
