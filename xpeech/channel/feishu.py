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
from .schema import Message, TextMessage, ImageMessage, FileMessage
import json
import lark_oapi as lark
from lark_oapi.api.im.v1 import GetImageRequest, GetImageResponse
from .helper import bytes_to_image_url
import random

_EMOJI_TYPES = """
OK THUMBSUP THANKS MUSCLE FINGERHEART APPLAUSE FISTBUMP JIAYI DONE LOVE
PROUD COMFORT CLAP PRAISE STRIVE HUG LGTM OnIt YouAreTheBest SALUTE
SHAKE HIGHFIVE ROSE HEART PARTY GIFT Yes CheckMark Hundred AWESOMEN
Trophy Fire FIREWORKS REDPACKET FORTUNE LUCK BeamingFace Delighted
GoGoGo ThanksFace SaluteFace HappyDragon
""".split()


class FeishuBridge:
    """Bridge normalized Feishu messages into the Xpeech /chat SSE endpoint."""

    LOG_LEVEL = LogLevel.WARNING

    def __init__(self, chat_url: str, app_id: str, app_secret: str):
        self.chat_url = chat_url
        self.app_id = app_id
        self.app_secret = app_secret
        self.receive_queues: dict[str, asyncio.Queue[InboundMessage]] = {}
        # 初始化通道
        self.channel = FeishuChannel(
            log_level=self.LOG_LEVEL,
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

    async def add_reaction(self, msg: Message):
        await self.channel.add_reaction(msg.message_id, random.choice(_EMOJI_TYPES))

    async def _on_message(self, inbound_message: InboundMessage) -> None:
        msg = await self._parse_msg(inbound_message)
        await self.add_reaction(msg)
        print(msg)
        # self.receive_queues[msg.chat_id].put_nowait(msg)
        # await self.channel.send(
        #     msg.chat_id,
        #     {"markdown": f"received: {msg.content_text}"},
        #     {"reply_to": msg.message_id},
        # )

    async def get_image_url_from_key(self, image_key: str) -> bytes:
        client = lark.Client.builder().app_id(self.app_id).app_secret(self.app_secret).log_level(self.LOG_LEVEL).build()
        request: GetImageRequest = GetImageRequest.builder().image_key(image_key).build()
        response: GetImageResponse = client.im.v1.image.get(request)
        if not response.success():
            lark.logger.error(
                f"client.im.v1.image.get failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}"
            )
            return
        return bytes_to_image_url(response.file.read())

    async def _parse_msg(self, inbound_msg: InboundMessage) -> Message:
        if inbound_msg.chat_type == "p2p":
            if isinstance(inbound_msg.content, TextContent):
                return Message(
                    message_id=inbound_msg.message_id,
                    session_id=f"{inbound_msg.chat_type}_{inbound_msg.chat_id}",
                    content=[TextMessage(text=inbound_msg.content.text)],
                    timestamp=int(inbound_msg.create_time),
                    session_metadata={"sender_id": inbound_msg.sender_id, "sender_name": inbound_msg.sender_name},
                )
            elif isinstance(inbound_msg.content, ImageContent):
                return Message(
                    message_id=inbound_msg.message_id,
                    session_id=f"{inbound_msg.chat_type}_{inbound_msg.chat_id}",
                    content=[ImageMessage(image_url=await self.get_image_url_from_key(inbound_msg.content.image_key))],
                    timestamp=int(inbound_msg.create_time),
                    session_metadata={"sender_id": inbound_msg.sender_id, "sender_name": inbound_msg.sender_name},
                )
            elif isinstance(inbound_msg.content, FileContent):
                return Message(
                    message_id=inbound_msg.message_id,
                    session_id=f"{inbound_msg.chat_type}_{inbound_msg.chat_id}",
                    content=[FileMessage(file_path=inbound_msg.content.file_path)],
                    timestamp=int(inbound_msg.create_time),
                    session_metadata={"sender_id": inbound_msg.sender_id, "sender_name": inbound_msg.sender_name},
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
