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
from lark_oapi.api.im.v1 import GetMessageResourceRequest, GetMessageResourceResponse
from .helper import bytes_to_image_url
import random
from pathlib import Path
import aiofiles

_EMOJI_TYPES = """
OK THUMBSUP THANKS MUSCLE FINGERHEART APPLAUSE FISTBUMP JIAYI DONE LOVE
PROUD COMFORT CLAP PRAISE STRIVE HUG LGTM OnIt YouAreTheBest SALUTE
SHAKE HIGHFIVE ROSE HEART PARTY GIFT Yes CheckMark Hundred AWESOMEN
Trophy Fire FIREWORKS REDPACKET FORTUNE LUCK BeamingFace Delighted
GoGoGo ThanksFace SaluteFace HappyDragon
""".split()

FEISHU_CACHE_DIR = Path("feishu_cache").resolve()
FEISHU_CACHE_DIR.mkdir(parents=True, exist_ok=True)


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

    async def _parse_msg(self, inbound_msg: InboundMessage) -> Message:
        async def get_image_url_from_key(message_id: str, image_key: str) -> bytes:
            client = self.channel.client
            request: GetMessageResourceRequest = (
                GetMessageResourceRequest.builder().message_id(message_id).file_key(image_key).type("image").build()
            )
            response: GetMessageResourceResponse = client.im.v1.message_resource.get(request)
            if not response.success():
                lark.logger.error(
                    f"client.im.v1.message_resource.get failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}"
                )
                return
            return bytes_to_image_url(response.file.read())

        async def _save_file(message_id: str, file_key: str, save_filepath: Path) -> None:
            client = self.channel.client
            request: GetMessageResourceRequest = (
                GetMessageResourceRequest.builder().message_id(message_id).file_key(file_key).type("file").build()
            )
            response: GetMessageResourceResponse = client.im.v1.message_resource.get(request)
            if not response.success():
                lark.logger.error(
                    f"client.im.v1.message_resource.get failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}"
                )
                return
            save_filepath.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(save_filepath, "wb") as f:
                await f.write(response.file.read())
            return save_filepath

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
                    content=[
                        ImageMessage(
                            image_url=await get_image_url_from_key(
                                inbound_msg.message_id, inbound_msg.content.image_key
                            )
                        )
                    ],
                    timestamp=int(inbound_msg.create_time),
                    session_metadata={"sender_id": inbound_msg.sender_id, "sender_name": inbound_msg.sender_name},
                )
            elif isinstance(inbound_msg.content, FileContent):
                save_filepath = (
                    FEISHU_CACHE_DIR
                    / f"{inbound_msg.sender_id}_{inbound_msg.sender_name}"
                    / inbound_msg.content.file_name
                )
                return Message(
                    message_id=inbound_msg.message_id,
                    session_id=f"{inbound_msg.chat_type}_{inbound_msg.chat_id}",
                    content=[
                        FileMessage(
                            file=await _save_file(inbound_msg.message_id, inbound_msg.content.file_key, save_filepath)
                        )
                    ],
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
