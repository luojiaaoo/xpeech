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
    MediaContent,
    PostContent,
    SendResult,
)
import asyncio
from .schema import ChatEvent, ChatEventType, Message, TextData, ImageData, FileData
import json
import lark_oapi as lark
from lark_oapi.api.im.v1 import GetMessageResourceRequest, GetMessageResourceResponse
from .helper import bytes_to_image_url, iter_chat_events
import random
from pathlib import Path
import aiofiles
from itertools import chain
from datetime import datetime
from typing import Any, NotRequired, TypedDict
from loguru import logger
from ..config.settings import settings
import base64


class OutputEventType(TypedDict):
    content: str | None | type(Ellipsis)
    text_size: NotRequired[str | dict[str, Any]]
    text_align: NotRequired[str]
    icon: NotRequired[dict[str, Any] | None]


# 图标： https://open.feishu.cn/document/feishu-cards/enumerations-for-icons
OUTPUT_EVENT_TYPES: dict[ChatEventType, OutputEventType] = {
    ChatEventType.ASSISTANT: {
        "content": ...,
        "text_size": "normal",
        "text_align": "left",
        "icon": {"tag": "standard_icon", "token": "robot_filled", "color": "red"},
    },
    ChatEventType.COMMAND: {
        "content": ...,
        "text_size": "notation",
        "text_align": "center",
        "icon": {"tag": "standard_icon", "token": "command_outlined", "color": "turquoise"},
    },
    ChatEventType.THINKING: {
        "content": "我正在思考，稍等一下。",
        "text_size": "notation",
        "text_align": "center",
        "icon": {"tag": "standard_icon", "token": "tab-more_outlined", "color": "green"},
    },
    ChatEventType.TOOL_CALL: {
        "content": "我需要调用工具处理一下。",
        "text_size": "notation",
        "text_align": "center",
        "icon": {"tag": "standard_icon", "token": "select-up_outlined", "color": "wathet"},
    },
    ChatEventType.TOOL_CALL_RESULT: {
        "content": "工具处理完成，我继续整理结果。",
        "text_size": "notation",
        "text_align": "center",
        "icon": {"tag": "standard_icon", "token": "bitableform_outlined", "color": "yellow"},
    },
    ChatEventType.TOKEN_USAGE: {
        "content": ...,
        "text_size": "notation",
        "text_align": "left",
        "icon": None,
    },
}

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

    LOG_LEVEL = LogLevel.INFO

    def __init__(self, chat_url: str, app_id: str, app_secret: str, parallel: int):
        self.semaphore = asyncio.Semaphore(parallel)
        self.chat_url = chat_url
        self.app_id = app_id
        self.app_secret = app_secret
        self.receive_queues: dict[str, asyncio.Queue[Message]] = {}
        self.session_tasks: dict[str, asyncio.Task] = {}
        self.session_update_message_id: dict[str, str] = {}
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
        if msg.session_id not in self.receive_queues:
            self.receive_queues[msg.session_id] = asyncio.Queue()
        self.receive_queues[msg.session_id].put_nowait(msg)

    async def channel_send(
        self, to: str, message: dict, opts: dict | None, session_id: str, message_type: ChatEventType
    ):
        persistence_tyle = (
            ChatEventType.ASSISTANT,
            ChatEventType.COMMAND,
            ChatEventType.TOKEN_USAGE,
        )
        if message_type in persistence_tyle:
            await self.channel.send(to=to, message=message, opts=opts)
            self.session_update_message_id.pop(session_id, None)
        else:
            if (_message_id := self.session_update_message_id.get(session_id)) is None:
                send_result: SendResult = await self.channel.send(to=to, message=message, opts=opts)
                self.session_update_message_id[session_id] = send_result.message_id
            else:
                await asyncio.sleep(0.25)
                await self.channel.update_card(_message_id, message["card"])

    async def consume(self, session_id: str, idle_timeout: int | None = None) -> None:
        idle_timeout = settings.feishu.idle_timeout if idle_timeout is None else idle_timeout
        # 无数据或者未超时，跳过
        if session_id not in self.receive_queues:
            return

        # 有数据且已超时，取出数据
        queue = self.receive_queues[session_id]
        qsize = queue.qsize()
        if qsize == 0:
            return
        last_message: Message = queue._queue[qsize - 1]
        if datetime.now().timestamp() - last_message.timestamp < idle_timeout:
            return
        # 取出全部数据并请求chat接口
        messages = [queue.get_nowait() for _ in range(qsize)]
        chat_id = messages[-1].chat_id
        reply_to = messages[-1].message_id
        # 添加已读反应
        for i in messages:
            try:
                await self.add_reaction(i)
            except Exception:
                logger.debug("Failed to add Feishu reaction message_id={}", i.message_id)
        # 处理消息
        try:
            async for event in iter_chat_events(messages, self.chat_url):
                # 检测有没有文件发送请求
                if event.event == ChatEventType.TOOL_CALL_RESULT:
                    for _, tool_call_name, tool_call_result in json.loads(event.context):
                        if tool_call_name != "send_file":
                            continue
                        filepath = tool_call_result
                        async with aiofiles.open(filepath, "rb") as f:
                            source_bytes = await f.read()
                        await self.channel.send(
                            chat_id, {"file": {"source": source_bytes, "file_name": Path(filepath).name}}
                        )

                # 返回给用户消息
                card = self._format_chat_event(event)
                if card:
                    await self.channel_send(
                        to=chat_id,
                        message=card,
                        opts=(
                            {"reply_to": reply_to}
                            if event.event == ChatEventType.ASSISTANT and reply_to is not None
                            else None
                        ),
                        session_id=session_id,
                        message_type=event.event,
                    )
        except Exception:
            logger.exception("Failed to consume Feishu messages session_id={}", session_id)
            card = self._format_chat_event(
                ChatEvent(event=ChatEventType.ASSISTANT, context="这次处理消息时出错了，请稍后再试。")
            )
            if card:
                await self.channel.send(
                    chat_id,
                    card,
                    *([{"reply_to": reply_to}] if reply_to is not None else []),
                )

    def _format_chat_event(self, event: ChatEvent) -> dict[str, Any] | None:
        if event.event not in OUTPUT_EVENT_TYPES:
            return None

        output = OUTPUT_EVENT_TYPES[event.event]
        content = output["content"]
        if content is None:
            return None

        text = self._format_chat_event_content(event) if content is ... else content
        if not text:
            return None

        element = {
            "tag": "markdown",
            "margin": "0px 0px 0px 0px",
            "content": text,
            "text_size": output.get("text_size", "normal"),
            "text_align": output.get("text_align", "left"),
        }
        if output.get("icon") is not None:
            element["icon"] = output["icon"]

        return {
            "card": {
                "schema": "2.0",
                "body": {
                    "elements": [element],
                },
            }
        }

    def _format_chat_event_content(self, event: ChatEvent) -> str:
        if event.event == ChatEventType.ASSISTANT:
            return event.context
        elif event.event == ChatEventType.COMMAND:
            return f"**[命令]** {event.context}"
        elif event.event == ChatEventType.THINKING:
            return f"**[思考中]** {event.context}"
        elif event.event == ChatEventType.TOOL_CALL:
            return f"**[调用工具]** {self._format_tool_call_event(event)}"
        elif event.event == ChatEventType.TOOL_CALL_RESULT:
            return f"**[工具调用结果]** {self._format_json_context(event.context)}"
        elif event.event == ChatEventType.TOKEN_USAGE:
            return f"**[词元]** {event.context}"
        return f"[{event.event}]\n{event.context}"

    def _format_tool_call_event(self, event: ChatEvent) -> str:
        try:
            tool_calls = json.loads(event.context)
        except json.JSONDecodeError:
            return event.context

        if not isinstance(tool_calls, list):
            return self._format_json_value(tool_calls)

        lines: list[str] = []
        for index, tool_call in enumerate(tool_calls, start=1):
            if isinstance(tool_call, list) and len(tool_call) >= 3:
                lines.append(f"{index}. {tool_call[1]}\n```json\n{self._format_json_value(tool_call[2])}\n```")
            else:
                lines.append(f"{index}. {self._format_json_value(tool_call)}")
        return "\n\n".join(lines)

    def _format_json_context(self, context: str) -> str:
        try:
            value = json.loads(context)
        except json.JSONDecodeError:
            return context
        return f"```json\n{self._format_json_value(value)}\n```"

    def _format_json_value(self, value) -> str:
        return json.dumps(value, ensure_ascii=False, indent=4)

    async def one_by_one_session_id(self):
        for session_id, task in list(self.session_tasks.items()):
            if not task.done():
                continue
            self.session_tasks.pop(session_id, None)
            self.session_update_message_id.pop(session_id, None)
            try:
                task.result()
            except Exception:
                logger.exception("Feishu session task failed session_id={}", session_id)

        for session_id in list(self.receive_queues.keys()):
            if session_id in self.session_tasks:
                continue
            self.session_tasks[session_id] = asyncio.create_task(self._run_session(session_id))

    async def _run_session(self, session_id: str) -> None:
        async with self.semaphore:
            await self.consume(session_id=session_id)

    async def poll_sessions(self, interval: float = 1.0) -> None:
        while True:
            try:
                await self.one_by_one_session_id()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"poll_sessions error: {exc}")
                await asyncio.sleep(interval)

    async def _parse_msg(self, inbound_msg: InboundMessage) -> Message:
        timestamp = inbound_msg.create_time // 1000

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
            session_id = f"{inbound_msg.chat_type}_{inbound_msg.chat_id}"
            if isinstance(inbound_msg.content, TextContent):
                return Message(
                    message_id=inbound_msg.message_id,
                    chat_id=inbound_msg.chat_id,
                    session_id=session_id,
                    content=[TextData(text=inbound_msg.content.text)],
                    timestamp=timestamp,
                    session_metadata={"sender_id": inbound_msg.sender_id, "sender_name": inbound_msg.sender_name},
                )
            elif isinstance(inbound_msg.content, ImageContent):
                return Message(
                    message_id=inbound_msg.message_id,
                    chat_id=inbound_msg.chat_id,
                    session_id=session_id,
                    content=[
                        ImageData(
                            image_url=await get_image_url_from_key(
                                inbound_msg.message_id, inbound_msg.content.image_key
                            )
                        )
                    ],
                    timestamp=timestamp,
                    session_metadata={"sender_id": inbound_msg.sender_id, "sender_name": inbound_msg.sender_name},
                )
            elif isinstance(inbound_msg.content, FileContent) or isinstance(inbound_msg.content, MediaContent):
                save_filepath = FEISHU_CACHE_DIR / session_id / inbound_msg.content.file_name
                return Message(
                    message_id=inbound_msg.message_id,
                    chat_id=inbound_msg.chat_id,
                    session_id=session_id,
                    content=[
                        FileData(
                            file=await _save_file(inbound_msg.message_id, inbound_msg.content.file_key, save_filepath)
                        )
                    ],
                    timestamp=timestamp,
                    session_metadata={"sender_id": inbound_msg.sender_id, "sender_name": inbound_msg.sender_name},
                )
            elif isinstance(inbound_msg.content, PostContent):
                parsed_content: list = []
                text_buffer: list[str] = []
                for item in chain.from_iterable(inbound_msg.content.post["content"]):
                    tag = item["tag"]
                    if tag == "text":
                        text_buffer.append(item["text"])
                        continue
                    if text_buffer:
                        parsed_content.append(TextData(text="\n".join(text_buffer)))
                        text_buffer = []
                    if tag == "img":
                        image_url = await get_image_url_from_key(inbound_msg.message_id, item["image_key"])
                        parsed_content.append(ImageData(image_url=image_url))
                    else:
                        raise ValueError(f"Unknown tag: {tag}")
                if text_buffer:
                    parsed_content.append(TextData(text="\n".join(text_buffer)))
                return Message(
                    message_id=inbound_msg.message_id,
                    chat_id=inbound_msg.chat_id,
                    session_id=session_id,
                    content=parsed_content,
                    timestamp=timestamp,
                    session_metadata={"sender_id": inbound_msg.sender_id, "sender_name": inbound_msg.sender_name},
                )

    async def start(self) -> None:
        poll_task = asyncio.create_task(self.poll_sessions())
        try:
            await self.channel.connect()
        finally:
            await self.channel.disconnect()
            poll_task.cancel()
            for task in self.session_tasks.values():
                task.cancel()
            await asyncio.gather(poll_task, *self.session_tasks.values(), return_exceptions=True)


def run(chat_url: str) -> None:
    try:
        asyncio.run(
            FeishuBridge(
                chat_url=chat_url,
                app_id=settings.feishu.app_id,
                app_secret=settings.feishu.app_secret,
                parallel=settings.feishu.parallel,
            ).start()
        )
    except KeyboardInterrupt:
        print("Stopped by user")
