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
    CardActionEvent,
)
import asyncio
from .schema import ChatEvent, ChatEventType, Message, TextData, FileData
import json
import mimetypes
import lark_oapi as lark
from lark_oapi.api.im.v1 import GetMessageResourceRequest, GetMessageResourceResponse
from .helper import download_file as download_channel_file
from .helper import iter_chat_events, notify_question
import random
from pathlib import Path
import aiofiles
from itertools import chain
from datetime import datetime
from typing import Any, NotRequired, TypedDict
from loguru import logger
from ..config.settings import settings
from ..agent.tools.question import validate_question_json
from ..utils.helper import detect_image_mime, ensure_path
from lark_oapi.channel import Events
from yarl import URL


class OutputEventType(TypedDict):
    content: str | None | type(Ellipsis)
    text_size: NotRequired[str | dict[str, Any]]
    text_align: NotRequired[str]
    icon: NotRequired[dict[str, Any] | None]


FINISH_CARD_CONTENT = {
    "schema": "2.0",
    "config": {
        "update_multi": True,
        "style": {"text_size": {"normal_v2": {"default": "normal", "pc": "normal", "mobile": "heading"}}},
    },
    "body": {
        "direction": "vertical",
        "horizontal_spacing": "8px",
        "vertical_spacing": "8px",
        "horizontal_align": "center",
        "vertical_align": "center",
        "padding": "12px 12px 12px 12px",
        "elements": [
            {
                "tag": "markdown",
                "content": ":OK:<font color='red'>表单填写完成</font>",
                "text_align": "left",
                "text_size": "normal_v2",
                "margin": "0px 0px 0px 0px",
            }
        ],
    },
}


def _plain_text(content: str) -> dict[str, str]:
    return {"tag": "plain_text", "content": content}


def _feishu_label(content: str) -> dict[str, Any]:
    return {
        "tag": "div",
        "text": {
            "tag": "plain_text",
            "content": content,
            "text_size": "normal_v2",
            "text_align": "left",
            "text_color": "default",
        },
        "margin": "0px 0px 0px 0px",
    }


def _feishu_options(options: list[dict[str, str]], *, include_icon: bool = False) -> list[dict[str, Any]]:
    feishu_options = []
    for option in options:
        feishu_option: dict[str, Any] = {
            "text": _plain_text(option["label"]),
            "value": option["value"],
        }
        if include_icon:
            feishu_option["icon"] = {"tag": "standard_icon", "token": "signature_outlined"}
        feishu_options.append(feishu_option)
    return feishu_options


def _feishu_field(field: dict[str, Any]) -> list[dict[str, Any]]:
    placeholder = {
        "tag": "plain_text",
        "content": field.get("placeholder") or "请选择",
    }
    field_type = field["type"]
    elements: list[dict[str, Any]] = [_feishu_label(field["label"])]

    if field_type == "input":
        elements.append(
            {
                "tag": "input",
                "placeholder": {
                    "tag": "plain_text",
                    "content": field.get("placeholder") or "请输入",
                },
                "default_value": field.get("default_value") or "",
                "width": "fill",
                "required": False,
                "name": field["name"],
                "margin": "0px 0px 0px 0px",
            }
        )
    elif field_type == "select":
        elements.append(
            {
                "tag": "select_static",
                "placeholder": placeholder,
                "options": _feishu_options(field["options"]),
                "type": "default",
                "width": "fill",
                "required": False,
                "name": field["name"],
                "margin": "0px 0px 0px 0px",
            }
        )
    elif field_type == "multi_select":
        elements.append(
            {
                "tag": "multi_select_static",
                "placeholder": placeholder,
                "options": _feishu_options(field["options"], include_icon=True),
                "type": "default",
                "width": "fill",
                "required": False,
                "name": field["name"],
                "margin": "0px 0px 0px 0px",
            }
        )
    elif field_type == "date":
        elements.append(
            {
                "tag": "date_picker",
                "placeholder": placeholder,
                "width": "fill",
                "required": False,
                "name": field["name"],
                "margin": "0px 0px 0px 0px",
            }
        )
    elif field_type == "datetime":
        elements.append(
            {
                "tag": "picker_datetime",
                "placeholder": placeholder,
                "width": "fill",
                "required": False,
                "name": field["name"],
                "margin": "0px 0px 0px 0px",
            }
        )
    else:
        raise ValueError(f"Unsupported question field type: {field_type}")

    return elements


def build_feishu_question_card(question_context: str) -> dict[str, Any]:
    form = validate_question_json(question_context)
    elements: list[dict[str, Any]] = []
    for field in form["fields"]:
        elements.extend(_feishu_field(field))

    elements.extend(
        [
            _feishu_label("自定义"),
            {
                "tag": "input",
                "placeholder": {
                    "tag": "plain_text",
                    "content": "请输入",
                },
                "default_value": "",
                "width": "fill",
                "required": False,
                "name": "user_customization",
                "margin": "0px 0px 0px 0px",
            },
        ]
    )

    elements.append(
        {
            "tag": "column_set",
            "horizontal_align": "left",
            "columns": [
                {
                    "tag": "column",
                    "width": "auto",
                    "elements": [
                        {
                            "tag": "button",
                            "text": _plain_text(form.get("submit_label") or "提交"),
                            "type": "primary",
                            "width": "default",
                            "form_action_type": "submit",
                            "name": "submit_question",
                        }
                    ],
                    "vertical_align": "top",
                },
                {"tag": "column", "width": "auto", "elements": [], "vertical_align": "top"},
            ],
        }
    )

    return {
        "schema": "2.0",
        "config": {
            "update_multi": True,
            "style": {"text_size": {"normal_v2": {"default": "normal", "pc": "normal", "mobile": "heading"}}},
        },
        "body": {
            "direction": "vertical",
            "padding": "12px 12px 12px 12px",
            "elements": [
                {
                    "tag": "form",
                    "elements": elements,
                    "direction": "vertical",
                    "padding": "4px 0px 4px 0px",
                    "margin": "0px 0px 0px 0px",
                    "name": "question_form",
                }
            ],
        },
        "header": {
            "title": _plain_text(form["title"]),
            "subtitle": _plain_text(form["subtitle"]),
            "template": "blue",
            "padding": "12px 12px 12px 12px",
        },
    }

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

FEISHU_CACHE_DIR = ensure_path(settings.path.cache_path.resolve())


class FeishuBridge:
    """Bridge normalized Feishu messages into the Xpeech /chat SSE endpoint."""

    LOG_LEVEL = LogLevel.INFO

    def __init__(self, chat_url: str, app_id: str, app_secret: str):
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
        self.channel.on(Events.MESSAGE, self._on_message)
        self.channel.on(Events.CARD_ACTION, self._on_card_action)

    def session_id(self, chat_id: str):
        return f"feishu_{chat_id}"

    async def add_reaction(self, msg: Message):
        await self.channel.add_reaction(msg.message_id, random.choice(_EMOJI_TYPES))

    async def _on_message(self, inbound_message: InboundMessage) -> None:
        msg = await self._parse_msg(inbound_message)
        if msg.session_id not in self.receive_queues:
            self.receive_queues[msg.session_id] = asyncio.Queue()
        self.receive_queues[msg.session_id].put_nowait(msg)

    async def _on_card_action(self, card_action_event: CardActionEvent) -> None:
        session_id = self.session_id(card_action_event.chat_id)
        form_data = card_action_event.raw["event"]["action"]["form_value"]
        await notify_question(session_id, form_data, self.chat_url)
        await self.channel.update_card(card_action_event.message_id, FINISH_CARD_CONTENT)

    async def channel_send(
        self, to: str, message: dict, opts: dict | None, session_id: str, message_type: ChatEventType
    ):
        persistence_tyle = (
            ChatEventType.ASSISTANT,
            ChatEventType.COMMAND,
            ChatEventType.TOKEN_USAGE,
            ChatEventType.QUESTION,
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
            async for event in iter_chat_events(messages, str(URL(self.chat_url) / "chat")):
                # 检测有没有文件发送请求
                if event.event == ChatEventType.SEND_FILE:
                    downloaded_file = await download_channel_file(session_id, event.context, self.chat_url)
                    await self.channel.send(
                        chat_id,
                        {"file": {"source": downloaded_file.content, "file_name": downloaded_file.filename}},
                    )
                    continue
                if event.event == ChatEventType.QUESTION:
                    await self.channel.send(chat_id, {"card": build_feishu_question_card(event.context)})
                    continue

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
            return f'**[词元]** {self._format_token_usage(event.context)}'
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

    def _format_token_usage(self, context: str) -> str:
        data: dict[str, str] = json.loads(context)
        return " | ".join(f"{k}：{v}" for k, v in data.items())

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
            self.session_tasks[session_id] = asyncio.create_task(self.consume(session_id))

    async def poll_sessions(self, interval: float = 2.0) -> None:
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

        async def _save_resource(
            message_id: str,
            file_key: str,
            resource_type: str,
            save_filepath: Path,
        ) -> Path:
            client = self.channel.client
            request: GetMessageResourceRequest = (
                GetMessageResourceRequest.builder()
                .message_id(message_id)
                .file_key(file_key)
                .type(resource_type)
                .build()
            )
            response: GetMessageResourceResponse = client.im.v1.message_resource.get(request)
            if not response.success():
                error = (
                    f"client.im.v1.message_resource.get failed, code: {response.code}, msg: {response.msg}, "
                    f"log_id: {response.get_log_id()}, resp: \n"
                    f"{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}"
                )
                lark.logger.error(error)
                raise RuntimeError(error)
            raw = response.file.read()
            if resource_type == "image" and not save_filepath.suffix:
                mime = detect_image_mime(raw)
                save_filepath = save_filepath.with_suffix(mimetypes.guess_extension(mime or "") or ".jpg")
            save_filepath.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(save_filepath, "wb") as f:
                await f.write(raw)
            return save_filepath

        if inbound_msg.chat_type == "p2p":
            session_id = self.session_id(inbound_msg.chat_id)
            if isinstance(inbound_msg.content, TextContent):
                return Message(
                    message_id=inbound_msg.message_id,
                    chat_id=inbound_msg.chat_id,
                    session_id=session_id,
                    sender_name=inbound_msg.sender_name,
                    content=[TextData(text=inbound_msg.content.text)],
                    timestamp=timestamp,
                    session_metadata={"sender_id": inbound_msg.sender_id},
                )
            elif isinstance(inbound_msg.content, ImageContent):
                save_filepath = FEISHU_CACHE_DIR / session_id / inbound_msg.message_id
                return Message(
                    message_id=inbound_msg.message_id,
                    chat_id=inbound_msg.chat_id,
                    session_id=session_id,
                    sender_name=inbound_msg.sender_name,
                    content=[
                        FileData(
                            file=await _save_resource(
                                inbound_msg.message_id,
                                inbound_msg.content.image_key,
                                "image",
                                save_filepath,
                            )
                        )
                    ],
                    timestamp=timestamp,
                    session_metadata={"sender_id": inbound_msg.sender_id},
                )
            elif isinstance(inbound_msg.content, FileContent) or isinstance(inbound_msg.content, MediaContent):
                save_filepath = FEISHU_CACHE_DIR / session_id / inbound_msg.content.file_name
                return Message(
                    message_id=inbound_msg.message_id,
                    chat_id=inbound_msg.chat_id,
                    session_id=session_id,
                    sender_name=inbound_msg.sender_name,
                    content=[
                        FileData(
                            file=await _save_resource(
                                inbound_msg.message_id,
                                inbound_msg.content.file_key,
                                "file",
                                save_filepath,
                            )
                        )
                    ],
                    timestamp=timestamp,
                    session_metadata={"sender_id": inbound_msg.sender_id},
                )
            elif isinstance(inbound_msg.content, PostContent):
                parsed_content: list = []
                text_buffer: list[str] = []
                image_index = 0
                for item in chain.from_iterable(inbound_msg.content.post["content"]):
                    tag = item["tag"]
                    if tag == "text":
                        text_buffer.append(item["text"])
                        continue
                    elif tag == "a":
                        text_buffer.append(f'[{item["text"]}]({item["href"]})')
                        continue
                    # 多行合并成整体Text数据
                    if text_buffer:
                        parsed_content.append(TextData(text="\n".join(text_buffer)))
                        text_buffer = []
                    if tag == "img":
                        image_index += 1
                        save_filepath = FEISHU_CACHE_DIR / session_id / f"{inbound_msg.message_id}_{image_index}"
                        saved_file = await _save_resource(
                            inbound_msg.message_id,
                            item["image_key"],
                            "image",
                            save_filepath,
                        )
                        parsed_content.extend(
                            (
                                TextData(text=f"[Attachment: {saved_file.name}]"),
                                FileData(file=saved_file),
                            )
                        )
                    else:
                        raise ValueError(f"Unknown tag: {tag}")
                if text_buffer:
                    parsed_content.append(TextData(text="\n".join(text_buffer)))
                return Message(
                    message_id=inbound_msg.message_id,
                    chat_id=inbound_msg.chat_id,
                    session_id=session_id,
                    sender_name=inbound_msg.sender_name,
                    content=parsed_content,
                    timestamp=timestamp,
                    session_metadata={"sender_id": inbound_msg.sender_id},
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
            ).start()
        )
    except KeyboardInterrupt:
        print("Stopped by user")
