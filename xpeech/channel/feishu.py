from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import httpx
from async_lru import alru_cache
from lark_channel import (
    CardActionEvent,
    DedupConfig,
    Events,
    FeishuChannel,
    FileContent,
    ImageContent,
    InboundMessage,
    MediaContent,
    OutboundConfig,
    PolicyConfig,
    PostContent,
    RetryConfig,
    SafetyConfig,
    SendResult,
    TextContent,
)
from lark_channel.api.contact.v3.model.get_user_request import GetUserRequest
from loguru import logger
from yarl import URL

from ..agent.tools.question import validate_question_json
from ..config.settings import settings
from ..utils.helper import ensure_path
from .helper import download_file as download_channel_file
from .helper import iter_chat_events, notify_question
from .schema import ChatEvent, ChatEventType, FileData, Message, TextData


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
# None 字符串；...流式； str 直接输出；_format_chat_event_content 也要联动修改
OUTPUT_EVENT_TYPES: dict[ChatEventType, OutputEventType] = {
    ChatEventType.THINKING: {
        "content": ...,
        "text_size": "notation",
        "text_align": "left",
        "icon": {"tag": "standard_icon", "token": "tab-more_outlined", "color": "green"},
    },
    ChatEventType.ASSISTANT: {
        "content": ...,
        "text_size": "normal",
        "text_align": "left",
        "icon": {"tag": "standard_icon", "token": "robot_filled", "color": "red"},
    },
    ChatEventType.COMMAND: {
        "content": None,
        "text_size": "notation",
        "text_align": "center",
        "icon": {"tag": "standard_icon", "token": "command_outlined", "color": "turquoise"},
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
        "content": None,
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
FEISHU_USER_CACHE_TTL_SECONDS = 3600
FEISHU_STREAM_UPDATE_INTERVAL_SECONDS = 0.5


class FeishuBridge:
    """Bridge normalized Feishu messages into the Xpeech /chat SSE endpoint."""

    def __init__(self, chat_url: str, app_id: str, app_secret: str):
        self.chat_url = chat_url
        self.app_id = app_id
        self.app_secret = app_secret
        self.receive_queues: dict[str, asyncio.Queue[Message]] = {}
        self.session_tasks: dict[str, asyncio.Task] = {}
        self.session_update_message_id: dict[str, str] = {}
        # 初始化通道
        self.channel = FeishuChannel(
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

    @alru_cache(maxsize=2048, ttl=FEISHU_USER_CACHE_TTL_SECONDS)
    async def get_user_identity(self, open_id: str) -> tuple[str, str | None]:
        if not open_id:
            raise RuntimeError("Cannot resolve Feishu user identity without an open_id")

        request = GetUserRequest.builder().user_id(open_id).user_id_type("open_id").build()
        response = await self.channel.client.contact.v3.user.aget(request)
        if not response.success():
            raise RuntimeError(
                f"Failed to get Feishu user identity: code={response.code}, msg={response.msg}, open_id={open_id}"
            )

        user = response.data.user if response.data else None
        employee_no = user.employee_no if user else None
        if not employee_no:
            raise RuntimeError(f"Feishu user has no employee_no: open_id={open_id}")
        email = (getattr(user, "email", None) or getattr(user, "enterprise_email", None)) if user else None
        if not email:
            logger.warning(
                "Feishu user has no readable email; grant the user email field permission: open_id={}",
                open_id,
            )
        return employee_no, email

    async def add_reaction(self, msg: Message):
        await self.channel.add_reaction(msg.message_id, random.choice(_EMOJI_TYPES))

    async def _on_message(self, inbound_message: InboundMessage) -> None:
        msg = await self._parse_msg(inbound_message)
        queue = self.receive_queues.setdefault(msg.session_id, asyncio.Queue())
        queue.put_nowait(msg)
        logger.info(
            "Feishu message added session_id={} chat_id={} message_id={} sender_name={} queue_size={}",
            msg.session_id,
            msg.chat_id,
            msg.message_id,
            msg.sender_name,
            queue.qsize(),
        )

    async def _on_card_action(self, card_action_event: CardActionEvent) -> None:
        session_id, _ = await self.get_user_identity(card_action_event.operator.open_id)
        form_data = card_action_event.action.form_value
        await notify_question(session_id, form_data, self.chat_url)
        result = await self.channel.update_card(card_action_event.message_id, FINISH_CARD_CONTENT)
        self._ensure_send_success(result, operation="update completed question card")

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
        if iter_content:
            card_id = await self.channel.create_card_instance(message["card"])
            send_result = await self.channel.send_card_by_reference(to, card_id, **(opts or {}))
            if not send_result.success:
                raise RuntimeError(send_result.error)

            seq = 0
            accumulated = ""
            last_sent = ""
            loop = asyncio.get_running_loop()
            next_update_at = loop.time() + FEISHU_STREAM_UPDATE_INTERVAL_SECONDS

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
                next_update_at = loop.time() + FEISHU_STREAM_UPDATE_INTERVAL_SECONDS

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
        else:
            return await self.channel.send(to=to, message=message, opts=opts)

    async def channel_send(
        self,
        to: str,
        message: dict | AsyncIterator[str],
        opts: dict | None,
        session_id: str,
        sender_name: str,
        message_type: ChatEventType,
        iter_content: AsyncIterator[str] | None,
    ):
        persistence_types = (
            ChatEventType.THINKING,
            ChatEventType.ASSISTANT,
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
            result = await self.send(to=to, message=message, iter_content=iter_content, opts=opts)
            self._ensure_send_success(result, operation="send message")
            self.session_update_message_id.pop(session_id, None)
        else:
            _message_id = self.session_update_message_id.get(session_id)
            if _message_id is None:
                send_result = await self.send(to=to, message=message, iter_content=iter_content, opts=opts)
                self._ensure_send_success(
                    send_result,
                    operation="send progress card",
                    require_message_id=True,
                )
                self.session_update_message_id[session_id] = send_result.message_id
            else:
                if isinstance(message, AsyncIterator):
                    raise RuntimeError("Cannot update progress card with streaming message")
                await asyncio.sleep(0.25)
                result = await self.channel.update_card(_message_id, message["card"])
                self._ensure_send_success(result, operation="update progress card")

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
        sender_name = messages[-1].sender_name
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
                    result = await self.channel.send(
                        chat_id,
                        {"file": {"source": downloaded_file.content, "file_name": downloaded_file.filename}},
                    )
                    self._ensure_send_success(result, operation="send file")
                    continue
                if event.event == ChatEventType.QUESTION:
                    result = await self.channel.send(chat_id, {"card": build_feishu_question_card(event.context)})
                    self._ensure_send_success(result, operation="send question card")
                    continue
                # 返回给用户消息
                message, iter_content = self._format_chat_event(event)
                if message:
                    await self.channel_send(
                        to=chat_id,
                        message=message,
                        opts=(
                            {"reply_to": reply_to}
                            if event.event == ChatEventType.ASSISTANT and reply_to is not None
                            else None
                        ),
                        session_id=session_id,
                        sender_name=sender_name,
                        message_type=event.event,
                        iter_content=iter_content,
                    )
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Feishu backend request failed session_id={} status_code={}",
                session_id,
                exc.response.status_code,
            )
            try:
                response_data = exc.response.json()
            except json.JSONDecodeError:
                response_content = exc.response.text
            else:
                response_content = (
                    response_data.get("detail", response_data) if isinstance(response_data, dict) else response_data
                )
                if not isinstance(response_content, str):
                    response_content = json.dumps(response_content, ensure_ascii=False)
            card = self._format_chat_event(
                ChatEvent(
                    event=ChatEventType.ASSISTANT,
                    context=f"{exc.response.status_code}: {response_content}",
                )
            )
            if card:
                result = await self.channel.send(
                    chat_id,
                    card,
                    *([{"reply_to": reply_to}] if reply_to is not None else []),
                )
                self._ensure_send_success(result, operation="send error message")
        except Exception:
            logger.exception("Failed to consume Feishu messages session_id={}", session_id)

    def _format_chat_event_content(self, event: ChatEvent) -> str:
        content = event.context
        if event.event == ChatEventType.ASSISTANT:
            return "输出中..."
        elif event.event == ChatEventType.COMMAND:
            return f"**[命令]** {content}"
        elif event.event == ChatEventType.THINKING:
            return "思考中..."
        elif event.event == ChatEventType.TOOL_CALL:
            return f"**[调用工具]** {self._format_tool_call_event(content)}"
        elif event.event == ChatEventType.TOOL_CALL_RESULT:
            return f"**[工具调用结果]** {self._format_json_context(content)}"
        elif event.event == ChatEventType.TOKEN_USAGE:
            return f"**[词元]** {self._format_token_usage(content)}"
        return f"[{event.event}]\n{content}"

    def _format_chat_event(self, event: ChatEvent) -> tuple[bool, dict[str, Any], AsyncIterator[str] | None]:
        if event.event not in OUTPUT_EVENT_TYPES:
            return None

        output = OUTPUT_EVENT_TYPES[event.event]
        content = output["content"]
        if isinstance(content, str):
            text = content
        else:
            text = self._format_chat_event_content(event)

        element = {
            "tag": "markdown",
            "margin": "0px 0px 0px 0px",
            "content": text,
            "text_size": output.get("text_size", "normal"),
            "text_align": output.get("text_align", "left"),
        }
        if output.get("icon") is not None:
            element["icon"] = output["icon"]

        card_element = element
        if event.event == ChatEventType.THINKING:
            card_element = {
                "tag": "collapsible_panel",
                "expanded": False,
                "header": {
                    "title": _plain_text("查看思考过程"),
                    "expanded_title": _plain_text("收起思考过程"),
                },
                "elements": [element],
            }

        if content is ...:
            element["element_id"] = "main"
            return (
                {
                    "card": {
                        "schema": "2.0",
                        "config": {"streaming_mode": True, "summary": {"content": ""}},
                        "body": {
                            "elements": [card_element],
                        },
                    }
                },
                event.context,
            )
        else:
            return (
                {
                    "card": {
                        "schema": "2.0",
                        "body": {
                            "elements": [card_element],
                        },
                    }
                },
                None,
            )

    def _format_tool_call_event(self, content: str) -> str:
        try:
            tool_calls = json.loads(content)
        except json.JSONDecodeError:
            return content

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
            dest_dir: Path,
            file_name: str | None = None,
        ) -> Path:
            kwargs: dict[str, Any] = {
                "resource_type": resource_type,
                "message_id": message_id,
                "dest_dir": dest_dir,
            }
            if file_name:
                kwargs["file_name"] = file_name
            return await self.channel.download_resource_to_file(file_key, **kwargs)

        if inbound_msg.chat_type == "p2p":
            session_id, email = await self.get_user_identity(inbound_msg.sender_id)
            resource_dest_dir = FEISHU_CACHE_DIR / session_id
            session_metadata = {"email": email} if email else {}
            if isinstance(inbound_msg.content, TextContent):
                return Message(
                    message_id=inbound_msg.message_id,
                    chat_id=inbound_msg.chat_id,
                    session_id=session_id,
                    sender_name=inbound_msg.sender_name,
                    content=[TextData(text=inbound_msg.content.text)],
                    timestamp=timestamp,
                    session_metadata=session_metadata,
                )
            elif isinstance(inbound_msg.content, ImageContent):
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
                                resource_dest_dir,
                            )
                        )
                    ],
                    timestamp=timestamp,
                    session_metadata=session_metadata,
                )
            elif isinstance(inbound_msg.content, (FileContent, MediaContent)):
                resource_type = "video" if isinstance(inbound_msg.content, MediaContent) else "file"
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
                                resource_type,
                                resource_dest_dir,
                                inbound_msg.content.file_name,
                            )
                        )
                    ],
                    timestamp=timestamp,
                    session_metadata=session_metadata,
                )
            elif isinstance(inbound_msg.content, PostContent):
                parsed_content: list = []
                text_buffer: list[str] = []
                post = inbound_msg.content.post
                post_document = (
                    post
                    if "content" in post
                    else next(
                        (document for document in post.values() if isinstance(document, dict)),
                        {},
                    )
                )
                for item in chain.from_iterable(post_document.get("content") or []):
                    tag = item["tag"]
                    if tag == "text":
                        text_buffer.append(item["text"])
                        continue
                    elif tag == "a":
                        text_buffer.append(f"[{item['text']}]({item['href']})")
                        continue
                    # 多行合并成整体Text数据
                    if text_buffer:
                        parsed_content.append(TextData(text="\n".join(text_buffer)))
                        text_buffer = []
                    if tag == "img":
                        saved_file = await _save_resource(
                            inbound_msg.message_id,
                            item["image_key"],
                            "image",
                            resource_dest_dir,
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
                    session_metadata=session_metadata,
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
