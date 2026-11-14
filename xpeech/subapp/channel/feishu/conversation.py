from __future__ import annotations

import json
import random

import httpx
from loguru import logger
from yarl import URL

from ..helper import download_file as download_channel_file
from ..helper import iter_chat_events
from ..schema import ChatEvent, ChatEventType, Message
from .channel import XpeechFeishuChannel
from .config import DEFAULT_SESSION_ID, EMOJI_TYPES
from .delivery import FeishuDelivery, ensure_send_success
from .formatting import FeishuEventFormatter
from .questions import FeishuQuestions, QuestionCard
from .tools import FeishuTools, ToolProgress


class FeishuConversation:
    """将聊天事件分发给发送、工具进度和问答组件，并清理每轮状态。"""

    def __init__(
        self,
        chat_url: str,
        channel: XpeechFeishuChannel,
        delivery: FeishuDelivery,
        formatter: FeishuEventFormatter,
        questions: FeishuQuestions,
        tools: FeishuTools,
    ):
        self.chat_url = chat_url
        self.channel = channel
        self.delivery = delivery
        self.formatter = formatter
        self.questions = questions
        self.tools = tools

    async def add_reaction(self, msg: Message) -> None:
        await self.channel.add_reaction(msg.message_id, random.choice(EMOJI_TYPES))

    async def consume(self, messages: list[Message]) -> None:
        message = messages[-1]
        user_id = message.user_id
        chat_id = message.chat_id
        reply_to = message.message_id
        sender_name = message.sender_name
        tool_progress = ToolProgress()
        tool_question_cards: dict[str, QuestionCard] = {}
        question_message_ids: set[str] = set()
        for source in messages:
            try:
                await self.add_reaction(source)
            except Exception:  # noqa: BLE001 - 表情反馈失败不应中断聊天。
                logger.debug(
                    "Failed to add Feishu reaction message_id={}",
                    source.message_id,
                )

        try:
            async for event in iter_chat_events(
                messages,
                str(URL(self.chat_url) / "chat"),
                "feishu",
                session_id=DEFAULT_SESSION_ID,
            ):
                if event.event in {ChatEventType.TOOL_CALL, ChatEventType.TOOL_CALL_RESULT}:
                    if event.event == ChatEventType.TOOL_CALL:
                        await self.tools.finish_tool_progress(tool_progress)
                        tool_progress = ToolProgress()
                    await self.tools.send_tool_progress(chat_id, event, tool_progress)
                    if event.event == ChatEventType.TOOL_CALL_RESULT:
                        payload = json.loads(event.context)
                        tool_call_id = payload["tool_call_id"]
                        if tool_call_id in tool_question_cards:
                            await self.questions.finish_confirmation(tool_question_cards.pop(tool_call_id), payload)
                    continue
                if event.event == ChatEventType.SEND_FILE:
                    downloaded_file = await download_channel_file(
                        user_id,
                        event.context,
                        self.chat_url,
                        "feishu",
                    )
                    result = await self.channel.send(
                        chat_id,
                        {
                            "file": {
                                "source": downloaded_file.content,
                                "file_name": downloaded_file.filename,
                            }
                        },
                    )
                    ensure_send_success(result, operation="send file")
                    continue
                if event.event == ChatEventType.QUESTION:
                    card = await self.questions.send_question(chat_id, event.context)
                    question = json.loads(event.context)
                    question_message_ids.add(card.message_id)
                    if question.get("type") == "confirm" and question.get("tool_call_id"):
                        tool_question_cards[question["tool_call_id"]] = card
                    continue

                formatted = self.formatter._format_chat_event(event)
                if formatted:
                    message, iter_content = formatted
                    await self.delivery.channel_send(
                        to=chat_id,
                        message=message,
                        opts=(
                            {"reply_to": reply_to}
                            if event.event == ChatEventType.ASSISTANT and reply_to is not None
                            else None
                        ),
                        user_id=user_id,
                        sender_name=sender_name,
                        message_type=event.event,
                        iter_content=iter_content,
                    )
        except httpx.HTTPStatusError as exc:
            await self._send_backend_error(
                exc,
                chat_id=chat_id,
                reply_to=reply_to,
                user_id=user_id,
                sender_name=sender_name,
            )
        except Exception:  # noqa: BLE001 - 聊天请求的异常边界；finally 负责清理卡片。
            logger.exception(
                "Failed to consume Feishu messages user_id={}",
                user_id,
            )
        finally:
            for message_id in question_message_ids:
                self.questions.question_cards.pop(message_id, None)
            await self.tools.finish_tool_progress(tool_progress)

    async def _send_backend_error(
        self,
        exc: httpx.HTTPStatusError,
        *,
        chat_id: str,
        reply_to: str | None,
        user_id: str,
        sender_name: str,
    ) -> None:
        logger.warning(
            "Feishu backend request failed user_id={} status_code={}",
            user_id,
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
                response_content = json.dumps(
                    response_content,
                    ensure_ascii=False,
                )

        formatted = self.formatter._format_chat_event(
            ChatEvent(
                event=ChatEventType.ERROR,
                context=f"{exc.response.status_code}: {response_content}",
            )
        )
        if formatted is None:
            return
        message, iter_content = formatted
        await self.delivery.channel_send(
            to=chat_id,
            message=message,
            opts={"reply_to": reply_to} if reply_to is not None else None,
            user_id=user_id,
            sender_name=sender_name,
            message_type=ChatEventType.ERROR,
            iter_content=iter_content,
        )
