from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from lark_channel import CardActionEvent
from loguru import logger

from ..helper import notify_question
from .cards import build_feishu_question_card
from .channel import XpeechFeishuChannel
from .config import DEFAULT_SESSION_ID
from .delivery import ensure_send_success
from .inbound import FeishuInbound


@dataclass
class QuestionCard:
    message_id: str
    context: str
    disabled: bool = False
    submitting: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class FeishuQuestions:
    """管理问题卡片及提交、执行完成之间的并发状态。"""

    def __init__(self, chat_url: str, channel: XpeechFeishuChannel, inbound: FeishuInbound):
        self.chat_url = chat_url
        self.channel = channel
        self.inbound = inbound
        self.question_cards: dict[str, QuestionCard] = {}

    async def send_question(self, to: str, context: str) -> QuestionCard:
        result = await self.channel.send(to, {"card": build_feishu_question_card(context)})
        ensure_send_success(result, operation="send question card", require_message_id=True)
        card = QuestionCard(result.message_id, context)
        self.question_cards[card.message_id] = card
        return card

    async def handle_action(self, card_action_event: CardActionEvent) -> None:
        user_id, _ = await self.inbound.get_user_identity(card_action_event.operator.open_id)
        form_data = card_action_event.action.form_value
        if form_data is None:
            button_data = card_action_event.action.value
            if isinstance(button_data, dict) and button_data.get("decision") in ("confirm", "cancel"):
                form_data = {"decision": button_data["decision"]}
        card = self.question_cards.get(card_action_event.message_id)
        if card is None:
            return
        async with card.lock:
            if card.disabled or card.submitting:
                return
            card.submitting = True
        try:
            await notify_question(
                user_id,
                form_data,
                self.chat_url,
                "feishu",
                session_id=DEFAULT_SESSION_ID,
            )
            async with card.lock:
                # 执行结果可能先于提交响应到达，保留已经完成的卡片。
                if card.disabled:
                    return
                card.disabled = True
                try:
                    result = await self.channel.update_card(
                        card.message_id,
                        build_feishu_question_card(card.context, disabled=True, answers=form_data),
                    )
                    ensure_send_success(result, operation="disable submitted question")
                except Exception:  # noqa: BLE001 - 答案已提交，卡片刷新失败只记录日志。
                    logger.exception("Failed to disable Feishu question card message_id={}", card.message_id)
        finally:
            async with card.lock:
                card.submitting = False

    async def finish_confirmation(self, card: QuestionCard, tool_result: dict[str, Any]) -> None:
        async with card.lock:
            card.disabled = True
            result = await self.channel.update_card(
                card.message_id,
                build_feishu_question_card(card.context, tool_result=tool_result),
            )
            ensure_send_success(result, operation="update execution confirmation card")
            self.question_cards.pop(card.message_id, None)
