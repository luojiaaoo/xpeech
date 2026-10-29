from __future__ import annotations

import asyncio
import json
import random

import httpx
from lark_channel import (
    CardActionEvent,
    DedupConfig,
    Events,
    InboundMessage,
    OutboundConfig,
    PolicyConfig,
    RetryConfig,
    SafetyConfig,
)
from loguru import logger
from yarl import URL

from ....agent.server.schema import BackgroundMessageChannel, FeishuBackgroundMessage
from ....config.settings import settings
from ....utils.jwt_auth import create_access_token
from ..helper import download_file as download_channel_file
from ..helper import iter_chat_events, notify_question
from ..schema import ChatEvent, ChatEventType, Message
from .cards import (
    FINISH_CARD_CONTENT,
    build_feishu_background_task_card,
    build_feishu_question_card,
)
from .channel import XpeechFeishuChannel
from .config import EMOJI_TYPES
from .delivery import FeishuDeliveryMixin
from .formatting import FeishuEventFormatter
from .inbound import FeishuInboundMixin, UnsupportedFeishuMessageError

BACKGROUND_MESSAGE_RETRY_SECONDS = 2.0


class FeishuBridge(
    FeishuInboundMixin,
    FeishuDeliveryMixin,
    FeishuEventFormatter,
):
    """将标准化后的飞书消息桥接到 Xpeech 的 ``/chat`` 接口。"""

    def __init__(self, chat_url: str, app_id: str, app_secret: str):
        self.chat_url = chat_url
        self.app_id = app_id
        self.app_secret = app_secret
        self.session_update_message_id: dict[str, str] = {}
        self.channel = XpeechFeishuChannel(
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

    async def add_reaction(self, msg: Message) -> None:
        await self.channel.add_reaction(msg.message_id, random.choice(EMOJI_TYPES))

    async def _on_message(self, inbound_message: InboundMessage) -> None:
        scope = inbound_message.chat_id or inbound_message.message_id
        self.channel.block_batch_scope(scope)
        messages: list[Message] = []
        try:
            for source in inbound_message.batched_sources or [inbound_message]:
                try:
                    messages.append(await self._parse_msg(source))
                except UnsupportedFeishuMessageError as exc:
                    logger.warning(str(exc))
            if messages:
                await self.consume(messages)
        finally:
            for message in messages:
                self.session_update_message_id.pop(message.user_id, None)
            self.channel.unblock_batch_scope(scope)

    async def _on_card_action(self, card_action_event: CardActionEvent) -> None:
        user_id, _ = await self.get_user_identity(card_action_event.operator.open_id)
        form_data = card_action_event.action.form_value
        await notify_question(user_id, form_data, self.chat_url)
        result = await self.channel.update_card(
            card_action_event.message_id,
            FINISH_CARD_CONTENT,
        )
        self._ensure_send_success(
            result,
            operation="update completed question card",
        )

    async def consume(self, messages: list[Message]) -> None:
        message = messages[-1]
        user_id = message.user_id
        chat_id = message.chat_id
        reply_to = message.message_id
        sender_name = message.sender_name
        for source in messages:
            try:
                await self.add_reaction(source)
            except Exception:
                logger.debug(
                    "Failed to add Feishu reaction message_id={}",
                    source.message_id,
                )

        try:
            async for event in iter_chat_events(
                messages,
                str(URL(self.chat_url) / "chat"),
            ):
                if event.event == ChatEventType.SEND_FILE:
                    downloaded_file = await download_channel_file(
                        user_id,
                        event.context,
                        self.chat_url,
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
                    self._ensure_send_success(result, operation="send file")
                    continue
                if event.event == ChatEventType.QUESTION:
                    result = await self.channel.send(
                        chat_id,
                        {"card": build_feishu_question_card(event.context)},
                    )
                    self._ensure_send_success(
                        result,
                        operation="send question card",
                    )
                    continue

                formatted = self._format_chat_event(event)
                if formatted:
                    message, iter_content = formatted
                    await self.channel_send(
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
        except Exception:
            logger.exception(
                "Failed to consume Feishu messages user_id={}",
                user_id,
            )

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

        formatted = self._format_chat_event(
            ChatEvent(
                event=ChatEventType.ERROR,
                context=f"{exc.response.status_code}: {response_content}",
            )
        )
        if formatted is None:
            return
        message, iter_content = formatted
        await self.channel_send(
            to=chat_id,
            message=message,
            opts={"reply_to": reply_to} if reply_to is not None else None,
            user_id=user_id,
            sender_name=sender_name,
            message_type=ChatEventType.ERROR,
            iter_content=iter_content,
        )

    async def poll_background_message_once(
        self,
        client: httpx.AsyncClient,
    ) -> bool:
        """Poll once and immediately deliver an available background message."""
        response = await client.get(
            str(URL(self.chat_url) / "background_message"),
            headers={"authorization": f"Bearer {create_access_token()}"},
            params={"channel": BackgroundMessageChannel.FEISHU.value},
        )
        if response.status_code == 204:
            return False
        response.raise_for_status()

        try:
            background_message = FeishuBackgroundMessage.model_validate(response.json())
        except ValueError as exc:
            raise RuntimeError("Backend returned an invalid Feishu background message") from exc

        result = await self.channel.send(
            to=background_message.open_id,
            message={
                "card": build_feishu_background_task_card(
                    background_message.content,
                    schedule_type=background_message.schedule_type,
                    schedule_value=background_message.schedule_value,
                ),
            },
        )
        self._ensure_send_success(result, operation="send background message")
        logger.info(
            "Feishu background message sent open_id={}",
            background_message.open_id,
        )
        return True

    async def poll_background_messages(self) -> None:
        """Continuously long-poll and deliver scheduled Agent results."""
        async with httpx.AsyncClient(timeout=None) as client:
            while True:
                try:
                    await self.poll_background_message_once(client)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Failed to poll or send Feishu background message")
                    await asyncio.sleep(BACKGROUND_MESSAGE_RETRY_SECONDS)

    async def start(self) -> None:
        try:
            async with asyncio.TaskGroup() as task_group:
                background_message_task = task_group.create_task(self.poll_background_messages())
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
