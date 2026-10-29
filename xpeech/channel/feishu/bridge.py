from __future__ import annotations

import asyncio
import json
import random

import httpx
from lark_channel import (
    CardActionEvent,
    DedupConfig,
    Events,
    FeishuChannel,
    InboundMessage,
    OutboundConfig,
    PolicyConfig,
    RetryConfig,
    SafetyConfig,
)
from loguru import logger
from yarl import URL

from ...agent.background import BackgroundMessageChannel, FeishuBackgroundMessage
from ...config.settings import settings
from ...utils.jwt_auth import create_access_token
from ..helper import download_file as download_channel_file
from ..helper import iter_chat_events, notify_question
from ..schema import ChatEvent, ChatEventType, Message
from .cards import (
    FINISH_CARD_CONTENT,
    build_feishu_background_task_card,
    build_feishu_question_card,
)
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
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._task_group: asyncio.TaskGroup | None = None
        self._closing = False
        self.receive_queues: dict[str, asyncio.Queue[Message]] = {}
        self.session_tasks: dict[str, asyncio.Task] = {}
        self.session_update_message_id: dict[str, str] = {}
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

    async def add_reaction(self, msg: Message) -> None:
        await self.channel.add_reaction(msg.message_id, random.choice(EMOJI_TYPES))

    async def _on_message(self, inbound_message: InboundMessage) -> None:
        try:
            msg = await self._parse_msg(inbound_message)
        except UnsupportedFeishuMessageError as exc:
            logger.warning(str(exc))
            return
        if self._main_loop is None:
            raise RuntimeError("Feishu bridge has not been started")
        self._main_loop.call_soon_threadsafe(self._enqueue_message, msg)

    def _enqueue_message(self, msg: Message) -> None:
        """Enqueue and start consumption on the bridge's main event loop."""
        if self._closing:
            return
        queue = self.receive_queues.setdefault(msg.user_id, asyncio.Queue())
        queue.put_nowait(msg)
        logger.info(
            "Feishu message added user_id={} chat_id={} message_id={} sender_name={} queue_size={}",
            msg.user_id,
            msg.chat_id,
            msg.message_id,
            msg.sender_name,
            queue.qsize(),
        )
        self._start_session_task(msg.user_id)

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

    async def consume(self, user_id: str) -> None:
        if user_id not in self.receive_queues:
            return

        queue = self.receive_queues[user_id]
        if queue.empty():
            return

        message = queue.get_nowait()
        chat_id = message.chat_id
        reply_to = message.message_id
        sender_name = message.sender_name
        try:
            await self.add_reaction(message)
        except Exception:
            logger.debug(
                "Failed to add Feishu reaction message_id={}",
                message.message_id,
            )

        try:
            async for event in iter_chat_events(
                [message],
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

    def _start_session_task(self, user_id: str) -> None:
        if self._task_group is None:
            raise RuntimeError("Feishu bridge task group is not running")
        task = self.session_tasks.get(user_id)
        if task is not None and not task.done():
            return
        if task is not None:
            self._on_session_task_done(user_id, task)

        task = self._task_group.create_task(self._consume_user_queue(user_id))
        self.session_tasks[user_id] = task
        task.add_done_callback(
            lambda completed_task: self._on_session_task_done(user_id, completed_task)
        )

    async def _consume_user_queue(self, user_id: str) -> None:
        try:
            queue = self.receive_queues[user_id]
            while not queue.empty():
                await self.consume(user_id)
        except Exception:
            logger.exception(
                "Feishu session task failed user_id={}",
                user_id,
            )

    def _on_session_task_done(self, user_id: str, task: asyncio.Task) -> None:
        if self.session_tasks.get(user_id) is not task:
            return
        self.session_tasks.pop(user_id, None)
        self.session_update_message_id.pop(user_id, None)

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
            message={"card": build_feishu_background_task_card(background_message.content)},
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
        self._main_loop = asyncio.get_running_loop()
        self._closing = False
        try:
            async with asyncio.TaskGroup() as task_group:
                self._task_group = task_group
                background_message_task = task_group.create_task(self.poll_background_messages())
                try:
                    await self.channel.connect()
                finally:
                    self._closing = True
                    background_message_task.cancel()
                    for task in tuple(self.session_tasks.values()):
                        task.cancel()
        finally:
            self._task_group = None
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
