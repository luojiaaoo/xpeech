from __future__ import annotations

import asyncio
import json
import random
from time import time

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
    build_feishu_markdown_card,
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
        result = await self.channel.update_card(
            card_action_event.message_id,
            FINISH_CARD_CONTENT,
        )
        self._ensure_send_success(
            result,
            operation="update completed question card",
        )

    async def consume(
        self,
        session_id: str,
        idle_timeout: int | None = None,
    ) -> None:
        idle_timeout = settings.feishu.idle_timeout if idle_timeout is None else idle_timeout
        if session_id not in self.receive_queues:
            return

        queue = self.receive_queues[session_id]
        qsize = queue.qsize()
        if qsize == 0:
            return
        last_message: Message = queue._queue[qsize - 1]
        if time() - last_message.timestamp < idle_timeout:
            return

        messages = [queue.get_nowait() for _ in range(qsize)]
        chat_id = messages[-1].chat_id
        reply_to = messages[-1].message_id
        sender_name = messages[-1].sender_name
        for message in messages:
            try:
                await self.add_reaction(message)
            except Exception:
                logger.debug(
                    "Failed to add Feishu reaction message_id={}",
                    message.message_id,
                )

        try:
            async for event in iter_chat_events(
                messages,
                str(URL(self.chat_url) / "chat"),
            ):
                if event.event == ChatEventType.SEND_FILE:
                    downloaded_file = await download_channel_file(
                        session_id,
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
                        session_id=session_id,
                        sender_name=sender_name,
                        message_type=event.event,
                        iter_content=iter_content,
                    )
        except httpx.HTTPStatusError as exc:
            await self._send_backend_error(
                exc,
                chat_id=chat_id,
                reply_to=reply_to,
                session_id=session_id,
                sender_name=sender_name,
            )
        except Exception:
            logger.exception(
                "Failed to consume Feishu messages session_id={}",
                session_id,
            )

    async def _send_backend_error(
        self,
        exc: httpx.HTTPStatusError,
        *,
        chat_id: str,
        reply_to: str | None,
        session_id: str,
        sender_name: str,
    ) -> None:
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
            session_id=session_id,
            sender_name=sender_name,
            message_type=ChatEventType.ERROR,
            iter_content=iter_content,
        )

    async def one_by_one_session_id(self) -> None:
        for session_id, task in list(self.session_tasks.items()):
            if not task.done():
                continue
            self.session_tasks.pop(session_id, None)
            self.session_update_message_id.pop(session_id, None)
            try:
                task.result()
            except Exception:
                logger.exception(
                    "Feishu session task failed session_id={}",
                    session_id,
                )

        for session_id in list(self.receive_queues):
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
            message={"card": build_feishu_markdown_card(background_message.content)},
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
        poll_task = asyncio.create_task(self.poll_sessions())
        background_message_task = asyncio.create_task(self.poll_background_messages())
        try:
            await self.channel.connect()
        finally:
            await self.channel.disconnect()
            poll_task.cancel()
            background_message_task.cancel()
            for task in self.session_tasks.values():
                task.cancel()
            await asyncio.gather(
                poll_task,
                background_message_task,
                *self.session_tasks.values(),
                return_exceptions=True,
            )


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
