from __future__ import annotations

import asyncio

import httpx
from loguru import logger
from yarl import URL

from ....agent.server.schema import FeishuBackgroundMessage
from ....utils.channel_jwt import ChannelTokenParams, create_token
from .cards import build_feishu_background_task_card
from .channel import XpeechFeishuChannel
from .delivery import ensure_send_success

BACKGROUND_MESSAGE_RETRY_SECONDS = 2.0


class FeishuBackgroundMessages:
    """长轮询后台任务结果并投递到飞书。"""

    def __init__(self, chat_url: str, channel: XpeechFeishuChannel):
        self.chat_url = chat_url
        self.channel = channel

    async def poll_background_message_once(
        self,
        client: httpx.AsyncClient,
    ) -> bool:
        """Poll once and immediately deliver an available background message."""
        response = await client.get(
            str(URL(self.chat_url) / "background_message"),
            headers={"authorization": (f"Bearer {create_token(ChannelTokenParams('feishu'))}")},
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
        ensure_send_success(result, operation="send background message")
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
                except Exception:  # noqa: BLE001 - 单次轮询失败后继续服务，取消单独向上传播。
                    logger.exception("Failed to poll or send Feishu background message")
                    await asyncio.sleep(BACKGROUND_MESSAGE_RETRY_SECONDS)
