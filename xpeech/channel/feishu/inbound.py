from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Any

from async_lru import alru_cache
from lark_channel import (
    FileContent,
    ImageContent,
    InboundMessage,
    MediaContent,
    PostContent,
    TextContent,
)
from lark_channel.api.contact.v3.model.get_user_request import GetUserRequest
from loguru import logger

from ..schema import FileData, Message, TextData
from .config import FEISHU_CACHE_DIR, FEISHU_USER_CACHE_TTL_SECONDS


class UnsupportedFeishuMessageError(ValueError):
    """表示当前适配器不支持该飞书消息。"""


class FeishuInboundMixin:
    """解析发送者身份，并将飞书入站消息转换为统一格式。"""

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
        email = getattr(user, "email", None) or getattr(user, "enterprise_email", None) if user else None
        if not email:
            logger.warning(
                "Feishu user has no readable email; grant the user email field permission: open_id={}",
                open_id,
            )
        return employee_no, email

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
            session_metadata = {"open_id": inbound_msg.sender_id}
            if email:
                session_metadata["email"] = email
            common_fields = {
                "message_id": inbound_msg.message_id,
                "chat_id": inbound_msg.chat_id,
                "session_id": session_id,
                "sender_name": inbound_msg.sender_name,
                "timestamp": timestamp,
                "session_metadata": session_metadata,
            }

            if isinstance(inbound_msg.content, TextContent):
                return Message(
                    **common_fields,
                    content=[TextData(text=inbound_msg.content.text)],
                )

            if isinstance(inbound_msg.content, ImageContent):
                saved_file = await _save_resource(
                    inbound_msg.message_id,
                    inbound_msg.content.image_key,
                    "image",
                    resource_dest_dir,
                )
                return Message(
                    **common_fields,
                    content=[FileData(file=saved_file)],
                )

            if isinstance(inbound_msg.content, (FileContent, MediaContent)):
                resource_type = "video" if isinstance(inbound_msg.content, MediaContent) else "file"
                saved_file = await _save_resource(
                    inbound_msg.message_id,
                    inbound_msg.content.file_key,
                    resource_type,
                    resource_dest_dir,
                    inbound_msg.content.file_name,
                )
                return Message(
                    **common_fields,
                    content=[FileData(file=saved_file)],
                )

            if isinstance(inbound_msg.content, PostContent):
                parsed_content: list[TextData | FileData] = []
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
                    if tag == "a":
                        text_buffer.append(f"[{item['text']}]({item['href']})")
                        continue

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
                return Message(**common_fields, content=parsed_content)

        raise UnsupportedFeishuMessageError(
            f"Unsupported Feishu message: chat_type={inbound_msg.chat_type}, "
            f"content_type={type(inbound_msg.content).__name__}, "
            f"message_id={inbound_msg.message_id}"
        )
