from pydantic import BaseModel, Field, field_validator
from typing import Annotated
from model.enum import ChannelType


class TextMessage(BaseModel):
    """Text message."""
    text: str

class ImageMessage(BaseModel):
    """Image message."""
    image_url: str



class InboundMessage(BaseModel):
    """Message received from a chat channel."""

    channel: ChannelType  # 通道
    sender_id: str  # 发送人
    group_id: Annotated[str | None, Field(default=None)]  # TODO: 支持群聊
    chat_id: str  # 聊天编号
    content: str  # 文字内容
    timestamp: datetime = field(default_factory=datetime.now)
    media: list[str] = field(default_factory=list)  # Media URLs
    metadata: dict[str, Any] = field(default_factory=dict)  # Channel-specific data
    session_key_override: str | None = (
        None  # Optional override for thread-scoped sessions
    )

    @property
    def session_key(self) -> str:
        """Unique key for session identification."""
        return self.session_key_override or f"{self.channel}:{self.chat_id}"
