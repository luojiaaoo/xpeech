from typing import Literal, TypeAlias
from pydantic import BaseModel
from pathlib import Path


ChatEventType: TypeAlias = Literal["assistant", "thinking", "tool_call", "tool_call_result", "command"]


class TextData(BaseModel):
    text: str


class ImageData(BaseModel):
    image_url: str


class FileData(BaseModel):
    file: Path


class Message(BaseModel):
    """Message to be sent to Feishu."""

    message_id: str  # Unique identifier for the message.
    session_id: str  # Unique identifier for the session.
    content: list[TextData | ImageData | FileData]  # List of messages.
    timestamp: int  # Timestamp of the message.
    session_metadata: dict[str, str | int]  # Metadata for the session.


class ChatEvent(BaseModel):
    """Event payload yielded by AgentLoop SSE responses."""

    event: ChatEventType
    context: str
