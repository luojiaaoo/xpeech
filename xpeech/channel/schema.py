from pydantic import BaseModel
from pathlib import Path


class TextMessage(BaseModel):
    text: str


class ImageMessage(BaseModel):
    image_url: str


class FileMessage(BaseModel):
    file: Path


class Messages(BaseModel):
    """Messages to be sent to Feishu."""

    message_id: str  # Unique identifier for the message.
    session_id: str  # Unique identifier for the session.
    content: list[TextMessage | ImageMessage | FileMessage]  # List of messages.
    timestamp: int  # Timestamp of the message.
    session_metadata: dict[str, str]  # Metadata for the session.
