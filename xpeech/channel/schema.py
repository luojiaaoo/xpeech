from collections.abc import AsyncIterator
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ChatEventType(StrEnum):
    ASSISTANT = "assistant"
    THINKING = "thinking"
    ERROR = "error"
    TOOL_CALL = "tool_call"
    TOOL_CALL_RESULT = "tool_call_result"
    COMMAND = "command"
    SEND_FILE = "send_file"
    QUESTION = "question"
    TOKEN_USAGE = "token_usage"


class TextData(BaseModel):
    text: str


class FileData(BaseModel):
    file: Path


class Message(BaseModel):
    """Message to be sent to Feishu."""

    message_id: str  # Unique identifier for the message.
    chat_id: str  # Unique identifier for the chat.
    session_id: str  # Unique identifier for the session.
    sender_name: str  # Username of the message sender.
    content: list[TextData | FileData]  # List of messages.
    timestamp: int  # Unix timestamp of the message, in seconds.
    session_metadata: dict[str, str | int]  # Metadata for the session.


class ChatEvent(BaseModel):
    """Event payload yielded by AgentLoop SSE responses."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    event: ChatEventType
    context: str | AsyncIterator[str]
