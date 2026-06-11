from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable

from yarl import URL

from ..helper import download_file as download_channel_file
from ..helper import iter_chat_events, notify_question
from ..schema import ChatEvent, FileData, Message, TextData
from .identity import DesktopIdentity


ChatEventHandler = Callable[[dict[str, Any]], None]


class XpeechDesktopClient:
    """Python-side client for the existing Xpeech HTTP/SSE API."""

    def __init__(self, api_base_url: str):
        self.api_base_url = api_base_url.rstrip("/")

    async def send_message(
        self,
        identity: DesktopIdentity,
        content: str,
        files: list[Path],
        on_event: ChatEventHandler,
    ) -> None:
        message = self._build_message(identity=identity, content=content, files=files)
        chat_url = str(URL(self.api_base_url) / "chat")

        async for event in iter_chat_events([message], chat_url):
            on_event(event.model_dump())

    async def answer_question(self, session_id: str, answer: Any) -> None:
        await notify_question(session_id, answer, self.api_base_url)

    async def download_file(self, session_id: str, remote_path: str, save_path: Path) -> None:
        downloaded_file = await download_channel_file(session_id, remote_path, self.api_base_url)
        save_path.write_bytes(downloaded_file.content)

    def _build_message(self, identity: DesktopIdentity, content: str, files: list[Path]) -> Message:
        message_content: list[TextData | FileData] = []

        if content.strip():
            message_content.append(TextData(text=content))

        for file_path in files:
            if file_path.exists() and file_path.is_file():
                message_content.append(FileData(file=file_path))

        if not message_content:
            raise ValueError("Message content is empty")

        return Message(
            message_id=f"desktop_{int(time.time() * 1000)}",
            chat_id=identity.session_id,
            session_id=identity.session_id,
            content=message_content,
            timestamp=int(time.time()),
            session_metadata={"channel": "desktop", "sender_name": identity.username},
        )


def serialize_event_for_js(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False)
