import json
from collections.abc import AsyncIterator
from contextlib import ExitStack
from dataclasses import dataclass
from email.message import Message as EmailMessage
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from httpx_sse import EventSource
from loguru import logger
from pydantic import ValidationError
from yarl import URL

from ..utils.jwt_auth import create_access_token
from .schema import ChatEvent, FileData, Message, TextData


@dataclass(frozen=True)
class DownloadedFile:
    filename: str
    content: bytes


@dataclass(frozen=True)
class ChatStream:
    """An open response from the backend chat endpoint."""

    response: httpx.Response
    _client: httpx.AsyncClient

    async def aclose(self) -> None:
        await self.response.aclose()
        await self._client.aclose()


def _backend_headers(session_id: str, sender_name: str | None = None) -> dict[str, str]:
    headers = {
        "authorization": f"Bearer {create_access_token()}",
        "x-session-id": session_id,
    }
    if sender_name is not None:
        headers["sender-name"] = quote(sender_name, safe="")
    return headers


async def open_chat_stream(
    session_id: str,
    sender_name: str,
    content: str,
    session_metadata: str,
    chat_url: str,
    *,
    timestamp: str | None = None,
    files: Any = None,
) -> ChatStream:
    """Open a streaming request to the backend chat endpoint."""

    data = {
        "session_metadata": session_metadata,
        "content": content,
    }
    if timestamp is not None:
        data["timestamp"] = timestamp

    client = httpx.AsyncClient(timeout=None)
    try:
        headers = _backend_headers(session_id, sender_name)
        headers.update({"Accept": "text/event-stream", "Cache-Control": "no-store"})
        response = await client.send(
            client.build_request(
                "POST",
                chat_url,
                headers=headers,
                data=data,
                files=files,
            ),
            stream=True,
        )
    except BaseException:
        await client.aclose()
        raise
    return ChatStream(response=response, _client=client)


async def iter_chat_events(
    messages: list[Message],
    chat_url: str,
) -> AsyncIterator[ChatEvent]:
    """Send channel messages to the chat endpoint and yield parsed SSE event dicts."""

    if not messages:
        return

    session_id = messages[0].session_id
    if any(message.session_id != session_id for message in messages):
        raise ValueError("All messages must belong to the same session.")
    sender_name = messages[0].sender_name
    if any(message.sender_name != sender_name for message in messages):
        raise ValueError("All messages must belong to the same sender.")

    content: list[dict[str, str]] = []
    files: list[Path] = []
    session_metadata: dict[str, str | int] = {}

    for message in messages:
        session_metadata.update(message.session_metadata)
        for item in message.content:
            if isinstance(item, TextData):
                content.append({"text": item.text})
            elif isinstance(item, FileData):
                files.append(item.file)
            else:
                raise TypeError(f"Unsupported message content type: {type(item)!r}")

    with ExitStack() as stack:
        upload_files = [
            ("files", (file_path.name, stack.enter_context(file_path.open("rb")), "application/octet-stream"))
            for file_path in files
        ]
        stream = await open_chat_stream(
            session_id,
            sender_name,
            json.dumps(content, ensure_ascii=False),
            json.dumps(
                {key: str(value) for key, value in session_metadata.items()},
                ensure_ascii=False,
            ),
            chat_url,
            files=upload_files,
        )
        try:
            stream.response.raise_for_status()
            event_source = EventSource(stream.response)
            async for event in event_source.aiter_sse():
                try:
                    yield ChatEvent.model_validate_json(event.data)
                except ValidationError:
                    logger.exception("Skipping invalid SSE event payload: {}", event.data[:200])
        finally:
            await stream.aclose()


async def submit_question(session_id: str, result: Any, chat_url: str) -> httpx.Response:
    """Submit a question answer and return the backend response without changing its status."""

    answer_url = str(URL(chat_url) / "answer_question")
    answer = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    async with httpx.AsyncClient(timeout=30) as client:
        return await client.post(
            answer_url,
            headers=_backend_headers(session_id),
            data={"answer": answer},
        )


async def notify_question(session_id: str, result: Any, chat_url: str) -> None:
    response = await submit_question(session_id, result, chat_url)
    response.raise_for_status()


async def fetch_file(session_id: str, path: str, api_base_url: str) -> httpx.Response:
    """Fetch a session file and return the backend response without changing its status."""

    file_url = str(URL(api_base_url) / "sessions" / session_id / "files")
    async with httpx.AsyncClient(timeout=None) as client:
        return await client.get(
            file_url,
            headers=_backend_headers(session_id),
            params={"path": path},
        )


async def download_file(session_id: str, path: str, api_base_url: str) -> DownloadedFile:
    response = await fetch_file(session_id, path, api_base_url)
    response.raise_for_status()

    filename = _filename_from_content_disposition(response.headers.get("content-disposition")) or Path(path).name
    return DownloadedFile(filename=filename, content=response.content)


def _filename_from_content_disposition(value: str | None) -> str | None:
    if not value:
        return None

    message = EmailMessage()
    message["content-disposition"] = value
    return message.get_filename()
