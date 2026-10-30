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
from yarl import URL

from ...utils.channel_jwt import ChannelJWTAuth, create_channel_access_token
from .schema import ChatEvent, ChatEventType, FileData, Message, TextData


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


def _backend_headers(
    user_id: str,
    auth: ChannelJWTAuth,
    sender_name: str | None = None,
) -> dict[str, str]:
    headers = {
        "authorization": f"Bearer {create_channel_access_token(auth)}",
        "x-user-id": user_id,
    }
    if sender_name is not None:
        headers["x-sender-name"] = quote(sender_name, safe="")
    return headers


async def open_chat_stream(
    user_id: str,
    sender_name: str,
    content: str,
    session_metadata: str,
    chat_url: str,
    auth: ChannelJWTAuth,
    *,
    timestamp: str | None = None,
    files: Any = None,
    json_output: bool = False,
) -> ChatStream:
    """Open a streaming request to the backend chat endpoint."""

    data = {
        "session_metadata": session_metadata,
        "content": content,
    }
    if timestamp is not None:
        data["timestamp"] = timestamp
    if json_output:
        data["json_output"] = "true"

    client = httpx.AsyncClient(timeout=None)
    try:
        headers = _backend_headers(user_id, auth, sender_name)
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
    auth: ChannelJWTAuth,
) -> AsyncIterator[ChatEvent]:
    """Send channel messages to the chat endpoint and yield parsed SSE event dicts."""

    if not messages:
        return

    user_id = messages[0].user_id
    if any(message.user_id != user_id for message in messages):
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
            user_id,
            sender_name,
            json.dumps(content, ensure_ascii=False),
            json.dumps(
                {key: str(value) for key, value in session_metadata.items()},
                ensure_ascii=False,
            ),
            chat_url,
            auth,
            files=upload_files,
        )
        try:
            if not stream.response.is_success:
                await stream.response.aread()
            stream.response.raise_for_status()
            event_source = EventSource(stream.response)
            sse_events = event_source.aiter_sse()
            async for sse_event in sse_events:
                payload = json.loads(sse_event.data)
                event_type = payload["event"]
                if event_type in {ChatEventType.ASSISTANT, ChatEventType.THINKING}:
                    yield ChatEvent(
                        event=event_type,
                        context=_iter_text_chunks(payload["context"], f"{event_type}_end", sse_events),
                    )
                else:
                    yield ChatEvent.model_validate(payload)
        finally:
            await stream.aclose()


async def _iter_text_chunks(
    first_chunk: str,
    end_event: str,
    sse_events: AsyncIterator[Any],
) -> AsyncIterator[str]:
    """Yield text chunks until the matching stream-end event is received."""

    yield first_chunk

    async for sse_event in sse_events:
        payload = json.loads(sse_event.data)
        if payload["event"] == end_event:
            return
        event = ChatEvent.model_validate(payload)
        yield event.context


async def submit_question(
    user_id: str,
    result: Any,
    chat_url: str,
    auth: ChannelJWTAuth,
) -> httpx.Response:
    """Submit a question answer and return the backend response without changing its status."""

    answer_url = str(URL(chat_url) / "answer_question")
    answer = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    async with httpx.AsyncClient(timeout=30) as client:
        return await client.post(
            answer_url,
            headers=_backend_headers(user_id, auth),
            data={"answer": answer},
        )


async def notify_question(
    user_id: str,
    result: Any,
    chat_url: str,
    auth: ChannelJWTAuth,
) -> None:
    response = await submit_question(user_id, result, chat_url, auth)
    response.raise_for_status()


async def fetch_file(
    user_id: str,
    path: str,
    api_base_url: str,
    auth: ChannelJWTAuth,
) -> httpx.Response:
    """Fetch a session file and return the backend response without changing its status."""

    file_url = str(URL(api_base_url) / "sessions" / user_id / "files")
    async with httpx.AsyncClient(timeout=None) as client:
        return await client.get(
            file_url,
            headers=_backend_headers(user_id, auth),
            params={"path": path},
        )


async def download_file(
    user_id: str,
    path: str,
    api_base_url: str,
    auth: ChannelJWTAuth,
) -> DownloadedFile:
    response = await fetch_file(user_id, path, api_base_url, auth)
    response.raise_for_status()

    filename = _filename_from_content_disposition(response.headers.get("content-disposition")) or Path(path).name
    return DownloadedFile(filename=filename, content=response.content)


def _filename_from_content_disposition(value: str | None) -> str | None:
    if not value:
        return None

    message = EmailMessage()
    message["content-disposition"] = value
    return message.get_filename()
