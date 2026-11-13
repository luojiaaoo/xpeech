import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from email.message import Message as EmailMessage
from pathlib import Path
from typing import Any
from urllib.parse import quote

import anyio
import httpx
from asyncer import asyncify
from httpx_sse import EventSource
from yarl import URL

from ...utils.channel_jwt import UserTokenParams, create_token
from ...utils.helper import stack_ensure_aclosing
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
        with anyio.CancelScope(shield=True):
            try:
                await self.response.aclose()
            finally:
                await self._client.aclose()


@dataclass(frozen=True)
class BackendUserParams:
    user_id: str
    channel: str


@dataclass(frozen=True)
class BackendJWTParams:
    jwt_token: str


BackendAuthParams = BackendUserParams | BackendJWTParams


class _AsyncMultipartStream(httpx.AsyncByteStream):
    """Keep HTTPX's synchronous multipart file reads off the event loop."""

    def __init__(self, stream: httpx.SyncByteStream):
        self._stream = stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        iterator = iter(self._stream)
        while (chunk := await asyncify(next)(iterator, None)) is not None:
            yield chunk

    async def aclose(self) -> None:
        await asyncify(self._stream.close)()


def _backend_headers(
    auth: BackendAuthParams,
    sender_name: str | None = None,
) -> dict[str, str]:
    """Sign for a user identity or pass through the supplied JWT."""
    if isinstance(auth, BackendJWTParams):
        return _passthrough_headers(auth.jwt_token, sender_name)
    token = create_token(UserTokenParams(auth.channel, user_id=auth.user_id))
    return _passthrough_headers(token, sender_name)


def _passthrough_headers(jwt_token: str, sender_name: str | None = None) -> dict[str, str]:
    """Build headers using the original JWT without issuing a new token."""
    headers = {
        "authorization": f"Bearer {jwt_token}",
    }
    if sender_name is not None:
        headers["x-sender-name"] = quote(sender_name, safe="")
    return headers


async def proxy_get(
    auth: BackendAuthParams,
    url: str,
    *,
    params: list[tuple[str, str]],
) -> httpx.Response:
    """Send a buffered GET and return the backend response without changing its status."""
    async with httpx.AsyncClient(timeout=30) as client:
        return await client.get(url, headers=_backend_headers(auth), params=params)


async def open_chat_stream(
    user_id: str,
    sender_name: str,
    content: str,
    session_metadata: str,
    chat_url: str,
    channel: str,
    *,
    session_id: str,
    files: Any = None,
) -> ChatStream:
    """Open a streaming request to the backend chat endpoint."""

    data = {
        "session_metadata": session_metadata,
        "content": content,
    }
    client = httpx.AsyncClient(timeout=None)
    try:
        headers = _backend_headers(BackendUserParams(user_id, channel), sender_name)
        headers.update({"x-session-id": session_id, "Accept": "text/event-stream", "Cache-Control": "no-store"})
        request = client.build_request(
            "POST", chat_url, headers=headers, data=data, files=files,
        )
        if files:
            # https://github.com/encode/httpx/pull/3698
            request.stream = _AsyncMultipartStream(request.stream)
        response = await client.send(request, stream=True)
    except BaseException:
        await client.aclose()
        raise
    return ChatStream(response=response, _client=client)


async def iter_chat_events(
    messages: list[Message],
    chat_url: str,
    channel: str,
    *,
    session_id: str,
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

    async with stack_ensure_aclosing() as stack:
        upload_files = []
        for file_path in files:
            file = await stack.enter_async_context(await anyio.open_file(file_path, "rb"))
            upload_files.append(("files", (file_path.name, file.wrapped, "application/octet-stream")))
        stream = await open_chat_stream(
            user_id,
            sender_name,
            json.dumps(content, ensure_ascii=False),
            json.dumps(
                {key: str(value) for key, value in session_metadata.items()},
                ensure_ascii=False,
            ),
            chat_url,
            channel,
            session_id=session_id,
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
    channel: str,
    *,
    session_id: str,
) -> httpx.Response:
    """Submit a question answer and return the backend response without changing its status."""

    answer_url = str(URL(chat_url) / "answer_question")
    answer = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    async with httpx.AsyncClient(timeout=30) as client:
        return await client.post(
            answer_url,
            headers={**_backend_headers(BackendUserParams(user_id, channel)), "x-session-id": session_id},
            data={"answer": answer},
        )


async def notify_question(
    user_id: str,
    result: Any,
    chat_url: str,
    channel: str,
    *,
    session_id: str,
) -> None:
    response = await submit_question(user_id, result, chat_url, channel, session_id=session_id)
    response.raise_for_status()


async def fetch_file(
    auth: BackendAuthParams,
    path: str,
    api_base_url: str,
) -> httpx.Response:
    """Fetch a session file and return the backend response without changing its status."""

    file_url = str(URL(api_base_url) / "files")
    async with httpx.AsyncClient(timeout=None) as client:
        return await client.get(
            file_url,
            headers=_backend_headers(auth),
            params={"path": path},
        )


async def download_file(
    user_id: str,
    path: str,
    api_base_url: str,
    channel: str,
) -> DownloadedFile:
    response = await fetch_file(BackendUserParams(user_id, channel), path, api_base_url)
    response.raise_for_status()

    filename = _filename_from_content_disposition(response.headers.get("content-disposition")) or Path(path).name
    return DownloadedFile(filename=filename, content=response.content)


def _filename_from_content_disposition(value: str | None) -> str | None:
    if not value:
        return None

    message = EmailMessage()
    message["content-disposition"] = value
    return message.get_filename()
