import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AbstractAsyncContextManager, AsyncExitStack, aclosing, asynccontextmanager
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


@dataclass(frozen=True)
class DownloadedFile:
    filename: str
    content: bytes


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


def open_chat_stream(
    auth: BackendAuthParams,
    sender_name: str,
    content: str,
    session_metadata: str,
    chat_url: str,
    *,
    session_id: str,
    files: Any = None,
) -> AbstractAsyncContextManager[tuple[httpx.Response, AsyncIterator[dict[str, Any]]]]:
    """Prepare an authenticated request to the backend chat endpoint."""
    return sse_request(
        "POST", chat_url,
        headers={**_backend_headers(auth, sender_name), "x-session-id": session_id},
        data={"session_metadata": session_metadata, "content": content},
        files=files,
    )


@asynccontextmanager
async def sse_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    data: dict[str, str] | None = None,
    files: Any = None,
) -> AsyncIterator[tuple[httpx.Response, AsyncIterator[dict[str, Any]]]]:
    """Keep a streaming response and its JSON event iterator open for the caller."""
    resources = AsyncExitStack()
    client = httpx.AsyncClient(timeout=None)
    resources.push_async_callback(client.aclose)
    try:
        request = client.build_request(
            method, url, headers={**headers, "Accept": "text/event-stream", "Cache-Control": "no-store"},
            data=data, files=files,
        )
        if files:
            # https://github.com/encode/httpx/pull/3698
            request.stream = _AsyncMultipartStream(request.stream)
        response = await client.send(request, stream=True)
        resources.push_async_callback(response.aclose)
        events = _iter_sse_payloads(response)
        resources.push_async_callback(events.aclose)
        yield response, events
    finally:
        with anyio.CancelScope(shield=True):
            await resources.aclose()


async def _iter_sse_payloads(response: httpx.Response) -> AsyncGenerator[dict[str, Any], None]:
    """Decode backend SSE frames and retain the original event fields."""
    async with aclosing(EventSource(response).aiter_sse()) as events:
        async for event in events:
            if not event.data:
                continue
            try:
                payload = json.loads(event.data)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


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


async def stop_chat(
    user_id: str,
    chat_url: str,
    channel: str,
    *,
    session_id: str,
) -> httpx.Response:
    """Stop a session directly, independently of its interactive message stream."""
    async with httpx.AsyncClient(timeout=30) as client:
        return await client.post(
            str(URL(chat_url) / "chat" / "stop"),
            headers={**_backend_headers(BackendUserParams(user_id, channel)), "x-session-id": session_id},
        )


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
