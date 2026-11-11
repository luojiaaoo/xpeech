"""Web session forwarding on top of the shared user JWT header contract."""

import json
from typing import Any

import httpx
from yarl import URL

from ..helper import BackendAuthParams, ChatStream, _backend_headers


def _chat_form(content: str, session_metadata: str) -> dict[str, str]:
    return {"session_metadata": session_metadata, "content": content}


async def open_chat(
    auth: BackendAuthParams,
    sender_name: str,
    content: str,
    session_metadata: str,
    chat_url: str,
    *,
    files: Any = None,
    session_id: str,
) -> ChatStream:
    """Open the interactive /chat request as a text/event-stream."""
    headers = _backend_headers(auth, sender_name)
    headers.update({
        "x-session-id": session_id,
        "Accept": "text/event-stream",
        "Cache-Control": "no-store",
    })
    client = httpx.AsyncClient(timeout=None)
    try:
        response = await client.send(
            client.build_request(
                "POST", chat_url, headers=headers,
                data=_chat_form(content, session_metadata), files=files,
            ),
            stream=True,
        )
    except BaseException:
        await client.aclose()
        raise
    return ChatStream(response=response, _client=client)


async def open_chat_sync(
    auth: BackendAuthParams,
    sender_name: str,
    content: str,
    session_metadata: str,
    chat_url: str,
    *,
    files: Any = None,
) -> httpx.Response:
    """Call the non-interactive /chat_sync and wait for the full JSON event list.

    No ``x-session-id`` is sent: the backend generates a fresh session per run.
    """
    headers = _backend_headers(auth, sender_name)
    headers.update({
        "Accept": "application/json",
        "Cache-Control": "no-store",
    })
    async with httpx.AsyncClient(timeout=None) as client:
        return await client.post(
            chat_url,
            headers=headers,
            data=_chat_form(content, session_metadata),
            files=files,
        )


async def stop_chat(
    auth: BackendAuthParams,
    chat_url: str,
    *,
    session_id: str,
) -> httpx.Response:
    headers = _backend_headers(auth)
    headers["x-session-id"] = session_id
    async with httpx.AsyncClient(timeout=30) as client:
        return await client.post(str(URL(chat_url) / "chat" / "stop"), headers=headers)


async def submit_question(
    auth: BackendAuthParams,
    result: Any,
    chat_url: str,
    *,
    session_id: str,
) -> httpx.Response:
    headers = _backend_headers(auth)
    headers["x-session-id"] = session_id
    answer = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    async with httpx.AsyncClient(timeout=30) as client:
        return await client.post(
            str(URL(chat_url) / "answer_question"), headers=headers, data={"answer": answer},
        )
