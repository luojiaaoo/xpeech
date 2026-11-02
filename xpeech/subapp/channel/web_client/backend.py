"""Web session forwarding on top of the shared user JWT header contract."""

import json
from typing import Any

import httpx
from yarl import URL

from ....utils.ids import DEFAULT_SESSION_ID
from ..helper import BackendAuthParams, ChatStream, _backend_headers


async def open_chat_stream(
    auth: BackendAuthParams,
    sender_name: str,
    content: str,
    session_metadata: str,
    chat_url: str,
    *,
    files: Any = None,
    json_output: bool = False,
    session_id: str | None = None,
) -> ChatStream:
    data = {"session_metadata": session_metadata, "content": content}
    if json_output:
        data["json_output"] = "true"
    headers = _backend_headers(auth, sender_name)
    headers.update({
        "x-session-id": session_id or DEFAULT_SESSION_ID,
        "Accept": "text/event-stream",
        "Cache-Control": "no-store",
    })
    client = httpx.AsyncClient(timeout=None)
    try:
        response = await client.send(
            client.build_request("POST", chat_url, headers=headers, data=data, files=files),
            stream=True,
        )
    except BaseException:
        await client.aclose()
        raise
    return ChatStream(response=response, _client=client)


async def submit_question(
    auth: BackendAuthParams,
    result: Any,
    chat_url: str,
    *,
    session_id: str | None = None,
) -> httpx.Response:
    headers = _backend_headers(auth)
    headers["x-session-id"] = session_id or DEFAULT_SESSION_ID
    answer = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    async with httpx.AsyncClient(timeout=30) as client:
        return await client.post(
            str(URL(chat_url) / "answer_question"), headers=headers, data={"answer": answer},
        )
