"""Adapt backend chat events for Feishu delivery."""

import json
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from typing import Any

import anyio

from ....config.settings import settings
from ....utils.helper import stack_ensure_aclosing
from ..helper import BackendUserParams, open_chat_stream
from ..schema import ChatEvent, ChatEventType, FileData, Message, TextData


async def iter_chat_events(
    messages: list[Message],
    chat_url: str,
    *,
    session_id: str,
) -> AsyncGenerator[ChatEvent, None]:
    """Send channel messages and adapt the response for Feishu delivery."""

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

    await settings.extra_user_info.merge_session_metadata(user_id, session_metadata)
    async with stack_ensure_aclosing() as stack:
        upload_files = []
        for file_path in files:
            file = await stack.enter_async_context(await anyio.open_file(file_path, "rb"))
            upload_files.append(("files", (file_path.name, file.wrapped, "application/octet-stream")))
        async with open_chat_stream(
            BackendUserParams(user_id, "feishu"),
            sender_name,
            json.dumps(content, ensure_ascii=False),
            json.dumps(
                {key: str(value) for key, value in session_metadata.items()},
                ensure_ascii=False,
            ),
            chat_url,
            session_id=session_id,
            files=upload_files,
        ) as (response, payloads):
            if not response.is_success:
                await response.aread()
            response.raise_for_status()
            async for payload in payloads:
                event_type = payload["event"]
                if event_type == "user":
                    continue
                if event_type in {ChatEventType.ASSISTANT, ChatEventType.THINKING}:
                    yield ChatEvent(
                        event=event_type,
                        context=_iter_text_chunks(payload["context"], f"{event_type}_end", payloads),
                    )
                else:
                    yield ChatEvent.model_validate(payload)


async def _iter_text_chunks(
    first_chunk: str,
    end_event: str,
    payloads: AsyncIterator[dict[str, Any]],
) -> AsyncIterator[str]:
    """Yield text chunks until the matching stream-end event is received."""

    yield first_chunk

    async for payload in payloads:
        if payload["event"] == end_event:
            return
        event = ChatEvent.model_validate(payload)
        yield event.context
