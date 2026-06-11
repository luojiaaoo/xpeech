from contextlib import ExitStack
from dataclasses import dataclass
from email.message import Message as EmailMessage
from pathlib import Path
from typing import AsyncIterator
import json
import httpx
from httpx_sse import aconnect_sse
from loguru import logger
from pydantic import ValidationError
from .schema import ChatEvent, FileData, Message, TextData
from typing import Any
from yarl import URL


@dataclass(frozen=True)
class DownloadedFile:
    filename: str
    content: bytes


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

    data = {
        "session_metadata": json.dumps(
            {key: str(value) for key, value in session_metadata.items()},
            ensure_ascii=False,
        ),
        "content": json.dumps(content, ensure_ascii=False),
    }
    headers = {"x-session-id": session_id}

    with ExitStack() as stack:
        upload_files = [
            ("files", (file_path.name, stack.enter_context(file_path.open("rb")), "application/octet-stream"))
            for file_path in files
        ]
        async with httpx.AsyncClient(timeout=None) as client:
            async with aconnect_sse(
                client,
                "POST",
                chat_url,
                headers=headers,
                data=data,
                files=upload_files,
            ) as event_source:
                event_source.response.raise_for_status()
                async for event in event_source.aiter_sse():
                    try:
                        yield ChatEvent.model_validate_json(event.data)
                    except ValidationError:
                        logger.exception("Skipping invalid SSE event payload: {}", event.data[:200])


async def notify_question(session_id: str, result: Any, chat_url: str) -> None:
    answer_url = str(URL(chat_url) / "answer_question")
    answer = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            answer_url,
            headers={"x-session-id": session_id},
            data={"answer": answer},
        )
        response.raise_for_status()


async def download_file(session_id: str, path: str, api_base_url: str) -> DownloadedFile:
    file_url = str(URL(api_base_url) / "sessions" / session_id / "files")
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.get(file_url, params={"path": path})
        response.raise_for_status()

    filename = _filename_from_content_disposition(response.headers.get("content-disposition")) or Path(path).name
    return DownloadedFile(filename=filename, content=response.content)


def _filename_from_content_disposition(value: str | None) -> str | None:
    if not value:
        return None

    message = EmailMessage()
    message["content-disposition"] = value
    return message.get_filename()
