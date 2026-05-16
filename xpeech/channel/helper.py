from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator
import json
import httpx
import base64
from loguru import logger
from pydantic import ValidationError
from ..utils.helper import detect_image_mime
from io import BytesIO
from PIL import Image
from .schema import ChatEvent, FileData, ImageData, Message, TextData


def compress_image_bytes_to_jpg(
    input_bytes: bytes,
    target_kb: int = 500,
    min_quality: int = 10,
    max_quality: int = 95,
) -> bytes:
    target_bytes = target_kb * 1024
    img = Image.open(BytesIO(input_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    low, high = min_quality, max_quality
    best_data = None
    while low <= high:
        quality = (low + high) // 2
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        data = buffer.getvalue()
        if len(data) <= target_bytes:
            best_data = data
            low = quality + 1
        else:
            high = quality - 1
    if best_data is None:
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=min_quality, optimize=True)
        best_data = buffer.getvalue()
    return best_data


async def iter_sse_payloads(
    response: httpx.Response,
) -> AsyncIterator[ChatEvent]:
    buffer = ""
    async for chunk in response.aiter_bytes():
        buffer += chunk.decode("utf-8", errors="ignore")
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            payload = _parse_sse_block(block)
            if payload is not None:
                yield payload
    payload = _parse_sse_block(buffer)
    if payload is not None:
        yield payload


def _parse_sse_block(block: str) -> ChatEvent | None:
    data_lines = [line.removeprefix("data:").strip() for line in block.splitlines() if line.startswith("data:")]
    if not data_lines:
        return None
    raw = "\n".join(data_lines)
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("Skipping malformed SSE payload: {}", raw[:200])
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return ChatEvent.model_validate(payload)
    except ValidationError:
        logger.debug("Skipping invalid SSE event payload: {}", payload)
        return None


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
            elif isinstance(item, ImageData):
                content.append({"image_url": item.image_url})
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
            async with client.stream(
                "POST",
                chat_url,
                headers=headers,
                data=data,
                files=upload_files,
            ) as response:
                response.raise_for_status()
                async for payload in iter_sse_payloads(response):
                    yield payload


def bytes_to_image_url(raw: bytes) -> str:
    raw = compress_image_bytes_to_jpg(raw)
    mime = detect_image_mime(raw)
    b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"
