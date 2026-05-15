from typing import Any, AsyncIterator
import json
import httpx
import base64
from loguru import logger
from ..utils.helper import detect_image_mime


async def iter_sse_payloads(
    response: httpx.Response,
) -> AsyncIterator[dict[str, Any]]:
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


def _parse_sse_block(block: str) -> dict[str, Any] | None:
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
    return payload if isinstance(payload, dict) else None


def bytes_to_image_url(raw: bytes) -> str:
    mime = detect_image_mime(raw)
    b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"
