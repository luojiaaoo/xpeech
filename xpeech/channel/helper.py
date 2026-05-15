from typing import Any, AsyncIterator
import json
import httpx
import base64
from loguru import logger
from ..utils.helper import detect_image_mime
from io import BytesIO
from PIL import Image


def compress_image_bytes_to_jpg(
    input_bytes: bytes,
    target_kb: int = 200,
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
    raw = compress_image_bytes_to_jpg(raw)
    mime = detect_image_mime(raw)
    b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"
