
async def iter_sse_payloads(
    cls,
    response: aiohttp.ClientResponse,
) -> AsyncIterator[dict[str, Any]]:
    buffer = ""
    async for chunk in response.content.iter_any():
        buffer += chunk.decode("utf-8", errors="ignore")
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            payload = cls._parse_sse_block(block)
            if payload is not None:
                yield payload

    payload = cls._parse_sse_block(buffer)
    if payload is not None:
        yield payload


def parse_sse_block(block: str) -> dict[str, Any] | None:
    data_lines = [
        line.removeprefix("data:").strip()
        for line in block.splitlines()
        if line.startswith("data:")
    ]
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