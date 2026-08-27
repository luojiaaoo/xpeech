from pathlib import Path

import pytest

from xpeech.utils.helper import (
    append_text_async,
    read_bytes_async,
    read_text_async,
    write_bytes_async,
    write_text_async,
)


@pytest.mark.asyncio
async def test_async_file_read_write_helpers(tmp_path: Path):
    text_path = tmp_path / "content.txt"
    await write_text_async(text_path, "first")
    await append_text_async(text_path, " second")
    assert await read_text_async(text_path) == "first second"

    await write_text_async(text_path, "café", encoding="latin-1")
    assert await read_text_async(text_path, encoding="latin-1") == "café"

    binary_path = tmp_path / "content.bin"
    await write_bytes_async(binary_path, b"\x00\x01\xff")
    assert await read_bytes_async(binary_path) == b"\x00\x01\xff"
