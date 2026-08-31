from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from xpeech.utils.helper import (
    append_text_async,
    compress_image,
    read_bytes_async,
    read_text_async,
    write_bytes_async,
    write_text_async,
)


def test_compress_image_returns_raw_and_mime_type():
    input_buffer = BytesIO()
    Image.new("RGB", (2, 2), color="red").save(input_buffer, format="PNG")

    raw, mime = compress_image(input_buffer.getvalue())

    assert mime == "image/jpeg"
    with Image.open(BytesIO(raw)) as image:
        assert image.format == "JPEG"


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
