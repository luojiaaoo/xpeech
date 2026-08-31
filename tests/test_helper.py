from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from xpeech.utils import helper
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
async def test_compress_video_returns_raw_and_mime_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    input_path = tmp_path / "input.mov"
    input_path.write_bytes(b"input")

    class FakeVideoClip:
        size = (640, 480)
        fps = 30
        audio = None

        def __init__(self, _path: str):
            pass

        def set_fps(self, _fps: int):
            return self

        def write_videofile(self, path: str, **_kwargs):
            Path(path).write_bytes(b"compressed-video")

        def close(self):
            pass

    monkeypatch.setattr(helper, "VideoFileClip", FakeVideoClip)

    raw, mime = await helper.compress_video(input_path, tmp_path / "output")

    assert raw == b"compressed-video"
    assert mime == "video/mp4"


@pytest.mark.asyncio
async def test_read_video_metadata_accepts_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")

    class FakeVideoClip:
        size = (1280, 720)
        duration = 12.5

        def __init__(self, path: str):
            assert path == str(input_path)

        def close(self):
            pass

    monkeypatch.setattr(helper, "VideoFileClip", FakeVideoClip)

    metadata = await helper.read_video_metadata(input_path)

    assert metadata == {"duration": 12.5, "width": 1280, "height": 720}


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
