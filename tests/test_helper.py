import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from starlette.datastructures import UploadFile

from xpeech.utils import file_io, helper
from xpeech.utils.file_io import LockedFile
from xpeech.utils.helper import compress_image


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
    await LockedFile(text_path).write_text("first")
    await LockedFile(text_path).append_text(" second")
    assert await LockedFile(text_path).read_text() == "first second"

    await LockedFile(text_path).write_text("café", encoding="latin-1")
    assert await LockedFile(text_path).read_text(encoding="latin-1") == "café"

    binary_path = tmp_path / "content.bin"
    await LockedFile(binary_path).write_bytes(b"\x00\x01\xff")
    assert await LockedFile(binary_path).read_bytes() == b"\x00\x01\xff"


@pytest.mark.asyncio
async def test_concurrent_same_name_uploads_preserve_all_contents(tmp_path: Path):
    directory = tmp_path / "attachments" / helper.format_now()
    directory.mkdir(parents=True)
    existing = directory / "report.csv"
    existing.write_bytes(b"existing")
    payloads = [f"upload-{index}".encode() * 1000 for index in range(12)]

    paths = await asyncio.gather(*(
        file_io.save_to_workspace(
            UploadFile(filename="report.csv", file=BytesIO(payload)), tmp_path,
        )
        for payload in payloads
    ))

    assert len(set(paths)) == len(payloads)
    assert {path.name for path in paths} == {f"report_{index}.csv" for index in range(1, 13)}
    assert [path.read_bytes() for path in paths] == payloads
    assert existing.read_bytes() == b"existing"
    assert len(list(directory.iterdir())) == 13


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_incomplete_uploads_are_not_published_and_temporary_is_cleaned(tmp_path, cancel):
    started = asyncio.Event()

    class InterruptedUpload:
        filename = "report.csv"
        reads = 0

        async def read(self, _size):
            self.reads += 1
            if self.reads == 1:
                return b"partial content"
            started.set()
            if cancel:
                await asyncio.Event().wait()
            raise RuntimeError("upload failed")

    task = asyncio.create_task(file_io.save_to_workspace(InterruptedUpload(), tmp_path))
    await asyncio.wait_for(started.wait(), 5)
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        await task

    directory = tmp_path / "attachments" / helper.format_now()
    assert list(directory.iterdir()) == []
