import asyncio
from pathlib import Path

import pytest

from xpeech.agent.tool_executor import ToolExecutor
from xpeech.agent.tools import filesystem
from xpeech.agent.tools.filesystem import EditFileArgs, ReadFileArgs, WriteFileArgs, build_file_tools
from xpeech.agent.tools.helper import get_mime_type
from xpeech.provider.schema import ToolCallRequest


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"\x89PNG\r\n\x1a\ncontent", "image/png"),
        (b"\xff\xd8\xffcontent", "image/jpeg"),
        (b"GIF87acontent", "image/gif"),
        (b"GIF89acontent", "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBPcontent", "image/webp"),
    ],
)
@pytest.mark.asyncio
async def test_get_mime_type_detects_content_before_extension(tmp_path: Path, data: bytes, expected: str):
    file_path = tmp_path / "misleading.txt"
    file_path.write_bytes(data)

    assert await get_mime_type(file_path) == expected


@pytest.mark.asyncio
async def test_get_mime_type_falls_back_to_extension():
    assert await get_mime_type(Path("example.mp4")) == "video/mp4"


@pytest.mark.asyncio
async def test_get_mime_type_returns_none_when_unknown():
    assert await get_mime_type(Path("example.unknown")) is None


@pytest.mark.asyncio
async def test_tool_paths_use_sandbox_home_instead_of_backend_home(tmp_path, monkeypatch):
    backend_home = tmp_path / "backend-home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("HOME", str(backend_home))
    _, _, read_file, write_file, _ = build_file_tools(workspace)

    await write_file(WriteFileArgs(path="~/notes.txt", content="user data"))

    assert (workspace / "home/notes.txt").read_text() == "user data"
    assert not (backend_home / "notes.txt").exists()
    assert "user data" in await read_file(ReadFileArgs(path="~/notes.txt"))


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace_kind", ["relative", "symlink"])
async def test_file_tools_normalize_workspace_at_construction(tmp_path, monkeypatch, workspace_kind):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(workspace, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    supplied_workspace = Path("workspace") if workspace_kind == "relative" else alias
    _, _, read_file, write_file, _ = build_file_tools(supplied_workspace)
    other_directory = tmp_path / "other"
    other_directory.mkdir()
    monkeypatch.chdir(other_directory)

    await write_file(WriteFileArgs(path="~/notes.txt", content="sandbox data"))

    assert (workspace / "home/notes.txt").read_text() == "sandbox data"
    assert "sandbox data" in await read_file(ReadFileArgs(path="~/notes.txt"))


def _read_file_tool(workspace: Path, max_result_chars: int = 10_000):
    _, _, read_file, _, _ = build_file_tools(
        workspace=workspace,
        max_result_chars=max_result_chars,
    )
    return read_file


def test_read_file_args_default_limit():
    assert ReadFileArgs(path="example.txt").limit == 2_000
    assert ReadFileArgs(path="example.txt", limit=None).limit is None


@pytest.mark.asyncio
async def test_read_file_accepts_long_line(tmp_path: Path):
    file_path = tmp_path / "long-line.txt"
    file_path.write_text("x" * 2_001, encoding="utf-8")
    read_file = _read_file_tool(tmp_path)

    result = await read_file(ReadFileArgs(path=file_path.name))

    assert result == f"1| {'x' * 2_001}\n\n(End of file — 1 lines total)"


@pytest.mark.asyncio
async def test_read_file_trims_oversized_result_at_line_boundary(tmp_path: Path):
    file_path = tmp_path / "many-lines.txt"
    file_path.write_text("\n".join("x" * 1_900 for _ in range(70)), encoding="utf-8")
    read_file = _read_file_tool(tmp_path)

    result = await read_file(ReadFileArgs(path=file_path.name, limit=70))

    assert result.endswith("(Showing lines 1-67 of 70. Use offset=68 to continue.)")
    assert not (tmp_path / "tool-results").exists()


@pytest.mark.asyncio
async def test_read_file_returns_paginated_result_under_limit(tmp_path: Path):
    file_path = tmp_path / "lines.txt"
    file_path.write_text("first\nsecond\nthird", encoding="utf-8")
    read_file = _read_file_tool(tmp_path, max_result_chars=100)

    result = await read_file(ReadFileArgs(path=file_path.name, offset=2, limit=1))

    assert result == "2| second\n\n(Showing lines 2-2 of 3. Use offset=3 to continue.)"


@pytest.mark.asyncio
async def test_edit_file_preserves_line_endings_and_symlink(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"first=old\r\nsecond=old\r\n")
    (tmp_path / "alias.txt").symlink_to(path.name)
    *_, edit = build_file_tools(tmp_path)

    await edit(EditFileArgs(path="alias.txt", old_text="first=old", new_text="first=new"))

    assert path.read_bytes() == b"first=new\r\nsecond=old\r\n"
    assert (tmp_path / "alias.txt").is_symlink()


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_all", [False, True])
async def test_edit_file_ambiguous_match_reports_failure_unless_replace_all(tmp_path, replace_all):
    path = tmp_path / "notes.txt"
    path.write_text("old\nold\n")
    *_, edit = build_file_tools(tmp_path)
    executor = ToolExecutor(tmp_path, 10000, {"edit_file": edit})
    result = await executor.execute_one(ToolCallRequest(
        id="edit", name="edit_file",
        arguments={"path": path.name, "old_text": "old", "new_text": "new", "replace_all": replace_all},
    ))
    assert result.succeeded is replace_all
    assert path.read_text() == ("new\nnew\n" if replace_all else "old\nold\n")
    if not replace_all:
        assert "old_text appears 2 times" in result.value


@pytest.mark.asyncio
async def test_same_file_write_does_not_wait_for_edit_processing(tmp_path, monkeypatch):
    path = tmp_path / "notes.txt"
    path.write_text("old", encoding="utf-8")
    first_read = asyncio.Event()
    release_read = asyncio.Event()

    async def read_snapshot(file_path):
        content = file_path.read_text(encoding="utf-8")
        first_read.set()
        await release_read.wait()
        return content, "utf-8"

    monkeypatch.setattr(filesystem, "super_read_text", read_snapshot)
    *_, edit = build_file_tools(tmp_path)
    _, _, _, write, _ = build_file_tools(tmp_path)
    first = asyncio.create_task(edit(EditFileArgs(path=path.name, old_text="old", new_text="edited")))
    try:
        await asyncio.wait_for(first_read.wait(), 5)
        await asyncio.wait_for(write(WriteFileArgs(path=path.name, content="written")), 5)
        assert path.read_text(encoding="utf-8") == "written"
    finally:
        release_read.set()
        await first

    assert path.read_text(encoding="utf-8") == "edited"


@pytest.mark.asyncio
async def test_read_file_decodes_one_snapshot_even_if_file_is_replaced(tmp_path, monkeypatch):
    path = tmp_path / "notes.txt"
    path.write_text("old", encoding="utf-8")
    decode = filesystem.super_read_text

    async def replace_then_decode(*args, **kwargs):
        path.write_text("new", encoding="utf-8")
        return await decode(*args, **kwargs)

    monkeypatch.setattr(filesystem, "super_read_text", replace_then_decode)

    assert await _read_file_tool(tmp_path)(ReadFileArgs(path=path.name)) == (
        "1| old\n\n(End of file — 1 lines total)"
    )
