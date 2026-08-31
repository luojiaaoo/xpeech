from pathlib import Path
from types import SimpleNamespace

import pytest

from xpeech.agent.tools import filesystem
from xpeech.agent.tools.filesystem import OfficeReadArgs, ReadFileArgs, build_file_tools


def _read_file_tool(workspace: Path, max_result_chars: int = 10_000):
    _, _, read_file, _, _, _ = build_file_tools(
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
async def test_read_office_file_resolves_relative_path_in_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    document = tmp_path / "example.docx"
    document.write_bytes(b"placeholder")
    converted_paths: list[str] = []

    class FakeMarkItDown:
        def convert(self, path: str):
            converted_paths.append(path)
            return SimpleNamespace(title="Example", text_content="# Content")

    monkeypatch.setattr(filesystem, "MarkItDown", FakeMarkItDown)
    _, _, _, _, _, read_office_file = build_file_tools(tmp_path)

    result = await read_office_file(OfficeReadArgs(path=document.name))

    assert converted_paths == [str(document)]
    assert result == "[TITLE: Example]\n\n# Content"
