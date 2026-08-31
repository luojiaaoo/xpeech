from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from xpeech.agent.tools import filesystem
from xpeech.agent.tools.filesystem import OfficeReadArgs, ReadFileArgs, build_file_tools


def _read_file_tool(workspace: Path, max_result_chars: int = 10_000):
    _, _, read_file, _, _, _ = build_file_tools(
        workspace=workspace,
        max_result_chars=max_result_chars,
    )
    return read_file


def test_read_file_args_limit_each_line():
    assert ReadFileArgs(path="example.txt").max_line_chars == 2_000
    with pytest.raises(ValidationError, match="max_line_chars"):
        ReadFileArgs(path="example.txt", max_line_chars=10_001)


@pytest.mark.asyncio
async def test_read_file_rejects_oversized_line(tmp_path: Path):
    file_path = tmp_path / "long-line.txt"
    file_path.write_text("x" * 2_001, encoding="utf-8")
    read_file = _read_file_tool(tmp_path)

    with pytest.raises(ValueError, match=r"line 1 contains 2001 characters.*max_line_chars=2000"):
        await read_file(ReadFileArgs(path=file_path.name))


@pytest.mark.asyncio
async def test_read_file_rejects_oversized_result_without_creating_result_file(tmp_path: Path):
    file_path = tmp_path / "many-lines.txt"
    file_path.write_text("\n".join("x" * 1_900 for _ in range(6)), encoding="utf-8")
    read_file = _read_file_tool(tmp_path)

    with pytest.raises(ValueError, match=r"would return .* exceeding the maximum 10000"):
        await read_file(ReadFileArgs(path=file_path.name, limit=6))

    assert not (tmp_path / "tool-results").exists()


@pytest.mark.asyncio
async def test_read_file_returns_paginated_result_under_limit(tmp_path: Path):
    file_path = tmp_path / "lines.txt"
    file_path.write_text("first\nsecond\nthird", encoding="utf-8")
    read_file = _read_file_tool(tmp_path, max_result_chars=100)

    result = await read_file(ReadFileArgs(path=file_path.name, offset=2, limit=1))

    assert result == "2| second\n\n(Showing lines 2-2 of 3. Use offset=3 to continue.)"


@pytest.mark.asyncio
async def test_office_read_resolves_relative_path_in_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    document = tmp_path / "example.docx"
    document.write_bytes(b"placeholder")
    converted_paths: list[str] = []

    class FakeMarkItDown:
        def convert(self, path: str):
            converted_paths.append(path)
            return SimpleNamespace(title="Example", text_content="# Content")

    monkeypatch.setattr(filesystem, "MarkItDown", FakeMarkItDown)
    _, _, _, _, _, office_read = build_file_tools(tmp_path)

    result = await office_read(OfficeReadArgs(path=document.name))

    assert converted_paths == [str(document)]
    assert result == "[TITLE: Example]\n\n# Content"
