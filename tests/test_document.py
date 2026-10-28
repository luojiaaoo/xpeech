from pathlib import Path

import pytest

from xpeech.agent.tools import document
from xpeech.agent.tools.document import DocReadArgs, build_document_tool
from xpeech.exceptions import PathProtectionError


@pytest.mark.asyncio
async def test_read_doc_file_resolves_relative_path_in_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    document_path = tmp_path / "example.docx"
    document_path.write_bytes(b"placeholder")
    converted_paths: list[str] = []

    def convert(path: str):
        converted_paths.append(path)
        return "# Content"

    monkeypatch.setattr(document.anydoc, "to_markdown", convert)
    read_doc_file = build_document_tool(tmp_path)

    result = await read_doc_file(DocReadArgs(path=document_path.name))

    assert converted_paths == [str(document_path)]
    assert result == "# Content"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filename,content,expected",
    [
        ("example.csv", b"name,value\nalpha,42\n", "| alpha | 42 |"),
        ("example.rtf", rb"{\rtf1\ansi Hello anydoc\par}", "Hello anydoc"),
        ("no-extension", rb"{\rtf1\ansi Detected from content\par}", "Detected from content"),
    ],
)
async def test_read_doc_file_converts_new_formats(tmp_path, filename, content, expected):
    (tmp_path / filename).write_bytes(content)
    read_doc_file = build_document_tool(tmp_path)

    result = await read_doc_file(DocReadArgs(path=filename))

    assert expected in result


@pytest.mark.asyncio
async def test_read_doc_file_preserves_workspace_protection(tmp_path):
    read_doc_file = build_document_tool(tmp_path)
    with pytest.raises(PathProtectionError):
        await read_doc_file(DocReadArgs(path="../outside.rtf"))


@pytest.mark.asyncio
async def test_read_doc_file_reports_conversion_failure(tmp_path):
    (tmp_path / "broken.docx").write_bytes(b"invalid document")
    read_doc_file = build_document_tool(tmp_path)
    with pytest.raises(RuntimeError, match="Failed to read document broken.docx"):
        await read_doc_file(DocReadArgs(path="broken.docx"))
