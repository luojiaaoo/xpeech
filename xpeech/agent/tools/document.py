"""使用 anydoc 将本地文档转换为 Markdown。"""

import asyncio
from pathlib import Path

import anydoc
from pydantic import BaseModel, Field

from .helper import safe_resolve_workspace_path


class DocReadArgs(BaseModel):
    path: str = Field(
        description="Path to a document (doc/docx, ppt/pptx, xls/xlsx, odt/ods/odp, rtf, epub, csv, pdf, etc.)"
    )


async def read_document(file_path: Path, display_path: str) -> str:
    """转换已解析的路径，保留面向用户的原始文件名。"""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {display_path}")
    if not file_path.is_file():
        raise ValueError(f"Not a file: {display_path}")

    try:
        content = await asyncio.to_thread(anydoc.to_markdown, str(file_path))
        if not content or not content.strip():
            return f"(Empty document: {display_path})"
        return content
    except Exception as error:
        raise RuntimeError(f"Failed to read document {display_path}: {error}") from error


def build_document_tool(workspace: str | Path):
    """构建绑定当前工作区的文档读取工具。"""
    base = Path(workspace).expanduser().resolve()
    if not base.exists():
        raise ValueError(f"Invalid workspace: {workspace}")

    async def read_doc_file(args: DocReadArgs) -> str:
        """Read Word, PowerPoint, Excel, OpenDocument, RTF, EPUB, CSV and PDF documents as Markdown.
        Supports legacy and macro-enabled Office formats. Detects formats from file content.
        """
        file_path = safe_resolve_workspace_path(args.path, base, protect_read_only_file=False)
        return await read_document(file_path, args.path)

    return read_doc_file
