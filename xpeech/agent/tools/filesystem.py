import asyncio
import base64
import difflib
import mimetypes
import tempfile
from pathlib import Path
from typing import Any

from markitdown import MarkItDown
from pydantic import BaseModel, Field

from ...utils.helper import (
    compress_image_bytes_to_jpg,
    compress_video_to_mp4,
    detect_image_mime,
    ensure_path_async,
    read_bytes_async,
    read_video_metadata,
    super_read_text,
    write_text_async,
)
from .helper import safe_resolve_workspace_path


class ReadImageArgs(BaseModel):
    path: str = Field(description="The image file path to read")


class ReadVideoArgs(BaseModel):
    path: str = Field(description="The video file path to read")
    start_time: int = Field(
        description="Start time in seconds (default 0, start from the beginning)",
        default=0,
        ge=0,
    )
    end_time: int | None = Field(
        description="End time in seconds. Omit this value to read until the end of the video. If provided, it must be greater than start_time.",
        default=None,
        ge=1,
    )


class ReadFileArgs(BaseModel):
    path: str = Field(description="The file path to read")
    offset: int = Field(
        description="Line number to start reading from (1-indexed, default 1)",
        default=1,
        ge=1,
    )
    limit: int = Field(
        description="Maximum number of lines to read (default 2000)",
        default=2000,
        ge=1,
    )
    max_line_chars: int = Field(
        description="Reject any selected line longer than this many characters (default 2000, maximum 10000)",
        default=2000,
        ge=1,
        le=10_000,
    )


class OfficeReadArgs(BaseModel):
    path: str = Field(description="Path to the office document (docx, xlsx, pdf, pptx, etc.)")


class WriteFileArgs(BaseModel):
    path: str = Field(description="The file path to write to")
    content: str = Field(description="The content to write")


class EditFileArgs(BaseModel):
    path: str = Field(description="The file path to edit")
    old_text: str = Field(description="The text to find and replace")
    new_text: str = Field(description="The text to replace with")
    replace_all: bool = Field(
        description="Replace all occurrences (default false)",
        default=False,
    )


def build_file_tools(workspace: str | Path, max_result_chars: int = 10_000):
    base = Path(workspace).expanduser().resolve()
    if not base.exists():
        raise ValueError(f"Invalid workspace: {workspace}")
    if max_result_chars < 1:
        raise ValueError("max_result_chars must be positive")

    def safe_resolve(user_path: str, protect_read_only_file: bool = True) -> Path:
        return safe_resolve_workspace_path(
            user_path,
            base,
            protect_read_only_file=protect_read_only_file,
        )

    async def read_image(args: ReadImageArgs) -> str | list[dict[str, Any]]:
        """
        Read an image from a local file path.
        """
        path = args.path
        file_path = safe_resolve(path, protect_read_only_file=False)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not file_path.is_file():
            raise ValueError(f"Not a file: {path}")

        raw = await read_bytes_async(file_path)
        if not raw:
            return f"(Empty file: {path})"

        try:
            raw = compress_image_bytes_to_jpg(raw)
        except Exception as e:
            raise RuntimeError(f"Cannot read image file {path}: {e}") from e

        mime = detect_image_mime(raw) or mimetypes.guess_type(file_path)[0]
        if not mime or not mime.startswith("image/"):
            raise ValueError(f"Not an image file: {path} (MIME: {mime or 'unknown'})")

        b64 = base64.b64encode(raw).decode()
        return [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
                "_meta": f"The read_image tool returned this image: {path}",
            },
            {"type": "text", "text": f"(Image file: {path})"},
            {"type": "text", "text": "Image content is attached in the next user message."},
        ]

    async def read_video(args: ReadVideoArgs) -> str | list[dict[str, Any]]:
        """
        Read a video from a local file path.
        start_time=0 starts at the beginning. Omit end_time to read until the end.
        The returned label includes duration, width, and height.
        """
        path = args.path
        file_path = safe_resolve(path, protect_read_only_file=False)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not file_path.is_file():
            raise ValueError(f"Not a file: {path}")

        start_time = args.start_time if args.start_time is not None and args.start_time >= 0 else 0
        end_time = args.end_time
        if end_time is not None and end_time <= start_time:
            raise ValueError(f"end_time must be greater than start_time: {path}")

        try:
            video_info = await read_video_metadata(file_path)
        except Exception as e:
            raise RuntimeError(f"Cannot read video metadata {path}: {e}") from e

        duration = video_info.get("duration")
        width = video_info.get("width")
        height = video_info.get("height")
        if not isinstance(duration, (int, float)) or not isinstance(width, int) or not isinstance(height, int):
            raise TypeError(f"Cannot read required video metadata {path}: {video_info}")
        if start_time >= duration:
            raise ValueError(f"start_time must be less than video duration: {path} (total {duration} seconds)")
        if end_time is not None and end_time > duration:
            raise ValueError(f"end_time must be less than video duration: {path} (total {duration} seconds)")
        if end_time is not None:
            effective_duration = end_time - start_time
        else:
            effective_duration = duration - start_time

        try:
            with tempfile.TemporaryDirectory(prefix="xpeech-video-") as temp_dir:
                output_path = await compress_video_to_mp4(
                    input_path=file_path,
                    output_path=Path(temp_dir) / f"{file_path.stem}_compressed.mp4",
                    start_time=start_time,
                    end_time=end_time,
                )
                raw = await read_bytes_async(output_path)
                output_mime = mimetypes.guess_type(output_path)[0] or "video/mp4"
        except Exception as e:
            raise RuntimeError(f"Cannot read video file {path}: {e}") from e

        if not raw:
            return f"(Empty video file: {path})"
        video_info = {"duration": effective_duration, "width": width, "height": height, "path": path}
        label = "\n".join(f"{key}: {value}" for key, value in video_info.items())
        b64 = base64.b64encode(raw).decode()
        return [
            {
                "type": "video_url",
                "video_url": {"url": f"data:{output_mime};base64,{b64}"},
                "_meta": f"The read_video tool returned this video: {path}",
            },
            {"type": "text", "text": label},
            {"type": "text", "text": "Video content is attached in the next user message."},
        ]

    async def read_file(args: ReadFileArgs) -> str:
        """
        Read the contents of a file. Returns numbered lines.
        Use offset and limit to paginate through large files.
        """
        path = args.path
        offset = args.offset
        limit = args.limit

        file_path = safe_resolve(path, protect_read_only_file=False)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not file_path.is_file():
            raise ValueError(f"Not a file: {path}")

        raw = await read_bytes_async(file_path)
        if not raw:
            return f"(Empty file: {path})"

        text_content, _ = await super_read_text(file_path)

        if text_content is None:
            raise ValueError(f"Cannot read binary file {path}.")

        all_lines = text_content.splitlines()
        total = len(all_lines)

        offset = max(offset, 1)
        if offset > total:
            raise ValueError(f"offset {offset} is beyond end of file ({total} lines)")

        start = offset - 1
        end = min(start + limit, total)
        selected_lines = all_lines[start:end]
        for index, line in enumerate(selected_lines, start=offset):
            if len(line) > args.max_line_chars:
                raise ValueError(
                    f"Cannot read {path}: line {index} contains {len(line)} characters, "
                    f"exceeding max_line_chars={args.max_line_chars}."
                )
        numbered = [f"{start + i + 1}| {line}" for i, line in enumerate(selected_lines)]
        result = "\n".join(numbered)

        if end < total:
            result += f"\n\n(Showing lines {offset}-{end} of {total}. Use offset={end + 1} to continue.)"
        else:
            result += f"\n\n(End of file — {total} lines total)"
        if len(result) > max_result_chars:
            raise ValueError(
                f"Cannot read {path}: lines {offset}-{end} would return {len(result)} characters, "
                f"exceeding the maximum {max_result_chars}. Reduce limit and try again."
            )
        return result

    async def office_read(args: OfficeReadArgs) -> str:
        """
        Read content from office documents (docx, xlsx, pdf, pptx, etc.) and extract as markdown.
        Supports Word documents, Excel spreadsheets, PDF files, PowerPoint presentations, and more.
        Returns the extracted text content with document title if available.
        """
        path = args.path
        file_path = safe_resolve(path, protect_read_only_file=False)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not file_path.is_file():
            raise ValueError(f"Not a file: {path}")

        supported_extensions = {".docx", ".xlsx", ".pdf", ".pptx", ".doc", ".xls", ".ppt"}
        if file_path.suffix.lower() not in supported_extensions:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported formats: {', '.join(sorted(supported_extensions))}"
            )

        try:
            md = MarkItDown()
            result = await asyncio.to_thread(md.convert, str(file_path))
            title = getattr(result, "title", None)
            title_info = f"[TITLE: {title}]\n\n" if title else ""
            content = getattr(result, "text_content", str(result))
            if not content or not content.strip():
                return f"(Empty document: {path})"
            return f"{title_info}{content}"
        except Exception as e:
            raise RuntimeError(f"Failed to read document {path}: {e}") from e

    async def write_file(args: WriteFileArgs) -> str:
        """Write content to a file at the given path. Creates parent directories if needed."""
        path = args.path
        content = args.content
        file_path = safe_resolve(path)
        await ensure_path_async(file_path.parent)
        await write_text_async(file_path, content)
        return f"Successfully wrote {len(content)} bytes to {path}"

    async def edit_file(args: EditFileArgs) -> str:
        """
        Edit a file by replacing old_text with new_text.
        Supports minor whitespace/line-ending differences.
        Set replace_all=true to replace every occurrence.
        """

        def _not_found_msg(old_text: str, content: str, path: str) -> str:
            lines = content.splitlines(keepends=True)
            old_lines = old_text.splitlines(keepends=True)
            window = len(old_lines)

            best_ratio, best_start = 0.0, 0
            for i in range(max(1, len(lines) - window + 1)):
                ratio = difflib.SequenceMatcher(None, old_lines, lines[i : i + window]).ratio()
                if ratio > best_ratio:
                    best_ratio, best_start = ratio, i

            if best_ratio > 0.5:
                diff = "\n".join(
                    difflib.unified_diff(
                        old_lines,
                        lines[best_start : best_start + window],
                        fromfile="old_text (provided)",
                        tofile=f"{path} (actual, line {best_start + 1})",
                        lineterm="",
                    )
                )
                return f"Error: old_text not found in {path}.\nBest match ({best_ratio:.0%} similar) at line {best_start + 1}:\n{diff}"
            return f"Error: old_text not found in {path}. No similar text found. Verify the file content."

        def _find_match(content: str, old_text: str) -> tuple[str | None, int]:
            """Locate old_text in content: exact first, then line-trimmed sliding window.

            Both inputs should use LF line endings (caller normalises CRLF).
            Returns (matched_fragment, count) or (None, 0).
            """
            if old_text in content:
                return old_text, content.count(old_text)

            old_lines = old_text.splitlines()
            if not old_lines:
                return None, 0
            stripped_old = [l.strip() for l in old_lines]
            content_lines = content.splitlines()

            candidates = []
            for i in range(len(content_lines) - len(stripped_old) + 1):
                window = content_lines[i : i + len(stripped_old)]
                if [l.strip() for l in window] == stripped_old:
                    candidates.append("\n".join(window))

            if candidates:
                return candidates[0], len(candidates)
            return None, 0

        path = args.path
        old_text = args.old_text
        new_text = args.new_text
        replace_all = args.replace_all
        file_path = safe_resolve(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not file_path.is_file():
            raise ValueError(f"Not a file: {path}")

        content, encoding = await super_read_text(file_path=file_path)
        if content is None:
            raise ValueError(f"Cannot edit binary file {path}.")
        uses_crlf = "\r\n" in content
        if uses_crlf:
            content = content.replace("\r\n", "\n")
        match, count = _find_match(content, old_text.replace("\r\n", "\n"))
        if match is None:
            raise ValueError(_not_found_msg(old_text, content, path))
        if count > 1 and not replace_all:
            return (
                f"Warning: old_text appears {count} times. "
                "Provide more context to make it unique, or set replace_all=true."
            )

        norm_new = new_text.replace("\r\n", "\n")
        new_content = content.replace(match, norm_new) if replace_all else content.replace(match, norm_new, 1)
        if uses_crlf:
            new_content = new_content.replace("\n", "\r\n")
        # 原本是纯英文，后面加了中文，encoding会变成utf-8
        if encoding == "ascii":
            encoding = "utf-8"
        await write_text_async(file_path, new_content, encoding=encoding)
        return f"Successfully edited {path}"

    return read_image, read_video, read_file, write_file, edit_file, office_read
