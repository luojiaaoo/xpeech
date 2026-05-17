from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field
from ...utils.helper import (
    msys_to_win,
    is_relative_path,
    detect_image_mime,
    super_read_text,
    compress_image_bytes_to_jpg,
    compress_video_to_mp4,
    read_video_metadata,
)
import platform
from ...agent.skills.skill import BUILTIN_SKILLS_DIR
import aiofiles
import mimetypes
from ..prompt.helper import build_image_content_blocks, build_video_content_blocks
import difflib
import tempfile


if platform.system() == "Windows":
    _IS_WINDOWS = True
else:
    _IS_WINDOWS = False


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


class ListDirArgs(BaseModel):
    path: str = Field(description="The directory path to list")
    recursive: bool = Field(description="Recursively list all files (default false)", default=False)
    max_entries: int = Field(
        description="Maximum entries to return (default 200)",
        default=200,
        ge=1,
    )


def build_file_tools(workspace: str, restrict_tools_to_workspace: bool):
    base = Path(workspace).expanduser().resolve()
    if not base.exists():
        raise ValueError(f"Invalid workspace: {workspace}")

    def safe_resolve(user_path: str, include_buildin_skills_path: bool = False) -> Path:
        """Resolve a user path safely."""
        if _IS_WINDOWS:
            # 可能是 MSYS2 路径，转换为 Windows 路径
            user_path = msys_to_win(user_path)
        # 如果不是绝对路径，则相对路径是相对用户工作路径的路径
        if not Path(user_path).is_absolute():
            ops_path = (base / user_path).resolve()
        else:
            ops_path = Path(user_path).resolve()
        if restrict_tools_to_workspace:
            # 检查是否逃逸
            if include_buildin_skills_path and is_relative_path(path_target=ops_path, base=BUILTIN_SKILLS_DIR):
                return ops_path

            if is_relative_path(path_target=ops_path, base=base):
                return ops_path
            else:
                raise PermissionError(f"Path escapes workspace: {user_path}")
        return ops_path

    async def read_image(args: ReadImageArgs) -> str | list[dict[str, Any]]:
        """
        Read an image from a local file path.
        Call this tool only when the image must be loaded from the filesystem.
        If the image is already provided in the conversation (for example, as an attachment or inline image input),
        do not call this tool and analyze the image directly.
        """
        path = args.path
        file_path = safe_resolve(path, include_buildin_skills_path=True)
        if not file_path.exists():
            return f"Error: File not found: {path}"
        if not file_path.is_file():
            return f"Error: Not a file: {path}"

        async with aiofiles.open(file_path, "rb") as f:
            raw = await f.read()
        if not raw:
            return f"(Empty file: {path})"

        try:
            raw = compress_image_bytes_to_jpg(raw)
        except Exception as e:
            return f"Error: Cannot read image file {path}: {e}"

        mime = detect_image_mime(raw) or mimetypes.guess_type(file_path)[0]
        if not mime or not mime.startswith("image/"):
            return f"Error: Not an image file: {path} (MIME: {mime or 'unknown'})"

        return build_image_content_blocks(raw, mime, path, f"(Image file: {path})")

    async def read_video(args: ReadVideoArgs) -> str | list[dict[str, Any]]:
        """
        Read a video from a local file path.
        start_time=0 starts at the beginning. Omit end_time to read until the end.
        The returned label includes duration, width, and height.
        """
        path = args.path
        file_path = safe_resolve(path, include_buildin_skills_path=True)
        if not file_path.exists():
            return f"Error: File not found: {path}"
        if not file_path.is_file():
            return f"Error: Not a file: {path}"

        start_time = args.start_time if args.start_time is not None and args.start_time >= 0 else 0
        end_time = args.end_time
        if end_time is not None and end_time <= start_time:
            return f"Error: end_time must be greater than start_time: {path}"

        try:
            video_info = await read_video_metadata(file_path)
        except Exception as e:
            return f"Error: Cannot read video metadata {path}: {e}"

        duration = video_info.get("duration")
        width = video_info.get("width")
        height = video_info.get("height")
        if not isinstance(duration, (int, float)) or not isinstance(width, int) or not isinstance(height, int):
            return f"Error: Cannot read required video metadata {path}: {video_info}"
        if start_time >= duration:
            return f"Error: start_time must be less than video duration: {path} (total {duration} seconds)"
        if end_time is not None and end_time > duration:
            return f"Error: end_time must be less than video duration: {path} (total {duration} seconds)"
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
                async with aiofiles.open(output_path, "rb") as f:
                    raw = await f.read()
                output_mime = mimetypes.guess_type(output_path)[0] or "video/mp4"
        except Exception as e:
            return f"Error: Cannot read video file {path}: {e}"

        if not raw:
            return f"(Empty video file: {path})"

        return build_video_content_blocks(
            raw,
            output_mime,
            path,
            **{
                "duration": effective_duration,
                "width": width,
                "height": height,
            },
        )

    async def read_file(args: ReadFileArgs) -> str:
        """
        Read the contents of a file. Returns numbered lines.
        Use offset and limit to paginate through large files.
        """
        _MAX_CHARS = 128_000
        path = args.path
        offset = args.offset
        limit = args.limit

        file_path = safe_resolve(path, include_buildin_skills_path=True)
        if not file_path.exists():
            return f"Error: File not found: {path}"
        if not file_path.is_file():
            return f"Error: Not a file: {path}"

        async with aiofiles.open(file_path, "rb") as f:
            raw = await f.read()
        if not raw:
            return f"(Empty file: {path})"

        text_content, _ = await super_read_text(file_path)

        if text_content is None:
            return f"Error: Cannot read binary file {path}."

        all_lines = text_content.splitlines()
        total = len(all_lines)

        if offset < 1:
            offset = 1
        if offset > total:
            return f"Error: offset {offset} is beyond end of file ({total} lines)"

        start = offset - 1
        end = min(start + limit, total)
        numbered = [f"{start + i + 1}| {line}" for i, line in enumerate(all_lines[start:end])]
        result = "\n".join(numbered)

        if len(result) > _MAX_CHARS:
            trimmed, chars = [], 0
            for line in numbered:
                chars += len(line) + 1
                if chars > _MAX_CHARS:
                    break
                trimmed.append(line)
            end = start + len(trimmed)
            result = "\n".join(trimmed)

        if end < total:
            result += f"\n\n(Showing lines {offset}-{end} of {total}. Use offset={end + 1} to continue.)"
        else:
            result += f"\n\n(End of file — {total} lines total)"
        return result

    async def write_file(args: WriteFileArgs) -> str:
        """Write content to a file at the given path. Creates parent directories if needed."""
        path = args.path
        content = args.content
        file_path = safe_resolve(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(content)
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
            return f"Error: File not found: {path}"
        if not file_path.is_file():
            return f"Error: Not a file: {path}"

        content, encoding = await super_read_text(file_path=file_path)
        if content is None:
            return f"Error: Cannot edit binary file {path}."
        uses_crlf = "\r\n" in content
        if uses_crlf:
            content = content.replace("\r\n", "\n")
        match, count = _find_match(content, old_text.replace("\r\n", "\n"))
        if match is None:
            return _not_found_msg(old_text, content, path)
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
        async with aiofiles.open(file_path, "w", encoding=encoding) as f:
            await f.write(new_content)
        return f"Successfully edited {path}"

    async def list_dir(args: ListDirArgs) -> str:
        """
        List the contents of a directory.
        Set recursive=true to explore nested structure.
        Common noise directories (.git, node_modules, __pycache__, etc.) are auto-ignored.
        """
        _DEFAULT_MAX = 200
        _IGNORE_DIRS = {
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            "dist",
            "build",
            ".tox",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".coverage",
            "htmlcov",
        }
        path = args.path
        max_entries = args.max_entries
        recursive = args.recursive
        dir_path = safe_resolve(path, include_buildin_skills_path=True)
        if not dir_path.exists():
            return f"Error: Directory not found: {path}"
        if not dir_path.is_dir():
            return f"Error: Not a directory: {path}"

        cap = max_entries or _DEFAULT_MAX
        items: list[str] = []
        total = 0

        if recursive:
            for item in sorted(dir_path.rglob("*")):
                if any(p in _IGNORE_DIRS for p in item.parts):
                    continue
                total += 1
                if len(items) < cap:
                    rel = item.relative_to(dir_path)
                    items.append(f"{rel}/" if item.is_dir() else str(rel))
        else:
            for item in sorted(dir_path.iterdir()):
                if item.name in _IGNORE_DIRS:
                    continue
                total += 1
                if len(items) < cap:
                    kind = "[dir] " if item.is_dir() else "[file] "
                    items.append(f"{kind}{item.name}")

        if not items and total == 0:
            return f"Directory {path} is empty"

        result = "\n".join(items)
        if total > cap:
            result += f"\n\n(truncated, showing first {cap} of {total} entries)"
        return result

    return read_image, read_video, read_file, write_file, edit_file, list_dir
