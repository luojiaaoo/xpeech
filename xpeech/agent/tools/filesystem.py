from pathlib import Path
from pydantic import BaseModel, Field
from ...utils.helper import msys_to_win, is_relative_path, detect_image_mime, super_read_text
import platform
from ...config.settings import settings
from ...agent.skills.skill import BUILTIN_SKILLS_DIR
import aiofiles
import mimetypes
from ..prompt.helper import build_image_content_blocks


if platform.system() == "Windows":
    _IS_WINDOWS = True
else:
    _IS_WINDOWS = False

_MAX_CHARS = 128_000
_DEFAULT_LIMIT = 2000


class ReadFileArgs(BaseModel):
    path: str = Field(description="The file path to read")
    offset: int = Field(description="The offset line to start reading from")
    limit: int = Field(description="The maximum number of lines to read")


class WriteFileArgs(BaseModel):
    path: str = Field(description="The file  path to write to")
    content: str = Field(description="The content to write")


class EditFileArgs(BaseModel):
    path: str = Field(description="The file  path to edit")
    old_text: str = Field(description="The exact text to find and replace")
    new_text: str = Field(description="The text to replace with")


class ListDirArgs(BaseModel):
    path: str = Field(description="The directory  path to list")


def build_file_tools(workspace: str, support_image: bool):
    base = Path(workspace).expanduser().resolve()
    if not base.exists():
        raise ValueError(f"Invalid workspace: {workspace}")

    def safe_resolve(user_path: str, include_buildin_skills_path: bool = False) -> Path:
        """Resolve a user path safely."""
        if _IS_WINDOWS:
            # 可能是 MSYS2 路径，转换为 Windows 路径
            user_path = msys_to_win(user_path)
        # 相对路径是相对用户工作路径的路径
        ops_path = (base / user_path).resolve()
        if settings.path.restrict_tools_to_workspace:
            # 检查是否逃逸
            if include_buildin_skills_path and is_relative_path(path_target=ops_path, base=BUILTIN_SKILLS_DIR):
                return ops_path

            if is_relative_path(path_target=ops_path, base=base):
                return ops_path
            else:
                raise PermissionError(f"Path escapes workspace: {user_path}")

    async def read_file(args: ReadFileArgs) -> str:
        """Read the contents of a file."""
        path = args.path
        offset = args.offset
        limit = args.limit

        file_path = safe_resolve(path, include_buildin_skills_path=True)
        if not file_path.exists():
            return f"Error: File not found: {path}"
        if not file_path.is_file():
            return f"Error: Not a file: {path}"

        async with aiofiles.open(file_path, "r+b") as f:
            raw = await f.read()
        if not raw:
            return f"(Empty file: {path})"
        if support_image:
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if mime and mime.startswith("image/"):
                return build_image_content_blocks(raw, mime, path, f"(Image file: {path})")

        text_content = super_read_text(file_path)
        if text_content is None:
            return f"Error: Cannot read binary file {path} (MIME: {mime or 'unknown'})."

        all_lines = text_content.splitlines()
        total = len(all_lines)

        if offset < 1:
            offset = 1
        if offset > total:
            return f"Error: offset {offset} is beyond end of file ({total} lines)"

        start = offset - 1
        end = min(start + (limit or _DEFAULT_LIMIT), total)
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
        """Write content to a file, creating parent directories if needed."""
        path = args.path
        content = args.content
        file_path = safe_resolve(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(content)
        return f"Successfully wrote {len(content)} bytes to {path}"

    async def edit_file(args: EditFileArgs) -> str:
        """Edit a file by replacing old_text with new_text."""
        path = args.path
        old_text = args.old_text
        new_text = args.new_text
        file_path = safe_resolve(path)
        if not file_path.exists():
            return f"Error: File not found: {path}"

        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()

        if old_text not in content:
            return "Error: old_text not found in file. Make sure it matches exactly."

        count = content.count(old_text)
        if count > 1:
            return f"Warning: old_text appears {count} times. Please provide more context to make it unique."

        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(content.replace(old_text, new_text, 1))
        return f"Successfully edited {path}"

    async def list_dir(args: ListDirArgs) -> str:
        """List the contents of a directory."""
        path = args.path
        dir_path = safe_resolve(path, include_buildin_skills_path=True)
        if not dir_path.exists():
            return f"Error: Directory not found: {path}"
        if not dir_path.is_dir():
            return f"Error: Not a directory: {path}"

        items = []
        for item in sorted(dir_path.iterdir()):
            prefix = "📁 " if item.is_dir() else "📄 "
            items.append(f"{prefix}{item.name}")

        return f"Directory {path} is empty" if not items else "\n".join(items)

    return read_file, write_file, edit_file, list_dir
