from pathlib import Path
from re import escape
from pydantic import BaseModel, Field
from ...utils.helper import msys_to_win, is_relative_path
import platform
from ...config.settings import settings
from ...agent.skills.skill import BUILTIN_SKILLS_DIR

if platform.system() == "Windows":
    _IS_WINDOWS = True
else:
    _IS_WINDOWS = False


class ReadFileArgs(BaseModel):
    path: str = Field(description="The file path to read")


class WriteFileArgs(BaseModel):
    path: str = Field(description="The file  path to write to")
    content: str = Field(description="The content to write")


class EditFileArgs(BaseModel):
    path: str = Field(description="The file  path to edit")
    old_text: str = Field(description="The exact text to find and replace")
    new_text: str = Field(description="The text to replace with")


class ListDirArgs(BaseModel):
    path: str = Field(description="The directory  path to list")


def build_file_tools(workspace: str):
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
        print(f"Resolved path: {ops_path}, {base}")
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
        file_path = safe_resolve(path, include_buildin_skills_path=True)
        if not file_path.exists():
            return f"Error: File not found: {path}"
        if not file_path.is_file():
            return f"Error: Not a file: {path}"
        return file_path.read_text(encoding="utf-8")

    async def write_file(args: WriteFileArgs) -> str:
        """Write content to a file, creating parent directories if needed."""
        path = args.path
        content = args.content
        file_path = safe_resolve(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} bytes to {path}"

    async def edit_file(args: EditFileArgs) -> str:
        """Edit a file by replacing old_text with new_text."""
        path = args.path
        old_text = args.old_text
        new_text = args.new_text
        file_path = safe_resolve(path)
        if not file_path.exists():
            return f"Error: File not found: {path}"

        content = file_path.read_text(encoding="utf-8")
        if old_text not in content:
            return "Error: old_text not found in file. Make sure it matches exactly."

        count = content.count(old_text)
        if count > 1:
            return f"Warning: old_text appears {count} times. Please provide more context to make it unique."

        file_path.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
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
