from pathlib import Path
from pydantic import BaseModel, Field


class ReadFileArgs(BaseModel):
    path: str = Field(description="The file path to read")


class WriteFileArgs(BaseModel):
    path: str = Field(description="The file path to write to")
    content: str = Field(description="The content to write")


class EditFileArgs(BaseModel):
    path: str = Field(description="The file path to edit")
    old_text: str = Field(description="The exact text to find and replace")
    new_text: str = Field(description="The text to replace with")


class ListDirArgs(BaseModel):
    path: str = Field(description="The directory path to list")


def build_file_tools(workdir: str):
    base = Path(workdir).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        raise ValueError(f"Invalid workdir: {workdir}")

    def safe_resolve(user_path: str) -> Path:
        """Resolve a user path safely inside the workdir."""
        p = Path(user_path).expanduser()
        target = p if p.is_absolute() else (base / p)
        resolved = target.resolve(strict=False)

        try:
            resolved.relative_to(base)
        except ValueError:
            raise PermissionError(f"Path escapes workdir: {user_path}")

        return resolved

    async def read_file(args: ReadFileArgs) -> str:
        """Read the contents of a file inside the workdir."""
        path = args.path
        try:
            file_path = safe_resolve(path)
            if not file_path.exists():
                return f"Error: File not found: {path}"
            if not file_path.is_file():
                return f"Error: Not a file: {path}"
            return file_path.read_text(encoding="utf-8")
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    async def write_file(args: WriteFileArgs) -> str:
        """Write content to a file inside the workdir, creating parent directories if needed."""
        path = args.path
        content = args.content
        try:
            file_path = safe_resolve(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to {path}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    async def edit_file(args: EditFileArgs) -> str:
        """Edit a file inside the workdir by replacing old_text with new_text."""
        path = args.path
        old_text = args.old_text
        new_text = args.new_text
        try:
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
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error editing file: {str(e)}"

    async def list_dir(args: ListDirArgs) -> str:
        """List the contents of a directory inside the workdir."""
        path = args.path
        try:
            dir_path = safe_resolve(path)
            if not dir_path.exists():
                return f"Error: Directory not found: {path}"
            if not dir_path.is_dir():
                return f"Error: Not a directory: {path}"

            items = []
            for item in sorted(dir_path.iterdir()):
                prefix = "📁 " if item.is_dir() else "📄 "
                items.append(f"{prefix}{item.name}")

            return f"Directory {path} is empty" if not items else "\n".join(items)
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error listing directory: {str(e)}"

    return read_file, write_file, edit_file, list_dir