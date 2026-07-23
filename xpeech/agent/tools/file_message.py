from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel, Field
from .helper import safe_resolve_workspace_path

# https://open.feishu.cn/document/server-docs/im-v1/message-content-description/create_json
# https://open.feishu.cn/document/server-docs/im-v1/file/create


class FilePathArgs(BaseModel):
    source: str = Field(description="local file path, for example a.pdf a.zip")


def build_file_message_tools(workspace: str | Path):
    base = Path(workspace).expanduser().resolve()
    if not base.exists():
        raise ValueError(f"Invalid workspace: {workspace}")

    def safe_resolve(user_path: str) -> Path:
        return safe_resolve_workspace_path(
            user_path,
            base,
            protect_builtin_skills=False,
        )

    def send_file(args: FilePathArgs) -> str:
        """Send a file to the user."""
        path = args.source
        file_path = safe_resolve(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not file_path.is_file():
            raise ValueError(f"Not a file: {path}")
        return str(file_path)

    return send_file
