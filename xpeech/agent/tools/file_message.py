from __future__ import annotations
import base64
import json
import aiofiles
from ...agent.skills.skill import BUILTIN_SKILLS_DIR
from pathlib import Path
from pydantic import BaseModel, Field
from ...utils.helper import is_relative_path, msys_to_win
from .filesystem import _IS_WINDOWS

# https://open.feishu.cn/document/server-docs/im-v1/message-content-description/create_json
# https://open.feishu.cn/document/server-docs/im-v1/file/create


class FilePathArgs(BaseModel):
    source: str = Field(description="local file path, for example a.pdf a.zip")


def build_file_message_tools(workspace: str | Path, restrict_tools_to_workspace: bool):
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

    def send_file(args: FilePathArgs) -> str:
        """Send a file to the user."""
        path = args.source
        file_path = safe_resolve(path, include_buildin_skills_path=True)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not file_path.is_file():
            raise ValueError(f"Not a file: {path}")
        return str(file_path)

    return send_file
