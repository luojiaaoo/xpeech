import asyncio
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from loguru import logger

from ..exceptions import PathProtectionError
from ..utils.helper import LiteralDumper, ensure_path, read_text_async, write_text_async
from .prompt.helper import remove_system_messages


class YamlHistoryRepository:
    """以 YAML 文件持久化各会话的对话历史。"""

    def __init__(self, base_dir: Path) -> None:
        """初始化历史目录。"""
        self._base_dir = ensure_path(base_dir.expanduser().resolve())

    def _history_path(self, session_id: str) -> Path:
        """校验会话 ID，并返回受保护的历史文件路径。"""
        if (
            not session_id
            or session_id in {".", ".."}
            or "/" in session_id
            or "\\" in session_id
            or "\0" in session_id
        ):
            raise PathProtectionError("Invalid session ID")

        path = (self._base_dir / f"{session_id}.yaml").resolve()
        try:
            path.relative_to(self._base_dir)
        except ValueError:
            raise PathProtectionError("Session history path escapes its storage directory") from None
        return path

    async def load(self, session_id: str) -> list[dict[str, Any]]:
        """加载会话历史，并移除已失效的系统提示词。"""
        path = self._history_path(session_id)
        if not path.exists():
            return []
        content = yaml.safe_load(await read_text_async(path)) or []

        if not isinstance(content, list) or any(not isinstance(message, dict) for message in content):
            raise ValueError(f"Invalid session history format: {path.name}")

        messages = remove_system_messages(content)
        logger.info("Session history loaded messages={}", len(messages))
        return messages

    async def save(self, session_id: str, history: list[dict[str, Any]]) -> None:
        """以原子替换方式保存会话历史。"""
        path = self._history_path(session_id)
        serialized = yaml.dump(
            history,
            Dumper=LiteralDumper,
            default_flow_style=False,
            allow_unicode=True,
            indent=4,
            sort_keys=False,
            width=1000,
        )
        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")

        try:
            await write_text_async(temporary_path, serialized)
            await asyncio.to_thread(os.replace, temporary_path, path)
        finally:
            if temporary_path.exists():
                await asyncio.to_thread(temporary_path.unlink)
        logger.info("Session history saved")

    async def delete(self, session_id: str) -> None:
        """删除指定会话的历史文件；文件不存在时不报错。"""
        path = self._history_path(session_id)
        if path.exists():
            await asyncio.to_thread(path.unlink)
        logger.info("Session history deleted")
