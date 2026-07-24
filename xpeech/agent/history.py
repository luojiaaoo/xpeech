import asyncio
import os
from pathlib import Path
from typing import Any
from uuid import uuid4
from weakref import WeakValueDictionary

import aiofiles
import yaml
from loguru import logger

from ..exceptions import PathProtectionError
from ..utils.helper import LiteralDumper, ensure_path
from .prompt.helper import remove_system_messages

_REPOSITORIES: dict[Path, "YamlHistoryRepository"] = {}


class YamlHistoryRepository:
    """以 YAML 文件持久化各会话的对话历史。"""

    def __init__(self, base_dir: Path) -> None:
        """初始化历史目录及会话级异步锁。"""
        self._base_dir = ensure_path(base_dir.expanduser().resolve())
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

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

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        """获取或创建指定会话的文件访问锁。"""
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    async def load(self, session_id: str) -> list[dict[str, Any]]:
        """加载会话历史，并移除已失效的系统提示词。"""
        path = self._history_path(session_id)
        async with self._lock_for(session_id):
            if not path.exists():
                return []
            async with aiofiles.open(path, "r", encoding="utf-8") as file:
                content = yaml.safe_load(await file.read()) or []

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

        async with self._lock_for(session_id):
            try:
                async with aiofiles.open(temporary_path, "w", encoding="utf-8") as file:
                    await file.write(serialized)
                    await file.flush()
                await asyncio.to_thread(os.replace, temporary_path, path)
            finally:
                if temporary_path.exists():
                    await asyncio.to_thread(temporary_path.unlink)
        logger.info("Session history saved")

    async def delete(self, session_id: str) -> None:
        """删除指定会话的历史文件；文件不存在时不报错。"""
        path = self._history_path(session_id)
        async with self._lock_for(session_id):
            if path.exists():
                await asyncio.to_thread(path.unlink)
        logger.info("Session history deleted")


def get_history_repository(base_dir: Path) -> YamlHistoryRepository:
    """获取指定历史目录在当前进程内共享的仓库实例。"""
    resolved_base_dir = base_dir.expanduser().resolve()
    repository = _REPOSITORIES.get(resolved_base_dir)
    if repository is None:
        repository = YamlHistoryRepository(resolved_base_dir)
        _REPOSITORIES[resolved_base_dir] = repository
    return repository
