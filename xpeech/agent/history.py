import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from ..exceptions import PathProtectionError
from ..utils.file_io import LockedFile, ensure_path
from ..utils.helper import LiteralDumper
from ..utils.ids import DEFAULT_SESSION_ID
from .prompt.helper import remove_system_messages


class YamlHistoryRepository:
    """以 YAML 文件持久化各会话的对话历史。"""

    SESSION_SEPARATOR = "~session~"
    BACKGROUND_SEPARATOR = "~background~"

    def __init__(self, base_dir: Path) -> None:
        """初始化历史目录。"""
        self._base_dir = ensure_path(base_dir.expanduser().resolve())

    def _history_path(
        self,
        user_id: str,
        history_id: str,
        *,
        background: bool = False,
    ) -> Path:
        """统一生成普通会话或后台任务的历史文件路径。"""
        separator = self.BACKGROUND_SEPARATOR if background else self.SESSION_SEPARATOR
        filename = f"{user_id}{separator}{history_id}.yaml"
        return self._resolve_history_path(filename)

    def _resolve_history_path(self, filename: str) -> Path:
        if len(filename.encode()) > 255:
            raise PathProtectionError("Session history filename is too long")
        path = (self._base_dir / filename).resolve()
        try:
            path.relative_to(self._base_dir)
        except ValueError:
            raise PathProtectionError("Session history path escapes its storage directory") from None
        return path

    async def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        """按更新时间倒序列出当前用户的普通会话和后台历史。"""
        def collect() -> list[dict[str, Any]]:
            sessions = []
            default_path = self._history_path(user_id, DEFAULT_SESSION_ID)
            for path in self._base_dir.iterdir():
                if path.suffix != ".yaml" or path.is_symlink() or not path.is_file():
                    continue
                if path.name == f"{user_id}.yaml" and not default_path.exists():
                    session_id, background = DEFAULT_SESSION_ID, False
                else:
                    stem = path.stem
                    separator = max(
                        (self.SESSION_SEPARATOR, self.BACKGROUND_SEPARATOR), key=stem.rfind
                    )
                    owner, found, session_id = stem.rpartition(separator)
                    if not found or owner != user_id:
                        continue
                    background = separator == self.BACKGROUND_SEPARATOR
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                sessions.append({
                    "session_id": session_id,
                    "background": background,
                    "updated_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                    "size_bytes": stat.st_size,
                })
            return sorted(
                sessions,
                key=lambda item: (item["updated_at"], item["session_id"], item["background"]),
                reverse=True,
            )

        return await asyncio.to_thread(collect)

    async def load(
        self,
        user_id: str,
        session_id: str,
        *,
        background: bool = False,
    ) -> list[dict[str, Any]]:
        """加载会话历史，并移除已失效的系统提示词。"""
        path = self._history_path(user_id, session_id, background=background)
        # TODO: 旧版 [user_id].yaml 全部迁移到 default 后，移除此兼容逻辑。
        if not background and session_id == DEFAULT_SESSION_ID and not path.exists():
            legacy_path = self._resolve_history_path(f"{user_id}.yaml")
            try:
                await asyncio.to_thread(os.link, legacy_path, path)
            except (FileNotFoundError, FileExistsError):
                pass
            else:
                await asyncio.to_thread(legacy_path.unlink, missing_ok=True)
        if not path.exists():
            return []
        content = yaml.safe_load(await LockedFile(path).read_text()) or []

        if not isinstance(content, list) or any(not isinstance(message, dict) for message in content):
            raise ValueError(f"Invalid session history format: {path.name}")

        messages = remove_system_messages(content)
        logger.info("Session history loaded messages={}", len(messages))
        return messages

    async def save(
        self,
        user_id: str,
        session_id: str,
        history: list[dict[str, Any]],
        *,
        background: bool = False,
    ) -> None:
        """以原子替换方式保存普通会话或后台任务的历史。"""
        path = self._history_path(user_id, session_id, background=background)
        serialized = yaml.dump(
            history,
            Dumper=LiteralDumper,
            default_flow_style=False,
            allow_unicode=True,
            indent=4,
            sort_keys=False,
            width=1000,
        )
        await LockedFile(path).write_text(serialized)
        logger.info("Session history saved")

    async def delete(self, user_id: str, session_id: str) -> None:
        """删除指定会话的历史文件；文件不存在时不报错。"""
        path = self._history_path(user_id, session_id)
        await asyncio.to_thread(path.unlink, missing_ok=True)
        # TODO: 旧版 YAML 兼容期结束后，移除此清理逻辑。
        if session_id == DEFAULT_SESSION_ID:
            legacy_path = self._resolve_history_path(f"{user_id}.yaml")
            await asyncio.to_thread(legacy_path.unlink, missing_ok=True)
        logger.info("Session history deleted")
