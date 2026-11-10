import asyncio
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode

import yaml
from loguru import logger

from ..exceptions import PathProtectionError
from ..utils.file_io import LockedFile, ensure_path
from ..utils.helper import LiteralDumper
from ..utils.ids import validate_session_id
from .prompt.helper import remove_system_messages


@dataclass(frozen=True)
class _HistoryFile:
    path: Path
    channel: str
    session_id: str
    modified_ns: int
    size_bytes: int


class YamlHistoryRepository:
    """以带有可解析元数据的 YAML 文件持久化会话历史。"""

    _FILE_KEYS = ("channel", "session_id")

    def __init__(self, base_dir: Path, max_history_days: int = 30) -> None:
        """初始化历史目录和历史文件保留天数。"""
        if max_history_days < 1:
            raise ValueError("max_history_days must be at least 1")
        self._base_dir = ensure_path(base_dir.resolve())
        self._max_history_days = max_history_days

    @classmethod
    def _history_filename(cls, session_id: str, *, channel: str) -> str:
        """根据会话元数据生成规范的历史文件名。"""
        if not session_id or not channel:
            raise PathProtectionError("History session_id and channel must be non-empty")
        session_id = validate_session_id(session_id)
        fields = (
            ("channel", channel),
            ("session_id", session_id),
        )
        return f"{urlencode(fields, quote_via=quote, safe='')}.yaml"

    def _user_history_dir(self, user_id: str) -> Path:
        """将用户 ID 编码为单层目录名，隔离各用户的历史文件。"""
        if not user_id:
            raise PathProtectionError("History user_id must be non-empty")
        directory = self._base_dir / quote(user_id, safe="").replace(".", "%2E")
        if directory.is_symlink() or directory.is_junction():
            raise PathProtectionError("History user directory must not be a link")
        return ensure_path(directory)

    def _history_path(self, user_id: str, session_id: str, *, channel: str) -> Path:
        """生成会话的历史文件路径。"""
        filename = self._history_filename(session_id, channel=channel)
        if len(filename.encode()) > 255:
            raise PathProtectionError("Session history filename is too long")
        return self._user_history_dir(user_id) / filename

    def _maintain_history_files(
        self,
        *,
        user_id: str,
        protected_session_id: str | None = None,
    ) -> list[_HistoryFile]:
        """清理当前用户的非法或超过保留天数未更新的历史文件，仅保护本次操作的会话。"""
        cutoff = (datetime.now(UTC) - timedelta(days=self._max_history_days)).timestamp()
        histories: list[_HistoryFile] = []
        for path in self._user_history_dir(user_id).iterdir():
            try:
                stat = path.lstat()
            except FileNotFoundError:
                continue
            if S_ISDIR(stat.st_mode):
                continue
            try:
                if not S_ISREG(stat.st_mode) or path.suffix != ".yaml":
                    raise ValueError("Not a regular YAML file")
                fields = parse_qsl(path.stem, keep_blank_values=True, strict_parsing=True)
                values = dict(fields)
                if tuple(key for key, _ in fields) != self._FILE_KEYS:
                    raise ValueError("Invalid history metadata")
                session_id = validate_session_id(values["session_id"])
                if path.name != self._history_filename(session_id, channel=values["channel"]):
                    raise ValueError("Non-canonical history filename")
            except (ValueError, PathProtectionError):
                path.unlink(missing_ok=True)
                logger.info("Invalid session history file deleted path={}", path.name)
                continue
            history = _HistoryFile(
                path=path,
                channel=values["channel"],
                session_id=session_id,
                modified_ns=stat.st_mtime_ns,
                size_bytes=stat.st_size,
            )
            if stat.st_mtime < cutoff and session_id != protected_session_id:
                history.path.unlink(missing_ok=True)
                logger.info("Expired session history file deleted path={}", history.path.name)
                continue
            histories.append(history)
        return histories

    async def list_sessions(
        self,
        user_id: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        current_session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """按更新时间列出用户的历史，提供当前会话 ID 时将其排除。"""

        def build_session_list() -> list[dict[str, Any]]:
            sessions = []
            for history in self._maintain_history_files(
                user_id=user_id,
                protected_session_id=current_session_id,
            ):
                if history.session_id == current_session_id:
                    continue
                updated_at = datetime.fromtimestamp(history.modified_ns / 1_000_000_000, UTC)
                if start_date is not None and updated_at.date() < start_date:
                    continue
                if end_date is not None and updated_at.date() > end_date:
                    continue
                sessions.append(
                    {
                        "session_id": history.session_id,
                        "channel": history.channel,
                        "updated_at": updated_at.isoformat(),
                        "size_bytes": history.size_bytes,
                    }
                )
            return sorted(
                sessions,
                key=lambda item: (
                    item["updated_at"],
                    item["session_id"],
                    item["channel"],
                ),
                reverse=True,
            )

        return await asyncio.to_thread(build_session_list)

    async def load_session(
        self,
        user_id: str,
        session_id: str,
        *,
        current_session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """仅使用 list_sessions 返回的唯一会话 ID 读取历史。"""
        session_id = validate_session_id(session_id)
        if session_id == current_session_id:
            raise PathProtectionError("Cannot read the current session history")
        histories = await asyncio.to_thread(
            self._maintain_history_files,
            user_id=user_id,
            protected_session_id=session_id,
        )
        candidates = [history for history in histories if history.session_id == session_id]
        if not candidates:
            raise FileNotFoundError("Session history not found for the current user")
        history = max(candidates, key=lambda item: (item.modified_ns, item.path.name))
        return await self._load_path(history.path)

    async def load(
        self,
        user_id: str,
        session_id: str,
        *,
        channel: str,
    ) -> list[dict[str, Any]]:
        """加载历史、移除失效的系统提示词，并触发文件清理。"""
        path = self._history_path(user_id, session_id, channel=channel)
        await asyncio.to_thread(self._maintain_history_files, user_id=user_id, protected_session_id=session_id)
        if not path.is_file():
            return []
        return await self._load_path(path)

    @staticmethod
    async def _load_path(path: Path) -> list[dict[str, Any]]:
        """读取并校验指定历史文件的内容。"""
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
        channel: str,
    ) -> None:
        """原子保存会话的完整历史。"""
        path = self._history_path(user_id, session_id, channel=channel)
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

    async def delete(self, user_id: str, session_id: str, *, channel: str) -> None:
        """删除会话历史，文件不存在时忽略。"""
        path = self._history_path(user_id, session_id, channel=channel)
        await asyncio.to_thread(path.unlink, missing_ok=True)
        logger.info("Session history deleted")
