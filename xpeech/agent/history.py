import asyncio
from dataclasses import dataclass
from datetime import UTC, date, datetime
from glob import escape
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from typing import Any, NoReturn
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
    user_id: str
    channel: str
    session_id: str
    not_interactive: bool
    modified_ns: int
    size_bytes: int


class YamlHistoryRepository:
    """以带有可解析元数据的 YAML 文件持久化会话历史。"""

    _FILE_KEYS = ("user_id", "channel", "type", "session_id")
    _SESSION_TYPE = "session"
    _NOT_INTERACTIVE_TYPE = "not_interactive"

    def __init__(self, base_dir: Path, max_files_per_user: int = 100) -> None:
        """初始化历史目录和每个用户的文件保留上限。"""
        if max_files_per_user < 1:
            raise ValueError("max_files_per_user must be at least 1")
        self._base_dir = ensure_path(base_dir.expanduser().resolve())
        self._max_files_per_user = max_files_per_user

    @classmethod
    def _history_filename(
        cls,
        user_id: str,
        session_id: str,
        *,
        channel: str,
        not_interactive: bool = False,
    ) -> str:
        """根据会话元数据生成规范的历史文件名。"""
        if not user_id or not session_id or not channel:
            raise PathProtectionError("History user_id, session_id, and channel must be non-empty")
        session_id = validate_session_id(session_id)
        history_type = cls._NOT_INTERACTIVE_TYPE if not_interactive else cls._SESSION_TYPE
        fields = (
            ("user_id", user_id),
            ("channel", channel),
            ("type", history_type),
            ("session_id", session_id),
        )
        return f"{urlencode(fields, quote_via=quote, safe='')}.yaml"

    def _history_path(
        self,
        user_id: str,
        session_id: str,
        *,
        channel: str,
        not_interactive: bool = False,
    ) -> Path:
        """生成普通会话或非交互任务的历史文件路径。"""
        filename = self._history_filename(
            user_id,
            session_id,
            channel=channel,
            not_interactive=not_interactive,
        )
        if len(filename.encode()) > 255:
            raise PathProtectionError("Session history filename is too long")
        return self._base_dir / filename

    def _maintain_history_files(
        self,
        *,
        user_id: str | None = None,
        protected_path: Path | None = None,
        protected_session: tuple[str, str] | None = None,
    ) -> list[_HistoryFile]:
        """删除非法文件，并按用户保留不超过配置上限的历史文件。"""

        def parse_history_path(path: Path, *, modified_ns: int, size_bytes: int) -> _HistoryFile:
            """解析历史文件名，并当场删除文件名不合规的文件。"""

            def delete_invalid_file() -> NoReturn:
                path.unlink(missing_ok=True)
                logger.info("Invalid session history file deleted path={}", path.name)
                raise FileNotFoundError(path)

            if path.suffix != ".yaml":
                delete_invalid_file()
            try:
                fields = parse_qsl(path.stem, keep_blank_values=True, strict_parsing=True)
            except ValueError:
                delete_invalid_file()
            if tuple(key for key, _value in fields) != self._FILE_KEYS:
                delete_invalid_file()
            values = dict(fields)
            history_type = values["type"]
            if (
                not values["user_id"]
                or not values["channel"]
                or not values["session_id"]
                or history_type not in {self._SESSION_TYPE, self._NOT_INTERACTIVE_TYPE}
            ):
                delete_invalid_file()
            try:
                session_id = validate_session_id(values["session_id"])
            except PathProtectionError:
                delete_invalid_file()
            not_interactive = history_type == self._NOT_INTERACTIVE_TYPE
            if path.name != self._history_filename(
                values["user_id"],
                session_id,
                channel=values["channel"],
                not_interactive=not_interactive,
            ):
                delete_invalid_file()
            return _HistoryFile(
                path=path,
                user_id=values["user_id"],
                channel=values["channel"],
                session_id=session_id,
                not_interactive=not_interactive,
                modified_ns=modified_ns,
                size_bytes=size_bytes,
            )

        histories: list[_HistoryFile] = []
        paths = self._base_dir.iterdir()
        if user_id is not None:
            encoded_user_id = escape(quote(user_id, safe=""))
            paths = self._base_dir.glob(f"user_id={encoded_user_id}&channel=*&type=*&session_id=*.yaml")
        for path in paths:
            try:
                stat = path.lstat()
            except FileNotFoundError:
                continue
            if S_ISDIR(stat.st_mode):
                continue
            if not S_ISREG(stat.st_mode):
                path.unlink(missing_ok=True)
                logger.info("Non-regular session history file deleted path={}", path.name)
                continue
            try:
                history = parse_history_path(
                    path,
                    modified_ns=stat.st_mtime_ns,
                    size_bytes=stat.st_size,
                )
            except FileNotFoundError:
                continue
            histories.append(history)

        if protected_path is None and protected_session is not None:
            protected_candidates = [
                history for history in histories if (history.user_id, history.session_id) == protected_session
            ]
            if protected_candidates:
                protected_path = max(
                    protected_candidates,
                    key=lambda item: (item.modified_ns, item.path.name),
                ).path

        by_user: dict[str, list[_HistoryFile]] = {}
        for history in histories:
            by_user.setdefault(history.user_id, []).append(history)

        retained: list[_HistoryFile] = []
        for user_histories in by_user.values():
            ordered = sorted(
                user_histories,
                key=lambda item: (item.modified_ns, item.path.name),
                reverse=True,
            )
            protected = next(
                (history for history in ordered if history.path == protected_path),
                None,
            )
            keep_count = self._max_files_per_user - int(protected is not None)
            keep = ([protected] if protected is not None else []) + [
                history for history in ordered if history is not protected
            ][:keep_count]
            keep_paths = {history.path for history in keep}
            retained.extend(keep)
            for history in ordered:
                if history.path in keep_paths:
                    continue
                try:
                    history.path.unlink()
                except FileNotFoundError:
                    continue
                logger.info("Expired session history file deleted path={}", history.path.name)
        return retained

    async def list_sessions(
        self,
        user_id: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict[str, Any]]:
        """按更新时间列出用户的普通会话和后台任务历史。"""

        def build_session_list() -> list[dict[str, Any]]:
            sessions = []
            for history in self._maintain_history_files(user_id=user_id):
                updated_at = datetime.fromtimestamp(history.modified_ns / 1_000_000_000, UTC)
                if start_date is not None and updated_at.date() < start_date:
                    continue
                if end_date is not None and updated_at.date() > end_date:
                    continue
                sessions.append(
                    {
                        "session_id": history.session_id,
                        "channel": history.channel,
                        "not_interactive": history.not_interactive,
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
                    item["not_interactive"],
                ),
                reverse=True,
            )

        return await asyncio.to_thread(build_session_list)

    async def load_session(self, user_id: str, session_id: str) -> list[dict[str, Any]]:
        """仅使用 list_sessions 返回的唯一会话 ID 读取历史。"""
        session_id = validate_session_id(session_id)
        histories = await asyncio.to_thread(
            self._maintain_history_files,
            protected_session=(user_id, session_id),
        )
        candidates = [
            history for history in histories if history.user_id == user_id and history.session_id == session_id
        ]
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
        not_interactive: bool = False,
    ) -> list[dict[str, Any]]:
        """加载历史、移除失效的系统提示词，并触发文件清理。"""
        path = self._history_path(
            user_id,
            session_id,
            channel=channel,
            not_interactive=not_interactive,
        )
        await asyncio.to_thread(self._maintain_history_files, protected_path=path)
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
        not_interactive: bool = False,
    ) -> None:
        """原子保存普通会话或非交互任务历史。"""
        path = self._history_path(
            user_id,
            session_id,
            channel=channel,
            not_interactive=not_interactive,
        )
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
        """删除普通会话历史，文件不存在时忽略。"""
        path = self._history_path(user_id, session_id, channel=channel)
        await asyncio.to_thread(path.unlink, missing_ok=True)
        logger.info("Session history deleted")
