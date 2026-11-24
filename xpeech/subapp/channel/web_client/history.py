import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from ....agent.history import YamlHistoryRepository
from ....utils.file_io import LockedFile


class JsonlEventHistoryRepository(YamlHistoryRepository):
    """按用户和通道以 JSONL 追加保存 Web 对话的原始事件。"""

    _FILE_SUFFIX = ".jsonl"

    @staticmethod
    async def _load_path(path: Path) -> list[dict[str, Any]]:
        content = await LockedFile(path).read_text()
        try:
            events = [json.loads(line) for line in content.splitlines() if line.strip()]
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid web event history format: {path.name}") from error
        if any(not isinstance(event, dict) for event in events):
            raise ValueError(f"Invalid web event history format: {path.name}")
        logger.info("Web event history loaded events={}", len(events))
        return events

    def event_appender(
        self, user_id: str, session_id: str, *, channel: str,
    ) -> Callable[[dict[str, Any]], Awaitable[None]]:
        """返回逐条写入函数，在首次写入时定位文件并清理历史目录。"""
        file: LockedFile | None = None

        async def append_event(event: dict[str, Any]) -> None:
            nonlocal file
            if file is None:
                path = await self._history_path(user_id, session_id, channel=channel)
                await self._maintain_history_files(user_id=user_id, protected_session_id=session_id)
                file = LockedFile(path)
            await file.append_text(f"{json.dumps(event, ensure_ascii=False)}\n")

        return append_event

    async def list_channel_sessions(self, user_id: str, *, channel: str) -> list[dict[str, Any]]:
        """按更新时间列出指定通道的会话，并附带首条用户消息作为摘要。"""
        histories = await self._maintain_history_files(user_id=user_id)
        sessions = []
        for history in histories:
            if history.channel != channel:
                continue
            events = await self._load_path(history.path)
            preview = next(
                (
                    event.get("context", "")
                    for event in events
                    if event.get("event") == "user" and isinstance(event.get("context"), str)
                ),
                "",
            )
            sessions.append({
                "session_id": history.session_id,
                "channel": history.channel,
                "updated_at": datetime.fromtimestamp(history.modified_ns / 1_000_000_000, UTC).isoformat(),
                "size_bytes": history.size_bytes,
                "preview": preview,
            })
        return sorted(
            sessions,
            key=lambda item: (item["updated_at"], item["session_id"]),
            reverse=True,
        )
