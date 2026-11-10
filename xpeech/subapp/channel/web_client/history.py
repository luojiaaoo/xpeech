import asyncio
import json
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
    def _validate_events(content: Any, filename: str) -> list[dict[str, Any]]:
        if not isinstance(content, list) or any(not isinstance(event, dict) for event in content):
            raise ValueError(f"Invalid web event history format: {filename}")
        return content

    @staticmethod
    async def _load_path(path: Path) -> list[dict[str, Any]]:
        content = await LockedFile(path).read_text()
        try:
            events = [json.loads(line) for line in content.splitlines() if line.strip()]
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid web event history format: {path.name}") from error
        events = JsonlEventHistoryRepository._validate_events(events, path.name)
        logger.info("Web event history loaded events={}", len(events))
        return events

    @staticmethod
    def _serialize(events: list[dict[str, Any]]) -> str:
        return "".join(f"{json.dumps(event, ensure_ascii=False)}\n" for event in events)

    async def append(
        self,
        user_id: str,
        session_id: str,
        events: list[dict[str, Any]],
        *,
        channel: str,
    ) -> None:
        """逐行追加事件，避免续聊时读取并重写已有内容。"""
        if not events:
            return
        path = self._history_path(user_id, session_id, channel=channel)
        await asyncio.to_thread(
            self._maintain_history_files,
            user_id=user_id,
            protected_session_id=session_id,
        )
        await LockedFile(path).append_text(self._serialize(events))
        logger.info("Web event history appended events={}", len(events))

    async def list_channel_sessions(self, user_id: str, *, channel: str) -> list[dict[str, Any]]:
        """按更新时间列出指定通道的会话，并附带首条用户消息作为摘要。"""
        histories = await asyncio.to_thread(self._maintain_history_files, user_id=user_id)
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
