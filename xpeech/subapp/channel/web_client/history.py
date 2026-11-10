import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from ....agent.history import YamlHistoryRepository
from ....utils.file_io import LockedFile
from ....utils.helper import LiteralDumper


class YamlEventHistoryRepository(YamlHistoryRepository):
    """按用户和通道保存 Web 对话的原始事件。"""

    @staticmethod
    def _validate_events(content: Any, filename: str) -> list[dict[str, Any]]:
        if not isinstance(content, list) or any(not isinstance(event, dict) for event in content):
            raise ValueError(f"Invalid web event history format: {filename}")
        return content

    @staticmethod
    async def _load_path(path: Path) -> list[dict[str, Any]]:
        content = yaml.safe_load(await LockedFile(path).read_text()) or []
        events = YamlEventHistoryRepository._validate_events(content, path.name)
        logger.info("Web event history loaded events={}", len(events))
        return events

    @staticmethod
    def _serialize(events: list[dict[str, Any]]) -> str:
        return yaml.dump(
            events,
            Dumper=LiteralDumper,
            default_flow_style=False,
            allow_unicode=True,
            indent=4,
            sort_keys=False,
            width=1000,
        )

    async def append(
        self,
        user_id: str,
        session_id: str,
        events: list[dict[str, Any]],
        *,
        channel: str,
    ) -> None:
        """在同一路径锁内读取并追加事件，避免续聊时覆盖已有内容。"""
        if not events:
            return
        path = self._history_path(user_id, session_id, channel=channel)
        await asyncio.to_thread(
            self._maintain_history_files,
            user_id=user_id,
            protected_session_id=session_id,
        )
        locked = LockedFile(path)
        async with locked.transaction():
            try:
                content = yaml.safe_load(await locked.read_text()) or []
            except FileNotFoundError:
                content = []
            current = self._validate_events(content, path.name)
            await locked.write_text(self._serialize([*current, *events]))
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
