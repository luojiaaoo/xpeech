from pathlib import Path

import pytest

from xpeech.agent.history import YamlHistoryRepository, get_history_repository
from xpeech.exceptions import PathProtectionError


class TestYamlHistoryRepository:
    def test_reuses_repository_for_same_directory(self, tmp_path: Path):
        assert get_history_repository(tmp_path) is get_history_repository(tmp_path)

    @pytest.mark.asyncio
    async def test_save_and_load_history_removes_system_messages(self, tmp_path: Path):
        repository = YamlHistoryRepository(tmp_path)
        await repository.save(
            "session",
            [
                {"role": "system", "content": "hidden"},
                {"role": "user", "content": "hello"},
            ],
        )

        messages = await repository.load("session")

        assert messages == [{"role": "user", "content": "hello"}]
        assert not list(tmp_path.glob("*.tmp"))

    @pytest.mark.asyncio
    async def test_delete_history(self, tmp_path: Path):
        repository = YamlHistoryRepository(tmp_path)
        await repository.save("session", [{"role": "user", "content": "hello"}])

        await repository.delete("session")

        assert await repository.load("session") == []

    @pytest.mark.parametrize("session_id", ["", ".", "..", "../escape", r"..\\escape", "/absolute"])
    @pytest.mark.asyncio
    async def test_rejects_unsafe_session_ids(self, tmp_path: Path, session_id: str):
        repository = YamlHistoryRepository(tmp_path)

        with pytest.raises(PathProtectionError):
            await repository.load(session_id)
