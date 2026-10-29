import asyncio
from pathlib import Path

import pytest
import yaml

from xpeech.agent.history import YamlHistoryRepository


class TestYamlHistoryRepository:
    @pytest.mark.asyncio
    async def test_not_interactive_runs_save_separate_files_without_overwriting(
        self,
        tmp_path: Path,
    ):
        repository = YamlHistoryRepository(tmp_path)
        normal_history = [{"role": "user", "content": "normal conversation"}]
        await repository.save("alice", "default", normal_history)
        session_ids = [
            "0123456789abcdef0123456789abcdef",
            "a1b2c3d4e5f60718293a4b5c6d7e8f90",
        ]
        histories = [
            [{"role": "user", "content": "not_interactive a"}],
            [{"role": "user", "content": "not_interactive b"}],
        ]

        await asyncio.gather(
            *(
                repository.save("alice", session_id, history, not_interactive=True)
                for session_id, history in zip(session_ids, histories)
            )
        )

        paths = sorted(tmp_path.glob("alice~not_interactive~*.yaml"))
        assert [path.name for path in paths] == [
            f"alice~not_interactive~{session_id}.yaml" for session_id in session_ids
        ]
        assert {
            yaml.safe_load(path.read_text(encoding="utf-8"))[0]["content"]
            for path in paths
        } == {"not_interactive a", "not_interactive b"}
        assert await repository.load("alice", "default") == normal_history
        assert not list(tmp_path.glob("*.tmp"))

    @pytest.mark.asyncio
    async def test_save_and_load_history_removes_system_messages(self, tmp_path: Path):
        repository = YamlHistoryRepository(tmp_path)
        await repository.save(
            "session",
            "chat-1",
            [
                {"role": "system", "content": "hidden"},
                {"role": "user", "content": "hello"},
            ],
        )

        messages = await repository.load("session", "chat-1")

        assert messages == [{"role": "user", "content": "hello"}]
        assert (tmp_path / "session~session~chat-1.yaml").is_file()
        assert not list(tmp_path.glob("*.tmp"))

    @pytest.mark.asyncio
    async def test_delete_history(self, tmp_path: Path):
        repository = YamlHistoryRepository(tmp_path)
        await repository.save(
            "session",
            "chat-1",
            [{"role": "user", "content": "hello"}],
        )

        await repository.delete("session", "chat-1")

        assert await repository.load("session", "chat-1") == []

    @pytest.mark.asyncio
    async def test_isolates_multiple_sessions_for_one_user(self, tmp_path: Path):
        repository = YamlHistoryRepository(tmp_path)
        await repository.save("alice", "chat-a", [{"role": "user", "content": "a"}])
        await repository.save("alice", "chat-b", [{"role": "user", "content": "b"}])

        assert await repository.load("alice", "chat-a") == [{"role": "user", "content": "a"}]
        assert await repository.load("alice", "chat-b") == [{"role": "user", "content": "b"}]

    @pytest.mark.asyncio
    async def test_migrates_legacy_history_only_to_default(self, tmp_path: Path):
        legacy_path = tmp_path / "alice.yaml"
        legacy_content = "- role: user\n  content: old\n"
        legacy_path.write_text(legacy_content, encoding="utf-8")
        repository = YamlHistoryRepository(tmp_path)

        assert await repository.load("alice", "chat-1") == []
        assert legacy_path.is_file()
        assert await repository.load("alice", "default", not_interactive=True) == []
        assert legacy_path.is_file()

        histories = await asyncio.gather(
            repository.load("alice", "default"),
            repository.load("alice", "default"),
        )

        assert histories == [[{"role": "user", "content": "old"}]] * 2
        assert not legacy_path.exists()
        assert (tmp_path / "alice~session~default.yaml").read_text(
            encoding="utf-8"
        ) == legacy_content
        assert await repository.load("alice", "default") == histories[0]

    @pytest.mark.asyncio
    async def test_legacy_history_does_not_overwrite_existing_default(self, tmp_path: Path):
        repository = YamlHistoryRepository(tmp_path)
        current_history = [{"role": "user", "content": "current"}]
        await repository.save("alice", "default", current_history)
        legacy_path = tmp_path / "alice.yaml"
        legacy_content = "- role: user\n  content: old\n"
        legacy_path.write_text(legacy_content, encoding="utf-8")

        assert await repository.load("alice", "default") == current_history
        assert legacy_path.read_text(encoding="utf-8") == legacy_content

    @pytest.mark.asyncio
    @pytest.mark.parametrize("has_default", [False, True])
    async def test_delete_default_also_clears_unmigrated_history(
        self, tmp_path: Path, has_default: bool
    ):
        repository = YamlHistoryRepository(tmp_path)
        if has_default:
            await repository.save("alice", "default", [{"role": "user", "content": "current"}])
        legacy_path = tmp_path / "alice.yaml"
        legacy_path.write_text("- role: user\n  content: old\n", encoding="utf-8")

        await repository.delete("alice", "chat-1")
        assert legacy_path.is_file()
        await repository.delete("alice", "default")

        assert not legacy_path.exists()
        assert await repository.load("alice", "default") == []
