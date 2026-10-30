import asyncio
import os
from pathlib import Path

import pytest
import yaml

from xpeech.agent.history import YamlHistoryRepository
from xpeech.exceptions import PathProtectionError

SESSION_IDS = [f"018f0000-0000-7000-8000-{index:012x}" for index in range(1, 10)]


def history_path(
    directory: Path,
    user_id: str,
    channel: str,
    history_type: str,
    session_id: str,
) -> Path:
    return directory / (f"user_id={user_id}&channel={channel}&type={history_type}&session_id={session_id}.yaml")


class TestYamlHistoryRepository:
    @pytest.mark.asyncio
    async def test_not_interactive_runs_save_separate_files_without_overwriting(self, tmp_path: Path):
        repository = YamlHistoryRepository(tmp_path)
        normal_history = [{"role": "user", "content": "normal conversation"}]
        await repository.save("alice", "default", normal_history, channel="feishu")
        session_ids = SESSION_IDS[:2]
        histories = [
            [{"role": "user", "content": "not_interactive a"}],
            [{"role": "user", "content": "not_interactive b"}],
        ]

        await asyncio.gather(
            *(
                repository.save(
                    "alice",
                    session_id,
                    history,
                    channel="feishu",
                    not_interactive=True,
                )
                for session_id, history in zip(session_ids, histories)
            )
        )

        paths = sorted(tmp_path.glob("user_id=alice&channel=feishu&type=not_interactive&*.yaml"))
        assert [path.name for path in paths] == [
            f"user_id=alice&channel=feishu&type=not_interactive&session_id={session_id}.yaml"
            for session_id in session_ids
        ]
        assert {yaml.safe_load(path.read_text(encoding="utf-8"))[0]["content"] for path in paths} == {
            "not_interactive a",
            "not_interactive b",
        }
        assert await repository.load("alice", "default", channel="feishu") == normal_history

    @pytest.mark.asyncio
    async def test_save_and_load_history_removes_system_messages(self, tmp_path: Path):
        repository = YamlHistoryRepository(tmp_path)
        await repository.save(
            "session",
            SESSION_IDS[0],
            [
                {"role": "system", "content": "hidden"},
                {"role": "user", "content": "hello"},
            ],
            channel="web",
        )

        messages = await repository.load("session", SESSION_IDS[0], channel="web")

        assert messages == [{"role": "user", "content": "hello"}]
        assert history_path(tmp_path, "session", "web", "session", SESSION_IDS[0]).is_file()

    @pytest.mark.asyncio
    async def test_channel_is_stored_in_filename_metadata(self, tmp_path: Path):
        repository = YamlHistoryRepository(tmp_path)
        await repository.save(
            "alice",
            SESSION_IDS[0],
            [{"role": "user", "content": "feishu chat"}],
            channel="feishu",
        )
        await repository.save(
            "alice",
            SESSION_IDS[1],
            [{"role": "user", "content": "web chat"}],
            channel="web",
        )

        assert history_path(tmp_path, "alice", "feishu", "session", SESSION_IDS[0]).is_file()
        assert history_path(tmp_path, "alice", "web", "session", SESSION_IDS[1]).is_file()
        assert await repository.load("alice", SESSION_IDS[0], channel="feishu") == [
            {"role": "user", "content": "feishu chat"}
        ]
        assert await repository.load("alice", SESSION_IDS[1], channel="web") == [
            {"role": "user", "content": "web chat"}
        ]

    @pytest.mark.asyncio
    async def test_delete_history(self, tmp_path: Path):
        repository = YamlHistoryRepository(tmp_path)
        await repository.save(
            "session",
            SESSION_IDS[0],
            [{"role": "user", "content": "hello"}],
            channel="web",
        )

        await repository.delete("session", SESSION_IDS[0], channel="web")

        assert await repository.load("session", SESSION_IDS[0], channel="web") == []

    @pytest.mark.asyncio
    async def test_filename_metadata_is_encoded_and_round_trips(self, tmp_path: Path):
        repository = YamlHistoryRepository(tmp_path)
        messages = [{"role": "user", "content": "encoded metadata"}]

        await repository.save("a&b", SESSION_IDS[0], messages, channel="custom/channel")

        assert history_path(
            tmp_path,
            "a%26b",
            "custom%2Fchannel",
            "session",
            SESSION_IDS[0],
        ).is_file()
        assert await repository.load("a&b", SESSION_IDS[0], channel="custom/channel") == messages
        [listed] = await repository.list_sessions("a&b")
        assert listed["session_id"] == SESSION_IDS[0]
        assert listed["channel"] == "custom/channel"

    @pytest.mark.asyncio
    async def test_load_deletes_invalid_files_and_prunes_each_user(self, tmp_path: Path):
        repository = YamlHistoryRepository(tmp_path, max_files_per_user=2)
        messages = [{"role": "user", "content": "hello"}]
        alice_ids = SESSION_IDS[:3]
        bob_ids = SESSION_IDS[3:6]
        for session_id in alice_ids:
            await repository.save("alice", session_id, messages, channel="web")
        for session_id in bob_ids:
            await repository.save("bob", session_id, messages, channel="web")

        timestamps = {
            ("alice", alice_ids[0]): 1,
            ("alice", alice_ids[1]): 2,
            ("alice", alice_ids[2]): 3,
            ("bob", bob_ids[0]): 1,
            ("bob", bob_ids[1]): 2,
            ("bob", bob_ids[2]): 3,
        }
        for (user_id, session_id), modified in timestamps.items():
            os.utime(history_path(tmp_path, user_id, "web", "session", session_id), (modified, modified))

        invalid_paths = [
            tmp_path / "alice.yaml",
            tmp_path / "notes.txt",
            tmp_path / "user_id=alice&channel=web&type=session.yaml",
            tmp_path / "user_id=alice&channel=web&type=session&session_id=chat-1.yaml",
        ]
        for invalid_path in invalid_paths:
            invalid_path.write_text("invalid", encoding="utf-8")
        nested = tmp_path / "keep-directory"
        nested.mkdir()

        assert await repository.load("alice", alice_ids[0], channel="web") == messages

        assert history_path(tmp_path, "alice", "web", "session", alice_ids[0]).is_file()
        assert not history_path(tmp_path, "alice", "web", "session", alice_ids[1]).exists()
        assert history_path(tmp_path, "alice", "web", "session", alice_ids[2]).is_file()
        assert not history_path(tmp_path, "bob", "web", "session", bob_ids[0]).exists()
        assert history_path(tmp_path, "bob", "web", "session", bob_ids[1]).is_file()
        assert history_path(tmp_path, "bob", "web", "session", bob_ids[2]).is_file()
        assert all(not path.exists() for path in invalid_paths)
        assert nested.is_dir()

    @pytest.mark.asyncio
    async def test_load_session_needs_only_unique_session_id(self, tmp_path: Path):
        repository = YamlHistoryRepository(tmp_path)
        await repository.save(
            "alice",
            SESSION_IDS[0],
            [{"role": "user", "content": "hello"}],
            channel="feishu",
        )

        assert await repository.load_session("alice", SESSION_IDS[0]) == [{"role": "user", "content": "hello"}]

    @pytest.mark.asyncio
    async def test_preserves_default_and_rejects_non_uuidv7_session_ids(self, tmp_path: Path):
        repository = YamlHistoryRepository(tmp_path)
        messages = [{"role": "user", "content": "main session"}]

        await repository.save("alice", "default", messages, channel="web")
        assert await repository.load("alice", "default", channel="web") == messages

        for invalid in ("chat-1", "01234567-89ab-4def-8123-456789abcdef"):
            with pytest.raises(PathProtectionError, match="UUIDv7"):
                await repository.save("alice", invalid, messages, channel="web")

    def test_rejects_invalid_retention_limit(self, tmp_path: Path):
        with pytest.raises(ValueError, match="at least 1"):
            YamlHistoryRepository(tmp_path, max_files_per_user=0)
