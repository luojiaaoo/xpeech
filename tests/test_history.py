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
    return directory / user_id / (f"channel={channel}&type={history_type}&session_id={session_id}.yaml")


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

        paths = sorted((tmp_path / "alice").glob("channel=feishu&type=not_interactive&*.yaml"))
        assert [path.name for path in paths] == [
            f"channel=feishu&type=not_interactive&session_id={session_id}.yaml"
            for session_id in session_ids
        ]
        assert {yaml.safe_load(path.read_text(encoding="utf-8"))[0]["content"] for path in paths} == {
            "not_interactive a",
            "not_interactive b",
        }
        assert await repository.load("alice", "default", channel="feishu") == normal_history

    @pytest.mark.asyncio
    async def test_save_preserves_system_messages_and_load_excludes_them(self, tmp_path: Path):
        repository = YamlHistoryRepository(tmp_path)
        history = [
            {"role": "system", "content": "hidden"},
            {"role": "user", "content": "hello"},
        ]
        await repository.save(
            "session",
            SESSION_IDS[0],
            history,
            channel="web",
        )

        messages = await repository.load("session", SESSION_IDS[0], channel="web")

        assert messages == [{"role": "user", "content": "hello"}]
        assert history_path(tmp_path, "session", "web", "session", SESSION_IDS[0]).is_file()
        saved = yaml.safe_load(
            history_path(tmp_path, "session", "web", "session", SESSION_IDS[0]).read_text(encoding="utf-8")
        )
        assert saved == history

    @pytest.mark.asyncio
    async def test_load_filters_system_prompts_from_existing_files(self, tmp_path: Path):
        repository = YamlHistoryRepository(tmp_path)
        path = history_path(tmp_path, "alice", "web", "session", SESSION_IDS[0])
        path.parent.mkdir()
        path.write_text(
            yaml.safe_dump(
                [
                    {"role": "system", "content": "legacy system prompt"},
                    {"role": "user", "content": "hello"},
                    {"role": "system", "content": "another internal prompt"},
                    {"role": "assistant", "content": "answer"},
                ]
            ),
            encoding="utf-8",
        )

        assert await repository.load_session("alice", SESSION_IDS[0]) == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "answer"},
        ]

    @pytest.mark.asyncio
    async def test_retention_preserves_all_default_channels_and_active_session(self, tmp_path: Path):
        repository = YamlHistoryRepository(tmp_path, max_files_per_user=1)
        messages = [{"role": "user", "content": "hello"}]
        for channel in ("web", "feishu"):
            await repository.save("alice", "default", messages, channel=channel)
            os.utime(history_path(tmp_path, "alice", channel, "session", "default"), (1, 1))
        for session_id in SESSION_IDS[:2]:
            await repository.save("alice", session_id, messages, channel="web")

        assert await repository.load("alice", SESSION_IDS[0], channel="web") == messages
        for channel in ("web", "feishu"):
            assert history_path(tmp_path, "alice", channel, "session", "default").is_file()
        assert history_path(tmp_path, "alice", "web", "session", SESSION_IDS[0]).is_file()
        assert not history_path(tmp_path, "alice", "web", "session", SESSION_IDS[1]).exists()

    @pytest.mark.asyncio
    async def test_list_retention_preserves_current_session_in_all_channels(self, tmp_path: Path):
        repository = YamlHistoryRepository(tmp_path, max_files_per_user=1)
        for channel in ("web", "feishu"):
            await repository.save("alice", SESSION_IDS[0], [], channel=channel)
            os.utime(history_path(tmp_path, "alice", channel, "session", SESSION_IDS[0]), (1, 1))
        await repository.save("alice", SESSION_IDS[1], [], channel="web")

        assert await repository.list_sessions("alice", current_session_id=SESSION_IDS[0]) == []
        assert all(
            history_path(tmp_path, "alice", channel, "session", SESSION_IDS[0]).is_file()
            for channel in ("web", "feishu")
        )
        assert not history_path(tmp_path, "alice", "web", "session", SESSION_IDS[1]).exists()

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
    @pytest.mark.parametrize("user_id, directory", [
        ("alice", "alice"),
        ("a/b", "a%2Fb"),
        ("../bob", "%2E%2E%2Fbob"),
        ("..", "%2E%2E"),
    ])
    async def test_history_is_stored_inside_user_directory(self, tmp_path: Path, user_id: str, directory: str):
        repository = YamlHistoryRepository(tmp_path)
        messages = [{"role": "user", "content": "hello"}]

        await repository.save(user_id, SESSION_IDS[0], messages, channel="web")

        [user_dir] = tmp_path.iterdir()
        assert user_dir.name == directory
        [path] = user_dir.iterdir()
        assert path.name == f"channel=web&type=session&session_id={SESSION_IDS[0]}.yaml"
        assert await repository.load(user_id, SESSION_IDS[0], channel="web") == messages
        await repository.delete(user_id, SESSION_IDS[0], channel="web")
        assert not path.exists()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("operation", ["load", "list_sessions", "load_session"])
    async def test_cleans_only_current_user_files(self, tmp_path: Path, operation: str):
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
            tmp_path / "alice" / "notes.txt",
            tmp_path / "alice" / "channel=web&type=session.yaml",
            tmp_path / "alice" / "channel=web&type=session&session_id=chat-1.yaml",
        ]
        for invalid_path in invalid_paths:
            invalid_path.write_text("invalid", encoding="utf-8")
        unrelated_paths = [tmp_path / "notes.txt", tmp_path / "bob" / "notes.txt"]
        for unrelated_path in unrelated_paths:
            unrelated_path.write_text("unrelated", encoding="utf-8")
        nested = tmp_path / "alice" / "keep-directory"
        nested.mkdir()

        if operation == "load":
            assert await repository.load("alice", alice_ids[0], channel="web") == messages
        elif operation == "load_session":
            assert await repository.load_session("alice", alice_ids[0]) == messages
        else:
            sessions = await repository.list_sessions("alice", current_session_id=alice_ids[0])
            assert [session["session_id"] for session in sessions] == [alice_ids[2]]

        assert history_path(tmp_path, "alice", "web", "session", alice_ids[0]).is_file()
        assert not history_path(tmp_path, "alice", "web", "session", alice_ids[1]).exists()
        assert history_path(tmp_path, "alice", "web", "session", alice_ids[2]).is_file()
        assert history_path(tmp_path, "bob", "web", "session", bob_ids[0]).is_file()
        assert history_path(tmp_path, "bob", "web", "session", bob_ids[1]).is_file()
        assert history_path(tmp_path, "bob", "web", "session", bob_ids[2]).is_file()
        assert all(not path.exists() for path in invalid_paths)
        assert all(path.read_text(encoding="utf-8") == "unrelated" for path in unrelated_paths)
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
