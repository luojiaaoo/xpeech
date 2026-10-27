import json
from datetime import datetime

import pytest

from xpeech.agent.history import YamlHistoryRepository
from xpeech.agent.prompt.helper import build_user_prompt
from xpeech.agent.server.schema import InboundMessage, InputText
from xpeech.agent.tools.helper import as_tool
from xpeech.agent.tools.sessions import ListSessionsArgs, ReadSessionArgs, build_session_tools
from xpeech.exceptions import PathProtectionError


@pytest.mark.asyncio
async def test_lists_only_current_user_sessions_with_type_filters_and_pagination(tmp_path):
    repository = YamlHistoryRepository(tmp_path)
    history = [{"role": "user", "content": "hello"}]
    await repository.save("alice", "chat-a", history)
    await repository.save("alice", "chat~b", history)
    await repository.save("alice", "run-1", history, background=True)
    await repository.save("alice-other", "private", history)
    await repository.save("alice~session~other", "private", history)
    list_sessions, _ = build_session_tools(user_id="alice", history_path=tmp_path)

    all_sessions = json.loads(await list_sessions(ListSessionsArgs()))
    assert all_sessions["total"] == 3
    assert all_sessions["has_more"] is False
    assert {
        (session["session_id"], session["background"])
        for session in all_sessions["sessions"]
    } == {("chat-a", False), ("chat~b", False), ("run-1", True)}
    updated = [datetime.fromisoformat(session["updated_at"]) for session in all_sessions["sessions"]]
    assert updated == sorted(updated, reverse=True)

    first = json.loads(await list_sessions(ListSessionsArgs(limit=1)))
    rest = json.loads(await list_sessions(ListSessionsArgs(offset=1)))
    assert first["has_more"] is True
    assert first["sessions"] + rest["sessions"] == all_sessions["sessions"]
    background = json.loads(await list_sessions(ListSessionsArgs(background=True)))
    assert background["total"] == 1
    assert background["sessions"][0]["session_id"] == "run-1"
    normal = json.loads(await list_sessions(ListSessionsArgs(background=False)))
    assert normal["total"] == 2


@pytest.mark.asyncio
async def test_reads_history_pages_without_system_prompts_or_changing_saved_history(tmp_path):
    repository = YamlHistoryRepository(tmp_path)
    messages = [
        {"role": "system", "content": "internal system prompt"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]
    await repository.save("alice", "chat-1", messages)
    background = [{"role": "assistant", "content": "scheduled result"}]
    await repository.save("alice", "chat-1", background, background=True)
    _, read_session = build_session_tools(user_id="alice", history_path=tmp_path)

    page = json.loads(await read_session(ReadSessionArgs(session_id="chat-1", offset=1, limit=2)))
    assert page == {
        "session_id": "chat-1",
        "background": False,
        "total_messages": 4,
        "offset": 1,
        "messages": messages[2:4],
        "has_more": True,
    }
    last = json.loads(await read_session(ReadSessionArgs(session_id="chat-1", offset=3)))
    assert last["messages"] == messages[4:]
    assert last["has_more"] is False
    run = json.loads(await read_session(ReadSessionArgs(session_id="chat-1", background=True)))
    assert run["messages"] == background
    assert await repository.load("alice", "chat-1") == messages[1:]


@pytest.mark.asyncio
async def test_history_tools_cannot_read_another_user_or_follow_symlinks(tmp_path):
    repository = YamlHistoryRepository(tmp_path)
    messages = [{"role": "user", "content": "private"}]
    await repository.save("bob", "secret", messages)
    await repository.save("alice~session~other", "private", messages)
    (tmp_path / "alice~session~linked.yaml").symlink_to(tmp_path / "bob~session~secret.yaml")
    list_sessions, read_session = build_session_tools(user_id="alice", history_path=tmp_path)

    assert json.loads(await list_sessions(ListSessionsArgs()))["total"] == 0
    for session_id in ("secret", "linked", "other~session~private"):
        with pytest.raises(FileNotFoundError):
            await read_session(ReadSessionArgs(session_id=session_id, user_id="bob"))
    with pytest.raises(PathProtectionError):
        await read_session(ReadSessionArgs(session_id="../bob~session~secret"))


@pytest.mark.asyncio
async def test_history_tools_find_and_read_legacy_default_history(tmp_path):
    legacy_path = tmp_path / "alice.yaml"
    legacy_path.write_text("- role: user\n  content: old question\n", encoding="utf-8")
    list_sessions, read_session = build_session_tools(user_id="alice", history_path=tmp_path)

    sessions = json.loads(await list_sessions(ListSessionsArgs()))
    assert sessions["total"] == 1
    assert sessions["sessions"][0]["session_id"] == "default"
    result = json.loads(await read_session(ReadSessionArgs(session_id="default")))
    assert result["messages"] == [{"role": "user", "content": "old question"}]
    assert not legacy_path.exists()
    assert (tmp_path / "alice~session~default.yaml").is_file()


def test_session_tool_schemas_do_not_allow_selecting_a_user(tmp_path):
    tools = build_session_tools(user_id="alice", history_path=tmp_path)
    assert [tool.__name__ for tool in tools] == ["list_sessions", "read_session"]
    for tool in tools:
        properties = as_tool(tool)["function"]["parameters"]["properties"]
        assert "user_id" not in properties
        assert "offset" in properties
        assert "limit" in properties


@pytest.mark.asyncio
async def test_current_session_metadata_cannot_be_overridden_by_channel_metadata(tmp_path):
    message = InboundMessage(
        user_id="alice",
        session_id="current-chat",
        sender_name="Alice",
        session_metadata={"channel": "feishu", "user_id": "other", "session_id": "other-chat"},
        content=[InputText(text="continue our previous discussion")],
        timestamp=datetime(2026, 10, 27, 10, 20, 30),
        files=[],
    )

    prompt = await build_user_prompt(message, tmp_path)

    metadata = prompt["content"][0]["text"]
    assert "user_id: alice\n" in metadata
    assert "session_id: current-chat\n" in metadata
    assert "other-chat" not in metadata
    assert "channel: feishu\n" in metadata
