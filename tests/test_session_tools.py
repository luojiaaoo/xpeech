import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from xpeech.agent.history import YamlHistoryRepository
from xpeech.agent.prompt.helper import build_user_prompt
from xpeech.agent.server.schema import InboundMessage, InputText
from xpeech.agent.tool_executor import ToolExecutor
from xpeech.agent.tools.helper import as_tool
from xpeech.agent.tools.sessions import ListSessionsArgs, ReadSessionArgs, build_session_tools
from xpeech.exceptions import PathProtectionError
from xpeech.provider.schema import ToolCallRequest

SESSION_IDS = [f"018f0000-0000-7000-8000-{index:012x}" for index in range(1, 10)]


@pytest.mark.asyncio
async def test_lists_only_current_user_sessions_with_date_filters_and_pagination(tmp_path):
    repository = YamlHistoryRepository(tmp_path)
    history = [{"role": "user", "content": "hello"}]
    await repository.save("alice", SESSION_IDS[0], history, channel="feishu")
    await repository.save("alice", SESSION_IDS[1], history, channel="web")
    await repository.save("alice", SESSION_IDS[2], history, channel="feishu")
    await repository.save("alice-other", SESSION_IDS[3], history, channel="feishu")
    await repository.save("alice~session~other", SESSION_IDS[4], history, channel="feishu")
    day0, day1, day2 = (datetime.now(UTC).date() - timedelta(days=offset) for offset in (3, 2, 1))
    dates = {
        f"channel=feishu&session_id={SESSION_IDS[0]}.yaml": f"{day0}T23:59:59+00:00",
        f"channel=web&session_id={SESSION_IDS[1]}.yaml": f"{day1}T00:00:00+00:00",
        f"channel=feishu&session_id={SESSION_IDS[2]}.yaml": f"{day2}T23:59:59+00:00",
    }
    for filename, timestamp in dates.items():
        modified = datetime.fromisoformat(timestamp).timestamp()
        os.utime(tmp_path / "alice" / filename, (modified, modified))
    list_sessions, _ = build_session_tools(user_id="alice", history_path=tmp_path)

    all_sessions = json.loads(await list_sessions(ListSessionsArgs(), session_id="00000000-0000-7000-8000-000000000000"))
    assert all_sessions["total"] == 3
    assert all_sessions["has_more"] is False
    assert {session["session_id"] for session in all_sessions["sessions"]} == set(SESSION_IDS[:3])
    assert all("not_interactive" not in session for session in all_sessions["sessions"])
    assert {session["session_id"]: session["channel"] for session in all_sessions["sessions"]} == {
        SESSION_IDS[0]: "feishu",
        SESSION_IDS[1]: "web",
        SESSION_IDS[2]: "feishu",
    }
    updated = [datetime.fromisoformat(session["updated_at"]) for session in all_sessions["sessions"]]
    assert updated == sorted(updated, reverse=True)

    first = json.loads(await list_sessions(ListSessionsArgs(limit=1), session_id="00000000-0000-7000-8000-000000000000"))
    rest = json.loads(await list_sessions(ListSessionsArgs(offset=1), session_id="00000000-0000-7000-8000-000000000000"))
    assert first["has_more"] is True
    assert first["sessions"] + rest["sessions"] == all_sessions["sessions"]
    filtered = json.loads(await list_sessions(
        ListSessionsArgs(start_date=day1, end_date=day2), session_id="00000000-0000-7000-8000-000000000000",
    ))
    assert [session["session_id"] for session in filtered["sessions"]] == [SESSION_IDS[2], SESSION_IDS[1]]
    assert filtered["total"] == 2
    page = json.loads(await list_sessions(ListSessionsArgs(start_date=day1, limit=1), session_id="00000000-0000-7000-8000-000000000000"))
    assert page["total"] == 2
    assert page["has_more"] is True
    assert page["sessions"] == filtered["sessions"][:1]
    older = json.loads(await list_sessions(ListSessionsArgs(end_date=day0), session_id="00000000-0000-7000-8000-000000000000"))
    assert [session["session_id"] for session in older["sessions"]] == [SESSION_IDS[0]]
    empty = json.loads(await list_sessions(ListSessionsArgs(start_date=day2 + timedelta(days=1)), session_id="00000000-0000-7000-8000-000000000000"))
    assert empty["total"] == 0
    assert empty["sessions"] == []


def test_session_date_range_validation():
    ListSessionsArgs(start_date="2026-10-29", end_date="2026-10-29")
    with pytest.raises(ValidationError, match="start_date must be on or before end_date"):
        ListSessionsArgs(start_date="2026-10-30", end_date="2026-10-29")
    with pytest.raises(ValidationError):
        ListSessionsArgs(start_date="invalid-date")


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
    await repository.save("alice", SESSION_IDS[0], messages, channel="feishu")
    _list_sessions, read_session = build_session_tools(user_id="alice", history_path=tmp_path)
    scheduled_history = [{"role": "assistant", "content": "scheduled result"}]
    await repository.save("alice", SESSION_IDS[1], scheduled_history, channel="feishu")

    page = json.loads(await read_session(
        ReadSessionArgs(session_id=SESSION_IDS[0], offset=1, limit=2), session_id="00000000-0000-7000-8000-000000000000",
    ))
    assert page == {
        "session_id": SESSION_IDS[0],
        "total_messages": 4,
        "offset": 1,
        "messages": messages[2:4],
        "has_more": True,
    }
    last = json.loads(await read_session(ReadSessionArgs(session_id=SESSION_IDS[0], offset=3), session_id="00000000-0000-7000-8000-000000000000"))
    assert last["messages"] == messages[4:]
    assert last["has_more"] is False
    run = json.loads(await read_session(ReadSessionArgs(session_id=SESSION_IDS[1]), session_id="00000000-0000-7000-8000-000000000000"))
    assert run["messages"] == scheduled_history
    assert "not_interactive" not in run
    assert await repository.load("alice", SESSION_IDS[0], channel="feishu") == messages[1:]


@pytest.mark.asyncio
async def test_reads_uuid_session_by_session_id(tmp_path):
    repository = YamlHistoryRepository(tmp_path)
    session_id = SESSION_IDS[0]
    messages = [{"role": "assistant", "content": "result"}]
    await repository.save("alice", session_id, messages, channel="feishu")
    _, read_session = build_session_tools(user_id="alice", history_path=tmp_path)

    result = json.loads(await read_session(ReadSessionArgs(session_id=session_id), session_id="00000000-0000-7000-8000-000000000000"))
    assert result["messages"] == messages


@pytest.mark.asyncio
async def test_list_cleans_only_current_user_files(tmp_path, monkeypatch):
    repository = YamlHistoryRepository(tmp_path)
    await repository.save("alice[1]", SESSION_IDS[0], [], channel="feishu")
    await repository.save("alice1", SESSION_IDS[1], [], channel="feishu")
    await repository.save("bob", SESSION_IDS[2], [], channel="feishu")
    invalid_path = tmp_path / "notes.txt"
    invalid_path.write_text("invalid", encoding="utf-8")
    own_invalid_path = tmp_path / "alice%5B1%5D" / "notes.txt"
    own_invalid_path.write_text("invalid", encoding="utf-8")
    original_stat = Path.stat
    original_iterdir = Path.iterdir

    def check_iterdir(path):
        assert path == own_invalid_path.parent, "Only the current user directory may be scanned"
        return original_iterdir(path)

    def check_stat(path, *args, **kwargs):
        if path.name in {"alice1", "bob"} or path.parent.name in {"alice1", "bob"} or path == invalid_path:
            raise AssertionError("Unrelated files must not be statted")
        return original_stat(path, *args, **kwargs)

    list_sessions, _read_session = build_session_tools(user_id="alice[1]", history_path=tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", check_stat)
        patch.setattr(Path, "iterdir", check_iterdir)
        result = json.loads(await list_sessions(ListSessionsArgs(), session_id="00000000-0000-7000-8000-000000000000"))
    assert result["total"] == 1
    assert result["sessions"][0]["session_id"] == SESSION_IDS[0]
    assert result["sessions"][0]["channel"] == "feishu"
    assert invalid_path.exists()
    assert not own_invalid_path.exists()
    assert len(list(tmp_path.iterdir())) == 4
    assert len(list((tmp_path / "alice%5B1%5D").iterdir())) == 1
    with pytest.raises(ValidationError):
        ReadSessionArgs(session_id="*")


@pytest.mark.asyncio
async def test_history_tools_cannot_read_another_user(tmp_path):
    repository = YamlHistoryRepository(tmp_path)
    messages = [{"role": "user", "content": "private"}]
    await repository.save("bob", SESSION_IDS[0], messages, channel="feishu")
    await repository.save("alice~session~other", SESSION_IDS[1], messages, channel="feishu")
    list_sessions, read_session = build_session_tools(user_id="alice", history_path=tmp_path)
    list_bob_sessions, _ = build_session_tools(user_id="bob", history_path=tmp_path)
    [bob_history] = json.loads(await list_bob_sessions(ListSessionsArgs(), session_id="00000000-0000-7000-8000-000000000000"))["sessions"]

    assert json.loads(await list_sessions(ListSessionsArgs(), session_id="00000000-0000-7000-8000-000000000000"))["total"] == 0
    with pytest.raises(FileNotFoundError):
        await read_session(
            ReadSessionArgs(
                session_id=bob_history["session_id"],
                user_id="bob",
            ),
            session_id="00000000-0000-7000-8000-000000000000",
        )
    with pytest.raises(ValidationError):
        ReadSessionArgs(session_id="../bob~session~secret")


@pytest.mark.asyncio
async def test_history_tools_do_not_follow_symlinks(tmp_path):
    repository = YamlHistoryRepository(tmp_path)
    await repository.save("bob", SESSION_IDS[0], [{"role": "user", "content": "private"}], channel="feishu")
    (tmp_path / "alice").mkdir()
    try:
        (tmp_path / "alice" / f"channel=feishu&session_id={SESSION_IDS[1]}.yaml").symlink_to(
            tmp_path / "bob" / f"channel=feishu&session_id={SESSION_IDS[0]}.yaml"
        )
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows requires permission to create symbolic links")
        raise
    list_sessions, _ = build_session_tools(user_id="alice", history_path=tmp_path)
    assert json.loads(await list_sessions(ListSessionsArgs(), session_id="00000000-0000-7000-8000-000000000000"))["total"] == 0


@pytest.mark.asyncio
async def test_read_session_ignores_legacy_filenames(tmp_path):
    legacy_path = tmp_path / "alice.yaml"
    legacy_path.write_text("- role: user\n  content: old question\n", encoding="utf-8")
    pre_channel_path = tmp_path / "alice~session~default.yaml"
    pre_channel_path.write_text("- role: user\n  content: old question\n", encoding="utf-8")
    flat_path = tmp_path / f"user_id=alice&channel=web&type=session&session_id={SESSION_IDS[0]}.yaml"
    flat_path.write_text("- role: user\n  content: flat history\n", encoding="utf-8")
    list_sessions, read_session = build_session_tools(user_id="alice", history_path=tmp_path)

    assert json.loads(await list_sessions(ListSessionsArgs(), session_id="00000000-0000-7000-8000-000000000000"))["sessions"] == []
    with pytest.raises(FileNotFoundError):
        await read_session(ReadSessionArgs(session_id="00000000-0000-7000-8000-000000000000"), session_id=SESSION_IDS[0])
    assert legacy_path.exists()
    assert pre_channel_path.exists()
    assert flat_path.exists()

    repository = YamlHistoryRepository(tmp_path)
    current_messages = [{"role": "user", "content": "current question"}]
    await repository.save("alice", "00000000-0000-7000-8000-000000000000", current_messages, channel="feishu")
    sessions = json.loads(await list_sessions(ListSessionsArgs(), session_id=SESSION_IDS[0]))
    assert sessions["total"] == 1
    assert sessions["sessions"][0]["session_id"] == "00000000-0000-7000-8000-000000000000"
    assert sessions["sessions"][0]["channel"] == "feishu"
    result = json.loads(await read_session(ReadSessionArgs(session_id="00000000-0000-7000-8000-000000000000"), session_id=SESSION_IDS[0]))
    assert result["messages"] == current_messages


def test_session_tool_schemas_do_not_allow_selecting_a_user(tmp_path):
    tools = build_session_tools(user_id="alice", history_path=tmp_path)
    assert [tool.__name__ for tool in tools] == ["list_sessions", "read_session"]
    for tool in tools:
        properties = as_tool(tool)["function"]["parameters"]["properties"]
        assert "user_id" not in properties
        assert "not_interactive" not in properties
        assert "offset" in properties
        assert "limit" in properties
    list_properties = as_tool(tools[0])["function"]["parameters"]["properties"]
    assert {"start_date", "end_date"} <= list_properties.keys()
    read_properties = as_tool(tools[1])["function"]["parameters"]["properties"]
    assert "session_id" in read_properties
    assert "channel" not in read_properties
    assert "history_id" not in read_properties


@pytest.mark.asyncio
@pytest.mark.parametrize("current_session_id", ["00000000-0000-7000-8000-000000000000", SESSION_IDS[0]])
async def test_history_tools_exclude_current_session_and_reject_direct_reads(tmp_path, current_session_id):
    repository = YamlHistoryRepository(tmp_path)
    for channel in ("web", "feishu"):
        await repository.save("alice", current_session_id, [{"role": "user", "content": "current"}], channel=channel)
    await repository.save("alice", SESSION_IDS[1], [{"role": "assistant", "content": "other answer"}], channel="web")
    list_sessions, read_session = build_session_tools(user_id="alice", history_path=tmp_path)

    page = json.loads(await list_sessions(ListSessionsArgs(limit=1), session_id=current_session_id))
    assert page["total"] == 1
    assert page["has_more"] is False
    assert [session["session_id"] for session in page["sessions"]] == [SESSION_IDS[1]]
    with pytest.raises(PathProtectionError, match="current session"):
        await read_session(ReadSessionArgs(session_id=current_session_id), session_id=current_session_id)
    other = json.loads(await read_session(ReadSessionArgs(session_id=SESSION_IDS[1]), session_id=current_session_id))
    assert other["messages"] == [{"role": "assistant", "content": "other answer"}]


@pytest.mark.asyncio
async def test_executor_injects_current_session_id_and_ignores_spoofed_arguments(tmp_path):
    repository = YamlHistoryRepository(tmp_path)
    await repository.save("alice", SESSION_IDS[0], [{"role": "user", "content": "private current"}], channel="web")
    list_sessions, read_session = build_session_tools(user_id="alice", history_path=tmp_path)
    executor = ToolExecutor(
        workspace=tmp_path,
        max_result_chars=10_000,
        mapping_tool_call_funcs={"list_sessions": list_sessions, "read_session": read_session},
    )

    results = [
        result
        async for result in executor.execute(
            [
                ToolCallRequest(id="list", name="list_sessions", arguments={"session_id": SESSION_IDS[1]}),
                ToolCallRequest(id="read", name="read_session", arguments={"session_id": SESSION_IDS[0]}),
            ],
            session_id=SESSION_IDS[0],
        )
    ]
    results.sort(key=lambda result: result.call.id)
    assert results[0].succeeded
    assert json.loads(results[0].value)["total"] == 0
    assert not results[1].succeeded
    assert "current session" in results[1].value
    assert "private current" not in results[1].value
    assert "session_id" not in as_tool(list_sessions)["function"]["parameters"]["properties"]


@pytest.mark.asyncio
@pytest.mark.parametrize("json_output", [False, True])
async def test_current_session_metadata_cannot_be_overridden_by_channel_metadata(tmp_path, json_output):
    message = InboundMessage(
        user_id="alice",
        session_id="current-chat",
        sender_name="Alice",
        channel="feishu",
        session_metadata={"channel": "spoofed", "user_id": "other", "session_id": "other-chat"},
        content=[InputText(text="continue our previous discussion")],
        timestamp=datetime(2026, 10, 27, 10, 20, 30, tzinfo=UTC),
        files=[tmp_path / "example.csv"],
        json_output=json_output,
    )

    prompt = await build_user_prompt(message, tmp_path)

    metadata = prompt["content"][0]["text"]
    assert "user_id: alice\n" in metadata
    assert "session_id: current-chat\n" in metadata
    assert "other-chat" not in metadata
    assert "channel: feishu\n" in metadata
    assert "spoofed" not in metadata
    parts = prompt["content"]
    assert parts[1]["text"] == "continue our previous discussion"
    assert "example.csv" in parts[2]["text"]
    assert len(parts) == 3
    assert prompt == await build_user_prompt(message.model_copy(update={"json_output": not json_output}), tmp_path)
