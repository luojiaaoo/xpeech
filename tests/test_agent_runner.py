from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xpeech.agent import runner as runner_module
from xpeech.agent.runner import AgentRunner
from xpeech.agent.server.schema import InboundMessage, InputText


class FakeProvider:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeAgentLoop:
    instances: list["FakeAgentLoop"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.messages: list[InboundMessage] = []
        self.not_interactive_values: list[bool] = []
        self.instances.append(self)

    async def run(self, message: InboundMessage, *, not_interactive: bool = False):
        self.messages.append(message)
        self.not_interactive_values.append(not_interactive)
        yield {"event": "assistant", "context": "hello"}


def make_config(workspace_base_path: Path):
    return SimpleNamespace(
        path=SimpleNamespace(workspace_base_path=workspace_base_path),
        llm=SimpleNamespace(
            api_key="test-key",
            api_base="https://llm.test",
            default_model="test-model",
            parameters=object(),
            support_image=True,
            support_video=False,
            support_json_output=True,
            default_tools=["tool-a"],
            summary_tokens=100,
            max_iterations=5,
        ),
        tool=object(),
    )


@pytest.fixture
def runner_dependencies(monkeypatch: pytest.MonkeyPatch):
    FakeAgentLoop.instances.clear()
    ensure_path = AsyncMock(side_effect=lambda path: path)
    create_templates = AsyncMock()
    register_tools = AsyncMock()
    monkeypatch.setattr(runner_module, "ensure_path_async", ensure_path)
    monkeypatch.setattr(runner_module, "create_workspace_templates", create_templates)
    monkeypatch.setattr(runner_module, "LiteLLMProvider", FakeProvider)
    monkeypatch.setattr(runner_module, "AgentLoop", FakeAgentLoop)
    monkeypatch.setattr(runner_module, "register_tools", register_tools)
    return ensure_path, create_templates, register_tools


@pytest.mark.asyncio
async def test_runner_initializes_dependencies_once_and_yields_events(
    tmp_path: Path,
    runner_dependencies,
):
    ensure_path, create_templates, register_tools = runner_dependencies
    config = make_config(tmp_path)
    runner = await AgentRunner.create("session-1", config=config)
    message = InboundMessage(
        user_id="session-1",
        session_id="chat-1",
        sender_name="Alice",
        channel="test",
        session_metadata={},
        content=[InputText(text="Hi")],
        timestamp="2026-09-05T10:00:00",
        files=[],
    )

    assert [event async for event in runner.run(message)] == [
        {"event": "assistant", "context": "hello"}
    ]
    assert runner.workspace == (tmp_path / "session-1").resolve()
    ensure_path.assert_awaited_once_with(runner.workspace)
    create_templates.assert_awaited_once_with(runner.workspace)
    register_tools.assert_awaited_once()
    assert set(register_tools.await_args.kwargs) == {
        "user_id", "provider", "workspace", "config", "not_interactive", "channel_tags",
    }
    assert register_tools.await_args.kwargs["not_interactive"] is False
    assert register_tools.await_args.kwargs["channel_tags"] == []
    assert len(FakeAgentLoop.instances) == 1
    assert FakeAgentLoop.instances[0].messages == [message]
    assert FakeAgentLoop.instances[0].not_interactive_values == [False]

    provider = FakeAgentLoop.instances[0].kwargs["provider"]
    assert provider.kwargs["api_key"] == "test-key"
    assert provider.kwargs["extra_headers"] == {"Authorization": "Bearer test-key"}


@pytest.mark.asyncio
async def test_runner_rejects_message_for_another_session(tmp_path: Path):
    runner = AgentRunner("session-1", config=make_config(tmp_path))
    message = InboundMessage(
        user_id="session-2",
        session_id="chat-1",
        sender_name="Alice",
        channel="test",
        session_metadata={},
        content=[InputText(text="Hi")],
        timestamp="2026-09-05T10:00:00",
        files=[],
    )

    with pytest.raises(ValueError, match="user_id"):
        _ = [event async for event in runner.run(message)]


@pytest.mark.asyncio
async def test_runner_requires_async_factory(tmp_path: Path):
    runner = AgentRunner("session-1", config=make_config(tmp_path))
    message = InboundMessage(
        user_id="session-1",
        session_id="chat-1",
        sender_name="Alice",
        channel="test",
        session_metadata={},
        content=[InputText(text="Hi")],
        timestamp="2026-09-05T10:00:00",
        files=[],
    )

    with pytest.raises(RuntimeError, match=r"AgentRunner\.create\(\)"):
        _ = [event async for event in runner.run(message)]


@pytest.mark.asyncio
async def test_runner_can_run_without_interaction(tmp_path: Path, runner_dependencies):
    runner = await AgentRunner.create(
        "session-1", config=make_config(tmp_path), not_interactive=True, channel_tags=["crm"],
    )
    message = InboundMessage(
        user_id="session-1",
        session_id="chat-1",
        sender_name="Scheduler",
        channel="feishu",
        session_metadata={"source": "scheduled_task"},
        content=[InputText(text="Run scheduled prompt")],
        timestamp="2026-09-06T10:00:00",
        files=[],
    )

    _ = [event async for event in runner.run(message)]

    assert FakeAgentLoop.instances[0].not_interactive_values == [True]
    register_tools = runner_dependencies[2]
    assert register_tools.await_args.kwargs["not_interactive"] is True
    assert register_tools.await_args.kwargs["channel_tags"] == ["crm"]
    assert FakeAgentLoop.instances[0].kwargs["channel_tags"] == ["crm"]


@pytest.mark.asyncio
async def test_run_not_interactive_preserves_events_and_merges_text_chunks(tmp_path: Path):
    runner = AgentRunner("session-1", config=make_config(tmp_path), not_interactive=True)

    class EventLoop:
        async def run(self, message, *, not_interactive=False):
            assert not_interactive is True
            yield {"event": "assistant", "context": "draft"}
            yield {"event": "assistant_end", "context": ""}
            yield {"event": "tool_call", "context": "ignored"}
            yield {"event": "thinking", "context": "think "}
            yield {"event": "thinking", "context": "more"}
            yield {"event": "thinking_end", "context": ""}
            yield {"event": "question", "context": {"question": "Continue?"}}
            yield {"event": "send_file", "context": "report.txt"}
            yield {"event": "tool_call_result", "context": "result"}
            yield {"event": "assistant", "context": "final "}
            yield {"event": "assistant", "context": "answer"}
            yield {"event": "assistant_end", "context": ""}
            yield {"event": "token_usage", "context": "ignored"}

    runner._agent_loop = EventLoop()
    message = InboundMessage(
        user_id="session-1",
        session_id="chat-1",
        sender_name="Alice",
        channel="test",
        session_metadata={},
        content=[InputText(text="Hi")],
        timestamp="2026-09-05T10:00:00",
        files=[],
    )

    assert await runner.run_not_interactive(message) == [
        {"event": "assistant", "context": "draft"},
        {"event": "tool_call", "context": "ignored"},
        {"event": "thinking", "context": "think more"},
        {"event": "question", "context": {"question": "Continue?"}},
        {"event": "send_file", "context": "report.txt"},
        {"event": "tool_call_result", "context": "result"},
        {"event": "assistant", "context": "final answer"},
        {"event": "token_usage", "context": "ignored"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events, expected",
    [
        ([], []),
        ([{"event": "assistant_end", "context": ""}, {"event": "thinking_end", "context": ""}], []),
        (
            [
                {"event": "assistant", "context": "first"},
                {"event": "assistant_end", "context": ""},
                {"event": "assistant", "context": "second"},
            ],
            [{"event": "assistant", "context": "first"}, {"event": "assistant", "context": "second"}],
        ),
        ([{"event": "command", "context": "help"}], [{"event": "command", "context": "help"}]),
        ([{"event": "thinking", "context": "only thinking"}], [{"event": "thinking", "context": "only thinking"}]),
        (
            [{"event": "assistant", "context": ""}, {"event": "assistant", "context": "fallback"}],
            [{"event": "assistant", "context": "fallback"}],
        ),
        (
            [{"event": "assistant", "context": "before"}, {"event": "thinking", "context": "after"}],
            [{"event": "assistant", "context": "before"}, {"event": "thinking", "context": "after"}],
        ),
        (
            [{"event": "command", "context": "one"}, {"event": "command", "context": "two"}],
            [{"event": "command", "context": "one"}, {"event": "command", "context": "two"}],
        ),
        (
            [{"event": "assistant", "context": "one", "id": 1}, {"event": "assistant", "context": "two", "id": 2}],
            [{"event": "assistant", "context": "one", "id": 1}, {"event": "assistant", "context": "two", "id": 2}],
        ),
    ],
)
async def test_run_not_interactive_preserves_partial_and_non_assistant_output(tmp_path: Path, events, expected):
    runner = AgentRunner("session-1", config=make_config(tmp_path), not_interactive=True)

    class EventLoop:
        async def run(self, message, *, not_interactive=False):
            assert not_interactive is True
            for event in events:
                yield event

    runner._agent_loop = EventLoop()
    message = InboundMessage(
        user_id="session-1",
        session_id="chat-1",
        sender_name="Alice",
        channel="test",
        session_metadata={},
        content=[InputText(text="Hi")],
        timestamp="2026-09-05T10:00:00",
        files=[],
    )

    original_events = [event.copy() for event in events]
    assert await runner.run_not_interactive(message) == expected
    assert events == original_events
