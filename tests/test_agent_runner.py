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
        self.use_history_values: list[bool] = []
        self.instances.append(self)

    async def run(self, message: InboundMessage, *, use_history: bool = True):
        self.messages.append(message)
        self.use_history_values.append(use_history)
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
    monkeypatch.setattr(runner_module, "register_default_tools", register_tools)
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
        session_id="session-1",
        sender_name="Alice",
        session_metadata={"channel": "test"},
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
    assert len(FakeAgentLoop.instances) == 1
    assert FakeAgentLoop.instances[0].messages == [message]
    assert FakeAgentLoop.instances[0].use_history_values == [True]

    provider = FakeAgentLoop.instances[0].kwargs["provider"]
    assert provider.kwargs["api_key"] == "test-key"
    assert provider.kwargs["extra_headers"] == {"Authorization": "Bearer test-key"}


@pytest.mark.asyncio
async def test_runner_rejects_message_for_another_session(tmp_path: Path):
    runner = AgentRunner("session-1", config=make_config(tmp_path))
    message = InboundMessage(
        session_id="session-2",
        sender_name="Alice",
        session_metadata={},
        content=[InputText(text="Hi")],
        timestamp="2026-09-05T10:00:00",
        files=[],
    )

    with pytest.raises(ValueError, match="session_id"):
        _ = [event async for event in runner.run(message)]


@pytest.mark.asyncio
async def test_runner_requires_async_factory(tmp_path: Path):
    runner = AgentRunner("session-1", config=make_config(tmp_path))
    message = InboundMessage(
        session_id="session-1",
        sender_name="Alice",
        session_metadata={},
        content=[InputText(text="Hi")],
        timestamp="2026-09-05T10:00:00",
        files=[],
    )

    with pytest.raises(RuntimeError, match=r"AgentRunner\.create\(\)"):
        _ = [event async for event in runner.run(message)]


@pytest.mark.asyncio
async def test_runner_can_disable_yaml_history(tmp_path: Path, runner_dependencies):
    runner = await AgentRunner.create("session-1", config=make_config(tmp_path))
    message = InboundMessage(
        session_id="session-1",
        sender_name="Scheduler",
        session_metadata={"source": "scheduled_task"},
        content=[InputText(text="Run scheduled prompt")],
        timestamp="2026-09-06T10:00:00",
        files=[],
    )

    _ = [event async for event in runner.run(message, use_history=False)]

    assert FakeAgentLoop.instances[0].use_history_values == [False]
