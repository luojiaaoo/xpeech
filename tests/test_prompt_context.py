from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xpeech.agent import loop as loop_module
from xpeech.agent.compression import ConversationCompressor
from xpeech.agent.history import YamlHistoryRepository
from xpeech.agent.helper import is_context_message
from xpeech.agent.loop import AgentLoop
from xpeech.agent.memory import MemoryStore
from xpeech.agent.prompt.context import build_context_messages
from xpeech.agent.prompt.system import build_system_prompt
from xpeech.agent.server.schema import InboundMessage, InputText
from xpeech.agent.skills.skill import SkillsLoader
from xpeech.config.settings import settings
from xpeech.provider.schema import LLMResponse


def response(content):
    async def chunks():
        yield "content", content

    return LLMResponse(iter_mix_chunks=chunks())


@pytest.mark.asyncio
async def test_dynamic_contexts_are_separate_and_do_not_change_system_prompt(tmp_path, monkeypatch):
    for filename in ("AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"):
        (tmp_path / filename).write_text(f"contents of {filename}", encoding="utf-8")
    await MemoryStore(tmp_path).write_long_term("prefers concise replies")
    monkeypatch.setattr(SkillsLoader, "build_skills_prompt", AsyncMock(return_value="skill catalog"))
    monkeypatch.setattr(settings.llm, "system_identity_prompt", "original identity")

    system = await build_system_prompt()
    contexts = await build_context_messages(tmp_path)

    assert len(contexts) == 9
    assert all(set(message) == {"role", "content"} and message["role"] == "user" for message in contexts)
    assert all(
        message["content"].startswith("<system-reminder>\n")
        and message["content"].endswith("\n</system-reminder>")
        for message in contexts
    )
    assert "original identity" in contexts[0]["content"]
    assert tmp_path.as_posix() in contexts[1]["content"]
    assert "contents of AGENTS.md" in contexts[2]["content"]
    assert "prefers concise replies" in contexts[6]["content"]
    for text in ("original identity", tmp_path.as_posix(), "contents of", "prefers concise replies", "skill catalog"):
        assert text not in system["content"]
    assert "You have access to tools that allow you to:" not in system["content"]

    (tmp_path / "AGENTS.md").write_text("updated rules", encoding="utf-8")
    await MemoryStore(tmp_path).write_long_term("updated preference")
    monkeypatch.setattr(settings.llm, "system_identity_prompt", "updated identity")
    assert await build_system_prompt() == system
    updated = await build_context_messages(tmp_path)
    assert "updated identity" in updated[0]["content"]
    assert "updated rules" in updated[2]["content"]
    assert "updated preference" in updated[6]["content"]


@pytest.mark.asyncio
async def test_run_replaces_front_contexts_without_accumulating_versions(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_module, "get_request_id", lambda: "test-request")
    monkeypatch.setattr(SkillsLoader, "build_skills_prompt", AsyncMock(return_value="skill catalog"))
    monkeypatch.setattr(AgentLoop, "_count_context_tokens", AsyncMock(return_value=10))
    monkeypatch.setattr(settings.llm, "system_identity_prompt", "old identity")
    (tmp_path / "AGENTS.md").write_text("old rules", encoding="utf-8")
    (tmp_path / "USER.md").write_text("old profile", encoding="utf-8")
    await MemoryStore(tmp_path).write_long_term("old memory")

    agent = AgentLoop.__new__(AgentLoop)
    agent.workspace = tmp_path
    agent.channel = "test"
    agent.user_organization = None
    agent.history = YamlHistoryRepository(tmp_path / "history")
    agent.records = SimpleNamespace(append=AsyncMock())
    agent.compressor = SimpleNamespace(should_compress=AsyncMock(return_value=False))
    agent.provider = SimpleNamespace(chat=AsyncMock(side_effect=lambda **_: response("answer")))
    agent.tools = []
    agent.max_iterations = 1
    agent.max_accept_token = 100_000
    agent._input_tokens = agent._output_tokens = agent._model_call_count = 0
    message = InboundMessage(
        user_id="alice", session_id="00000000-0000-7000-8000-000000000001",
        sender_name="Alice", channel="test", session_metadata={}, files=[],
        content=[InputText(text="first question")],
    )
    _ = [event async for event in agent.run(message)]
    first_request = agent.provider.chat.await_args.kwargs["messages"]

    (tmp_path / "AGENTS.md").write_text("new rules", encoding="utf-8")
    monkeypatch.setattr(settings.llm, "system_identity_prompt", "new identity")
    (tmp_path / "USER.md").unlink()
    await MemoryStore(tmp_path).write_long_term("")
    monkeypatch.setattr(SkillsLoader, "build_skills_prompt", AsyncMock(return_value=""))
    message.content = [InputText(text="second question")]
    _ = [event async for event in agent.run(message)]
    second_request = agent.provider.chat.await_args.kwargs["messages"]

    assert first_request[0] == second_request[0]
    assert second_request[0]["role"] == "system"
    assert all(set(item) == {"role", "content"} for item in second_request)
    assert all(
        item["role"] == "user" and item["content"].startswith("<system-reminder>\n")
        for item in second_request[1:5]
    )
    assert [item["role"] for item in second_request] == [
        "system", "user", "user", "user", "user", "user", "assistant", "user",
    ]
    assert "new identity" in second_request[1]["content"]
    assert "new rules" in second_request[3]["content"]
    for stale in ("old identity", "old rules", "old profile", "old memory", "skill catalog"):
        assert stale not in str(second_request)

    saved = await agent.history.load(message.user_id, message.session_id, channel=message.channel)
    assert [item["role"] for item in saved] == ["user", "assistant", "user", "assistant"]
    assert "first question" in str(saved[0]["content"])
    assert "second question" in str(saved[2]["content"])
    assert not any(is_context_message(item) for item in agent.records.append.await_args.kwargs["messages"])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["days", "turns", "fallback"])
async def test_compression_preserves_contexts_without_summarizing_them(mode, monkeypatch):
    monkeypatch.setattr("xpeech.agent.compression.SUMMARY_TOKENS", 1)
    async def count_tokens(*, messages):
        return sum(len(message["content"]) for message in messages)

    summarize = AsyncMock(side_effect=lambda **_: response("x"))
    prefix = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "<system-reminder>m</system-reminder>"},
        {"role": "user", "content": "<system-reminder>k</system-reminder>"},
    ]
    prefix_size = await count_tokens(messages=prefix)
    compressor = ConversationCompressor(
        chat=summarize, max_accept_tokens=prefix_size + 100, target_tokens=prefix_size + 7,
        token_counter=count_tokens,
    )
    history = [
        {"role": "user", "timestamp": 1, "content": "o" * 40},
        {"role": "assistant", "content": "a" * 10},
        {"role": "user", "timestamp": 10 * 86400 if mode == "days" else 2,
         "content": "n" * 40 if mode == "fallback" else "new"},
        {"role": "assistant", "content": "ok"},
    ]

    compressed = await compressor.compress(prefix + history)

    assert compressed[:3] == prefix
    assert compressed[3] == {"role": "assistant", "content": "x"}
    if mode != "fallback":
        assert compressed[4:] == history[-2:]
    else:
        assert len(compressed) == 4
    summary_input = summarize.await_args.kwargs["messages"]
    assert not any(is_context_message(item) for item in summary_input)
    assert sum(item["role"] == "system" for item in summary_input) == 1
