import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xpeech.agent import loop as loop_module
from xpeech.agent.compression import ConversationCompressor
from xpeech.agent.history import YamlHistoryRepository
from xpeech.agent.helper import strip_internal_message_metadata
from xpeech.agent.loop import AgentLoop
from xpeech.agent.memory import MemoryStore
from xpeech.agent.prompt.context import build_context_messages
from xpeech.agent.prompt.helper import is_system_message
from xpeech.agent.prompt.system import build_system_prompt
from xpeech.agent.server.schema import InboundMessage, InputText
from xpeech.agent.skills.skill import SkillsLoader
from xpeech.config.settings import settings
from xpeech.provider.openai_provider import OpenAIProvider
from xpeech.provider.schema import LLMParameters, LLMResponse
from xpeech.utils.helper import token_counter

pytestmark = pytest.mark.usefixtures("fake_vllm_tokenizer")


@pytest.mark.asyncio
async def test_context_breakdown_accumulates_blocks_and_leaves_messages_clean(tmp_path, monkeypatch, make_openai_sdk_client):
    monkeypatch.setattr(loop_module, "get_request_id", lambda: "test-request")
    monkeypatch.setattr(SkillsLoader, "build_skills_prompt", AsyncMock(return_value="available skills"))
    (tmp_path / "AGENTS.md").write_text("workspace rules", encoding="utf-8")
    (tmp_path / "SOUL.md").write_text("", encoding="utf-8")
    await MemoryStore(tmp_path).write_long_term("user preference")
    provider = OpenAIProvider(
        client=make_openai_sdk_client(), default_model="test", parameters=LLMParameters(),
    )
    agent = AgentLoop(channel="test", provider=provider, workspace=tmp_path)
    agent.history = SimpleNamespace(load=AsyncMock(return_value=[]), save=AsyncMock())
    agent.records = SimpleNamespace(append=AsyncMock())
    provider.chat = AsyncMock(side_effect=lambda **_: response("final answer " * 100))
    message = InboundMessage(
        user_id="alice", session_id="session", sender_name="Alice", channel="test",
        content=[InputText(text="hello")], files=[], session_metadata={},
    )

    events = [event async for event in agent.run(message)]
    details = json.loads(events[-1]["context"])["上下文明细"]
    parts = {name: part for name, part in details.items() if name != "剩余空间"}
    assert {"用户上下文", "长期记忆", "技能", "对话消息"} <= parts.keys()
    assert not {"身份", "工作区", "工作区指令", "人格设定", "用户信息", "工具说明", "备忘录目录"} & parts.keys()
    assert all(set(item) == {"role", "content"} for item in provider.chat.await_args.kwargs["messages"])
    saved = strip_internal_message_metadata(agent.history.save.await_args.args[2])
    total = sum(part["词元数"] for part in parts.values())
    assert sum(part["占比"] for part in parts.values()) == pytest.approx(total / agent.max_accept_token)
    history = [item for item in saved if item["role"] == "assistant" or isinstance(item["content"], list)]
    assert parts["系统提示词"]["词元数"] == await token_counter([saved[0]])
    assert parts["对话消息"]["词元数"] == await token_counter(history)
    assert "请求封装" not in parts
    assert "压缩预留" not in details
    assert "safety_margin" not in details
    assert total + details["剩余空间"]["词元数"] == agent.max_accept_token
    assert all(set(part) == {"词元数", "占比"} for part in details.values())


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

    system = await build_system_prompt(channel="web_client")
    token_counts = {}
    contexts = await build_context_messages(
        tmp_path, user_id="test-user", channel="web_client", user_organization=None, not_interactive=False, token_counts=token_counts,
    )

    assert len(contexts) == 9
    assert all(set(message) == {"role", "content"} and message["role"] == "user" for message in contexts)
    assert list(token_counts) == ["用户上下文", "长期记忆", "技能"]
    assert token_counts["用户上下文"] == sum([
        await token_counter([contexts[index]]) for index in (0, 1, 2, 3, 4, 5, 7)
    ])
    assert all(tokens > 0 for tokens in token_counts.values())
    assert all(
        message["content"].startswith("<system-reminder>\n")
        and message["content"].endswith("\n</system-reminder>")
        for message in contexts
    )
    assert "original identity" in contexts[0]["content"]
    assert tmp_path.as_posix() in contexts[1]["content"]
    for index, filename in enumerate(("AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"), start=2):
        assert f"## {filename}\n\ncontents of {filename}" in contexts[index]["content"]
        assert "original identity" not in contexts[index]["content"]
    assert "prefers concise replies" in contexts[6]["content"]
    for text in ("original identity", tmp_path.as_posix(), "contents of", "prefers concise replies", "skill catalog"):
        assert text not in system["content"]
    assert "You have access to tools that allow you to:" not in system["content"]

    (tmp_path / "AGENTS.md").write_text("updated rules", encoding="utf-8")
    await MemoryStore(tmp_path).write_long_term("updated preference")
    monkeypatch.setattr(settings.llm, "system_identity_prompt", "updated identity")
    assert await build_system_prompt(channel="web_client") == system
    updated = await build_context_messages(
        tmp_path, user_id="test-user", channel="web_client", user_organization=None, not_interactive=False, token_counts={},
    )
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
    agent.provider = SimpleNamespace(
        chat=AsyncMock(side_effect=lambda **_: response("answer")),
        default_context_token=200_000, default_model="test",
        chat_count_tool_tokens=AsyncMock(return_value={"system_tools": 0, "mcp_tools": 0}),
    )
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
    assert not any(is_system_message(item) for item in agent.records.append.await_args.kwargs["messages"])


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
    assert not any(is_system_message(item) for item in summary_input if item["role"] == "user")
    assert sum(item["role"] == "system" for item in summary_input) == 1
