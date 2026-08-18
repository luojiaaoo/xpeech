from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, create_engine, select
from sqlmodel.ext.asyncio.session import AsyncSession

import xpeech.agent.loop as loop_module
from xpeech.agent.history import YamlHistoryRepository
from xpeech.agent.loop import AgentLoop
from xpeech.agent.record import (
    TABLE_NAME,
    ConversationRecord,
    SqliteConversationRecordRepository,
    create_db_and_tables,
)
from xpeech.agent.server.schema import InboundMessage, InputText
from xpeech.provider.schema import LLMResponse, ToolCallRequest


class TestSqliteConversationRecordRepository:
    @pytest.mark.asyncio
    async def test_appends_all_session_records_to_one_database(self, tmp_path: Path):
        database_path = tmp_path / "record.db"
        schema_engine = create_engine(f"sqlite:///{database_path.as_posix()}")
        engine = create_async_engine(f"sqlite+aiosqlite:///{database_path.as_posix()}")
        create_db_and_tables(schema_engine)
        schema_engine.dispose()
        repository = SqliteConversationRecordRepository(engine)
        record = ConversationRecord(
            session_id="session-1",
            sender_name="张三",
            user_question="你好",
            model_response="你好！",
            input_tokens=12,
            output_tokens=4,
            model_call_count=1,
        )
        second_record = ConversationRecord(
            session_id="session-2",
            sender_name="张三",
            user_question="再见",
            model_response="再见！",
            input_tokens=8,
            output_tokens=3,
            model_call_count=1,
        )

        await repository.append(record)
        await repository.append(second_record)

        assert database_path.is_file()
        try:
            async with engine.connect() as connection:
                columns = await connection.run_sync(
                    lambda sync_connection: tuple(
                        column["name"] for column in inspect(sync_connection).get_columns(TABLE_NAME)
                    )
                )
            async with AsyncSession(engine) as session:
                records = list(
                    (await session.exec(select(ConversationRecord).order_by(ConversationRecord.id))).all()
                )
        finally:
            await engine.dispose()
        assert columns == tuple(ConversationRecord.model_fields)
        assert ConversationRecord.metadata is not SQLModel.metadata
        assert records == [
            ConversationRecord(
                id=1,
                session_id="session-1",
                sender_name="张三",
                user_question="你好",
                model_response="你好！",
                input_tokens=12,
                output_tokens=4,
                model_call_count=1,
            ),
            ConversationRecord(
                id=2,
                session_id="session-2",
                sender_name="张三",
                user_question="再见",
                model_response="再见！",
                input_tokens=8,
                output_tokens=3,
                model_call_count=1,
            ),
        ]


@pytest.mark.asyncio
async def test_chat_warns_when_provider_does_not_return_token_usage(monkeypatch):
    class Provider:
        async def chat(self, **_kwargs):
            return LLMResponse(
                content="answer",
                usage={"prompt_tokens": 0, "completion_tokens": None, "total_tokens": None},
            )

    warnings = []
    monkeypatch.setattr(loop_module.logger, "warning", lambda message, *args: warnings.append((message, args)))

    agent_loop = AgentLoop.__new__(AgentLoop)
    agent_loop.provider = Provider()
    agent_loop._model_call_count = 0
    agent_loop._input_tokens = 0
    agent_loop._output_tokens = 0

    await agent_loop.chat([{"role": "user", "content": "question"}])

    assert warnings == [("Provider response missing completion_tokens", ())]
    assert agent_loop._input_tokens == 0
    assert agent_loop._output_tokens == 0


@pytest.mark.asyncio
async def test_agent_loop_records_final_response_and_aggregated_usage(tmp_path: Path, monkeypatch):
    async def fake_tool():
        return "tool result"

    tool_call = ToolCallRequest(id="call-1", name="fake_tool", arguments={})
    responses = iter(
        [
            LLMResponse(
                content="working",
                tool_calls=[tool_call],
                mapping_tool_call_funcs={"fake_tool": fake_tool},
                usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            ),
            LLMResponse(
                content="final answer",
                usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            ),
        ]
    )

    class Provider:
        default_context_token = 100_000

        async def chat(self, **_kwargs):
            return next(responses)

    async def build_test_system_prompt(workspace: Path):
        assert workspace == tmp_path / "workspace"
        return {"role": "system", "content": "system"}

    async def count_test_tokens(_messages):
        return 10

    monkeypatch.setattr(loop_module, "build_system_prompt", build_test_system_prompt)
    monkeypatch.setattr(loop_module, "token_counter", count_test_tokens)

    agent_loop = AgentLoop(
        provider=Provider(),
        workspace=tmp_path / "workspace",
        tools=[],
        summary_tokens=100,
        max_iterations=3,
    )
    agent_loop.history = YamlHistoryRepository(tmp_path / "history")
    database_path = tmp_path / "record.db"
    schema_engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path.as_posix()}")
    create_db_and_tables(schema_engine)
    schema_engine.dispose()
    agent_loop.records = SqliteConversationRecordRepository(engine)
    message = InboundMessage(
        session_id="session",
        sender_name="alice",
        session_metadata={},
        content=[InputText(text="first question"), InputText(text="second line")],
        timestamp=datetime.now(UTC),
        files=[],
    )

    events = [event async for event in agent_loop.run(message)]

    assert events[-2] == {"event": "assistant", "context": "final answer"}
    assert "\"大模型请求次数\": \"2\"" in events[-1]["context"]
    try:
        async with AsyncSession(engine) as session:
            records = list((await session.exec(select(ConversationRecord))).all())
    finally:
        await engine.dispose()
    assert records == [
        ConversationRecord(
            id=1,
            session_id="session",
            sender_name="alice",
            user_question="first question\nsecond line",
            model_response="final answer",
            input_tokens=30,
            output_tokens=7,
            model_call_count=2,
        )
    ]
