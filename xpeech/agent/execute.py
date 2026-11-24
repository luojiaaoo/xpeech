"""Streaming and non-interactive Agent execution inside the backend."""

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

import anyio
from fastapi import HTTPException, UploadFile, status
from openai import AsyncOpenAI
from uuid6 import uuid7

from ..exceptions import PathProtectionError
from ..utils.file_io import save_to_workspace
from ..utils.ids import generate_session_id, validate_session_id
from .runner import AgentRunner
from .server.chat_streams import CHAT_STREAMS
from .server.schema import InboundMessage, InputContent
from .server.session_guard import CHAT_GUARD


@contextmanager
def _workspace_http_errors():
    try:
        yield
    except PathProtectionError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from None


async def _prepare_message(
    user_id: str,
    *,
    content: InputContent,
    sender_name: str,
    channel: str,
    session_id: str,
    workspace: Path,
    session_metadata: dict[str, str] | None = None,
    files: list[UploadFile] | None = None,
    json_output: bool = False,
) -> InboundMessage:
    """Prepare a message, saving attachments before it can enter another request's loop."""
    message = InboundMessage(
        user_id=user_id,
        session_id=validate_session_id(session_id),
        sender_name=sender_name,
        channel=channel,
        session_metadata=session_metadata or {},
        content=content,
        files=[],
        json_output=json_output,
    )
    for file in files or ():
        message.files.append(await save_to_workspace(file=file, workspace=workspace))
    return message


def _user_event(
    message: InboundMessage, *, turn_id: str, files: list[UploadFile] | None,
) -> dict[str, Any]:
    text = [block.text.strip() for block in message.content]
    attachments = [f"📎 {file.filename or 'attachment'}" for file in files or ()]
    return {
        "event": "user",
        "context": "\n".join(part for part in (*text, *attachments) if part),
        "turn_id": turn_id,
    }


@asynccontextmanager
async def execute_chat_stream(
    user_id: str,
    *,
    openai_client: AsyncOpenAI,
    content: InputContent,
    sender_name: str,
    channel: str,
    session_id: str,
    session_metadata: dict[str, str] | None = None,
    files: list[UploadFile] | None = None,
    json_output: bool = False,
) -> AsyncIterator[AsyncGenerator[dict[str, Any], None]]:
    """Prepare the workspace before exposing the event stream to the caller."""
    with _workspace_http_errors():
        runner = await AgentRunner.create(
            user_id, openai_client=openai_client, not_interactive=False, channel=channel,
        )
        message = await _prepare_message(
            user_id,
            content=content,
            sender_name=sender_name,
            channel=channel,
            session_id=session_id,
            workspace=runner.workspace,
            session_metadata=session_metadata,
            files=files,
            json_output=json_output,
        )
    try:
        inbox, owner = await CHAT_GUARD.submit(message)
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None
    user_event = _user_event(message, turn_id=inbox.turn_id, files=files)
    if not owner:
        acknowledgement = "这条消息将用于引导后续回复"
        text = " ".join(" ".join(block.text for block in message.content).split())
        if text:
            preview = text[:40]
            if len(text) > len(preview):
                preview = f"{preview.rstrip()}…"
            acknowledgement += f"：「{preview}」"

        async def acknowledge():
            yield user_event
            yield {"event": "command", "context": acknowledgement, "turn_id": inbox.turn_id}

        async with aclosing(acknowledge()) as events:
            yield events
        return

    async def run():
        try:
            yield user_event
            async with aclosing(runner.run(message=message, inbox=inbox)) as events:
                async for event in events:
                    yield {**event, "turn_id": inbox.turn_id}
        finally:
            with anyio.CancelScope(shield=True):
                await CHAT_GUARD.release(user_id, session_id, inbox)

    try:
        events = CHAT_STREAMS.stream(user_id, session_id, run())
        try:
            yield events
        finally:
            with anyio.CancelScope(shield=True):
                await events.aclose()
    finally:
        # The producer may never start.
        with anyio.CancelScope(shield=True):
            await CHAT_GUARD.release(user_id, session_id, inbox)


async def execute_chat(
    user_id: str,
    *,
    openai_client: AsyncOpenAI,
    content: InputContent,
    sender_name: str,
    channel: str,
    session_metadata: dict[str, str] | None = None,
    files: list[UploadFile] | None = None,
    json_output: bool = False,
) -> list[dict[str, Any]]:
    """Run a non-interactive Agent call in a freshly generated session and return all events."""
    with _workspace_http_errors():
        runner = await AgentRunner.create(
            user_id, openai_client=openai_client, not_interactive=True, channel=channel,
        )
        message = await _prepare_message(
            user_id,
            content=content,
            sender_name=sender_name,
            channel=channel,
            session_id=generate_session_id(),
            workspace=runner.workspace,
            session_metadata=session_metadata,
            files=files,
            json_output=json_output,
        )
        turn_id = str(uuid7())
        user_event = _user_event(message, turn_id=turn_id, files=files)
        events = await runner.run_not_interactive(message)
        return [user_event, *({**event, "turn_id": turn_id} for event in events)]
