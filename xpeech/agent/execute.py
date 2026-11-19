"""Streaming and non-interactive Agent execution inside the backend."""

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

import anyio
from fastapi import HTTPException, UploadFile, status

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


@asynccontextmanager
async def execute_chat_stream(
    user_id: str,
    *,
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
        runner = await AgentRunner.create(user_id, not_interactive=False, channel=channel)
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
    if not owner:
        async def acknowledge():
            yield {"event": "command", "context": "这条消息将用于引导后续回复"}

        async with aclosing(acknowledge()) as events:
            yield events
        return

    async def run():
        try:
            async with aclosing(runner.run(message=message, inbox=inbox)) as events:
                async for event in events:
                    yield event
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
    content: InputContent,
    sender_name: str,
    channel: str,
    session_metadata: dict[str, str] | None = None,
    files: list[UploadFile] | None = None,
    json_output: bool = False,
) -> list[dict[str, Any]]:
    """Run a non-interactive Agent call in a freshly generated session and return all events."""
    with _workspace_http_errors():
        runner = await AgentRunner.create(user_id, not_interactive=True, channel=channel)
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
        return await runner.run_not_interactive(message)
