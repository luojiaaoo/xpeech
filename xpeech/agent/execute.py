"""Streaming and non-interactive Agent execution inside the backend."""

from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any

from fastapi import UploadFile

from ..utils.file_io import save_to_workspace
from ..utils.ids import generate_session_id, validate_session_id
from .runner import AgentRunner
from .server.schema import InboundMessage, InputContent


async def _prepare_chat(
    user_id: str,
    *,
    content: InputContent,
    sender_name: str,
    channel: str,
    session_id: str,
    session_metadata: dict[str, str] | None = None,
    files: list[UploadFile] | None = None,
    json_output: bool = False,
    not_interactive: bool = False,
) -> tuple[AgentRunner, InboundMessage]:
    """Prepare a runner and inbound message, saving attachments in the workspace."""
    runner = await AgentRunner.create(user_id, not_interactive=not_interactive, channel=channel)
    saved_files = []
    for file in files or []:
        saved_files.append(await save_to_workspace(file=file, workspace=runner.workspace))

    message = InboundMessage(
        user_id=user_id,
        session_id=validate_session_id(session_id),
        sender_name=sender_name,
        channel=channel,
        session_metadata=session_metadata or {},
        content=content,
        files=saved_files,
        json_output=json_output,
    )
    return runner, message


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
) -> AsyncGenerator[dict[str, Any], None]:
    """Run an interactive Agent call and yield its events."""
    runner, message = await _prepare_chat(
        user_id,
        content=content,
        sender_name=sender_name,
        channel=channel,
        session_id=session_id,
        session_metadata=session_metadata,
        files=files,
        json_output=json_output,
    )
    async with aclosing(runner.run(message=message)) as events:
        async for event in events:
            yield event


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
    runner, message = await _prepare_chat(
        user_id,
        content=content,
        sender_name=sender_name,
        channel=channel,
        session_id=generate_session_id(),
        session_metadata=session_metadata,
        files=files,
        json_output=json_output,
        not_interactive=True,
    )
    return await runner.run_not_interactive(message)
