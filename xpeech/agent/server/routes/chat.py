"""Channel-facing chat, question-answer, and background-message endpoints."""

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile
from fastapi.sse import EventSourceResponse

from ...execute import execute_chat, execute_chat_stream
from ...loop import AgentLoop, QuestionEvent
from ...tools.schedule.client import scheduler_request
from ..auth import AuthenticatedChannel, require_channel_jwt
from ..dependencies import (
    acquire_chat_session,
    content,
    json_output_enabled,
    sender_name_header,
    session_id_header,
    session_metadata,
    user_id_header,
)
from ..schema import (
    BackgroundMessageChannel,
    FeishuBackgroundMessage,
    InputContent,
)

router = APIRouter()


@router.get("/background_message", response_model=FeishuBackgroundMessage)
async def poll_background_message(channel: BackgroundMessageChannel) -> FeishuBackgroundMessage | Response:
    """Any backend can long-poll the shared scheduler for a channel message."""
    payload = await scheduler_request(
        "GET", f"/background_message?channel={channel.value}", timeout=60,
    )
    if payload is None:
        return Response(status_code=204)
    return FeishuBackgroundMessage.model_validate(payload)


@router.post("/answer_question")
async def answer_question(
    user_id: Annotated[str, Depends(user_id_header)],
    session_id: Annotated[str, Depends(session_id_header)],
    answer: Annotated[str, Form(description="回答内容")],
):
    """提交用户答案并唤醒正在等待的 Agent 循环。"""
    question_event: QuestionEvent | None = AgentLoop.SESSION_QUESTION_EVENT.get(
        (user_id, session_id)
    )
    if question_event is not None:
        question_event.answer = answer
        question_event.event.set()
        return {"message": "Answer received"}
    return {"message": "Question event not found"}


@router.post(
    "/chat",
    response_class=EventSourceResponse,
    responses={400: {"description": "The configured model does not support the requested JSON output mode"}},
    dependencies=[Depends(acquire_chat_session, scope="request")],
)
async def chat(
    channel: Annotated[AuthenticatedChannel, Depends(require_channel_jwt)],
    user_id: Annotated[str, Depends(user_id_header)],
    session_id: Annotated[str, Depends(session_id_header)],
    sender_name: Annotated[str, Depends(sender_name_header)],
    session_metadata: Annotated[dict[str, str], Depends(session_metadata)],
    content: Annotated[InputContent, Depends(content)],
    timestamp: Annotated[datetime, Form(default_factory=datetime.now, description="消息的时间戳")],
    files: Annotated[list[UploadFile], File(default_factory=list, description="消息附件列表")],
    json_output: Annotated[bool, Depends(json_output_enabled)],
):
    """接收用户消息，并以 SSE 流持续返回 Agent 事件。"""
    async for event in execute_chat_stream(
        user_id,
        channel_tags=channel.channel_tags,
        content=content,
        sender_name=sender_name,
        session_id=session_id,
        session_metadata=session_metadata,
        timestamp=timestamp,
        files=files,
        json_output=json_output,
    ):
        yield event


@router.post(
    "/chat_sync",
    responses={400: {"description": "The configured model does not support the requested JSON output mode"}},
    dependencies=[Depends(acquire_chat_session, scope="request")],
)
async def chat_sync(
    channel: Annotated[AuthenticatedChannel, Depends(require_channel_jwt)],
    user_id: Annotated[str, Depends(user_id_header)],
    session_id: Annotated[str, Depends(session_id_header)],
    sender_name: Annotated[str, Depends(sender_name_header)],
    session_metadata: Annotated[dict[str, str], Depends(session_metadata)],
    content: Annotated[InputContent, Depends(content)],
    timestamp: Annotated[datetime, Form(default_factory=datetime.now, description="消息的时间戳")],
    files: Annotated[list[UploadFile], File(default_factory=list, description="消息附件列表")],
    json_output: Annotated[bool, Depends(json_output_enabled)],
) -> list[dict[str, Any]]:
    """接收用户消息，阻塞运行 Agent，并一次性返回合并后的全部事件。"""
    result = await execute_chat(
        user_id,
        channel_tags=channel.channel_tags,
        content=content,
        sender_name=sender_name,
        session_id=session_id,
        session_metadata=session_metadata,
        timestamp=timestamp,
        files=files,
        json_output=json_output,
    )
    return result
