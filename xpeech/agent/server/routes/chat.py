import asyncio
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.sse import EventSourceResponse

from ....utils.helper import save_to_workspace
from ...background import (
    BACKGROUND_MESSAGE_QUEUES,
    BackgroundMessageChannel,
    FeishuBackgroundMessage,
)
from ...loop import AgentLoop, QuestionEvent
from ...runner import AgentRunner
from ..dependencies import acquire_chat_session, content, sender_name_header, session_metadata
from ..schema import InboundMessage, InputContent

router = APIRouter()
BACKGROUND_MESSAGE_LONG_POLL_SECONDS = 20


@router.get("/background_message", response_model=FeishuBackgroundMessage)
async def poll_background_message(
    channel: BackgroundMessageChannel,
) -> FeishuBackgroundMessage:
    """Long-poll for the next background message for a delivery channel."""
    queue = BACKGROUND_MESSAGE_QUEUES.get(channel)
    if queue is None:
        raise HTTPException(status_code=400, detail="Unsupported background message channel")
    try:
        return await asyncio.wait_for(
            queue.get(),
            timeout=BACKGROUND_MESSAGE_LONG_POLL_SECONDS,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=404, detail="No background message available") from exc


@router.post("/answer_question")
async def answer_question(
    session_id: Annotated[str, Header(description="会话的ID", alias="x-session-id")],
    answer: Annotated[str, Form(description="回答内容")],
):
    """提交用户答案并唤醒正在等待的 Agent 循环。"""
    question_event: QuestionEvent | None = AgentLoop.SESSION_QUESTION_EVENT.get(session_id)
    if question_event is not None:
        question_event.answer = answer
        question_event.event.set()
        return {"message": "Answer received"}
    return {"message": "Question event not found"}


@router.post(
    "/chat",
    response_class=EventSourceResponse,
    dependencies=[Depends(acquire_chat_session, scope="request")],
)
async def chat(
    session_id: Annotated[str, Header(description="会话的ID", alias="x-session-id")],
    sender_name: Annotated[str, Depends(sender_name_header)],
    session_metadata: Annotated[dict[str, str], Depends(session_metadata)],
    content: Annotated[InputContent, Depends(content)],
    timestamp: Annotated[datetime, Form(default_factory=datetime.now, description="消息的时间戳")],
    files: Annotated[list[UploadFile], File(default_factory=list, description="消息附件列表")],
):
    """接收用户消息，并以 SSE 流持续返回 Agent 事件。"""
    runner = await AgentRunner.create(session_id)

    saved_files = []
    for file in files:
        saved_files.append(await save_to_workspace(file=file, workspace=runner.workspace))

    message = InboundMessage(
        session_id=session_id,
        sender_name=sender_name,
        session_metadata=session_metadata,
        content=content,
        timestamp=timestamp,
        files=saved_files,
    )

    async for event in runner.run(message=message):
        yield event
