import asyncio
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from fastapi.sse import EventSourceResponse

from ....utils.file_io import save_to_workspace
from ....utils.ids import DEFAULT_SESSION_ID
from ...background import (
    BACKGROUND_MESSAGE_QUEUES,
    BackgroundMessageChannel,
    FeishuBackgroundMessage,
)
from ...loop import AgentLoop, QuestionEvent
from ...runner import AgentRunner
from ..dependencies import (
    acquire_chat_session,
    content,
    json_output_enabled,
    sender_name_header,
    session_id_header,
    session_metadata,
    user_id_header,
)
from ..schema import InboundMessage, InputContent

router = APIRouter()
BACKGROUND_MESSAGE_LONG_POLL_SECONDS = 50


@router.get(
    "/background_message",
    response_model=FeishuBackgroundMessage,
    responses={204: {"description": "No background message available"}},
)
async def poll_background_message(
    channel: BackgroundMessageChannel,
) -> FeishuBackgroundMessage | Response:
    """Long-poll for the next background message for a delivery channel."""
    queue = BACKGROUND_MESSAGE_QUEUES.get(channel)
    if queue is None:
        raise HTTPException(status_code=400, detail="Unsupported background message channel")
    try:
        return await asyncio.wait_for(
            queue.get(),
            timeout=BACKGROUND_MESSAGE_LONG_POLL_SECONDS,
        )
    except TimeoutError:
        return Response(status_code=204)


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
    runner = await AgentRunner.create(
        user_id,
        include_background_tools=session_id == DEFAULT_SESSION_ID,
    )

    saved_files = []
    for file in files:
        saved_files.append(await save_to_workspace(file=file, workspace=runner.workspace))

    message = InboundMessage(
        user_id=user_id,
        session_id=session_id,
        sender_name=sender_name,
        session_metadata=session_metadata,
        content=content,
        timestamp=timestamp,
        files=saved_files,
        json_output=json_output,
    )

    async for event in runner.run(message=message):
        yield event
