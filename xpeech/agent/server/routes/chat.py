"""User-bound chat and question-answer endpoints."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.sse import EventSourceResponse

from ...execute import execute_chat, execute_chat_stream
from ...loop import AgentLoop, QuestionEvent
from ....utils.channel_jwt import UserJWTAuth
from ..auth import require_user_jwt
from ..dependencies import (
    acquire_chat_session,
    authenticated_user_id,
    content,
    json_output_enabled,
    lark_user_access_token,
    sender_name_header,
    session_id_header,
    session_metadata,
)
from ..schema import (
    InputContent,
)

router = APIRouter()


@router.post("/answer_question")
async def answer_question(
    user_id: Annotated[str, Depends(authenticated_user_id)],
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
    dependencies=[
        Depends(acquire_chat_session, scope="request"),
        Depends(lark_user_access_token, scope="request"),
    ],
)
async def chat(
    auth: Annotated[UserJWTAuth, Depends(require_user_jwt)],
    user_id: Annotated[str, Depends(authenticated_user_id)],
    session_id: Annotated[str, Depends(session_id_header)],
    sender_name: Annotated[str, Depends(sender_name_header)],
    session_metadata: Annotated[dict[str, str], Depends(session_metadata)],
    content: Annotated[InputContent, Depends(content)],
    files: Annotated[list[UploadFile], File(default_factory=list, description="消息附件列表")],
    json_output: Annotated[bool, Depends(json_output_enabled)],
):
    """接收用户消息，并以 SSE 流持续返回 Agent 事件。"""
    async for event in execute_chat_stream(
        user_id,
        content=content,
        sender_name=sender_name,
        channel=auth.channel,
        session_id=session_id,
        session_metadata=session_metadata,
        files=files,
        json_output=json_output,
    ):
        yield event


@router.post(
    "/chat_sync",
    responses={400: {"description": "The configured model does not support the requested JSON output mode"}},
    dependencies=[Depends(lark_user_access_token, scope="request")],
)
async def chat_sync(
    auth: Annotated[UserJWTAuth, Depends(require_user_jwt)],
    user_id: Annotated[str, Depends(authenticated_user_id)],
    sender_name: Annotated[str, Depends(sender_name_header)],
    session_metadata: Annotated[dict[str, str], Depends(session_metadata)],
    content: Annotated[InputContent, Depends(content)],
    files: Annotated[list[UploadFile], File(default_factory=list, description="消息附件列表")],
    json_output: Annotated[bool, Depends(json_output_enabled)],
) -> list[dict[str, Any]]:
    """接收用户消息，阻塞运行 Agent，并一次性返回合并后的全部事件。

    每次调用在服务端生成独立会话，不占用会话并发锁，可与 `/chat` 或其他同步调用并发。
    """
    result = await execute_chat(
        user_id,
        content=content,
        sender_name=sender_name,
        channel=auth.channel,
        session_metadata=session_metadata,
        files=files,
        json_output=json_output,
    )
    return result
