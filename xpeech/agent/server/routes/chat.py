"""User-bound chat and question-answer endpoints."""

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile
from fastapi.sse import EventSourceResponse

from ....utils.channel_jwt import UserJWTAuth
from ...execute import execute_chat, execute_chat_stream
from ...loop import AgentLoop, QuestionEvent
from ..auth import require_user_jwt
from ..chat_streams import CHAT_STREAMS
from ..dependencies import (
    authenticated_user_id,
    content,
    json_output_enabled,
    sender_name_header,
    session_id_header,
    session_metadata,
)
from ..schema import (
    InputContent,
)
from ..session_guard import CHAT_GUARD

router = APIRouter()


@router.get("/chat/status")
async def chat_status(
    response: Response,
    user_id: Annotated[str, Depends(authenticated_user_id)],
    session_id: Annotated[str, Depends(session_id_header)],
) -> dict[str, bool]:
    """查询会话是否仍有执行中的 loop，包括历史保存等清理阶段。"""
    response.headers["Cache-Control"] = "no-store"
    return {"active": await CHAT_GUARD.is_active(user_id, session_id)}


@router.post("/chat/stop")
async def stop_chat(
    user_id: Annotated[str, Depends(authenticated_user_id)],
    session_id: Annotated[str, Depends(session_id_header)],
) -> dict[str, bool]:
    """停止当前用户指定会话的 SSE 生成任务，等待模型并发名额归还。"""
    return {"stopped": await CHAT_STREAMS.stop(user_id, session_id)}


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


async def chat_events(
    auth: Annotated[UserJWTAuth, Depends(require_user_jwt)],
    user_id: Annotated[str, Depends(authenticated_user_id)],
    session_id: Annotated[str, Depends(session_id_header)],
    sender_name: Annotated[str, Depends(sender_name_header)],
    session_metadata: Annotated[dict[str, str], Depends(session_metadata)],
    content: Annotated[InputContent, Depends(content)],
    files: Annotated[list[UploadFile], File(default_factory=list, description="消息附件列表")],
    json_output: Annotated[bool, Depends(json_output_enabled)],
) -> AsyncIterator[AsyncGenerator[dict[str, Any], None]]:
    """在请求结束时关闭 SSE 数据源，包括断连时暂停在 yield 的情况。"""
    async with execute_chat_stream(
        user_id,
        content=content,
        sender_name=sender_name,
        channel=auth.channel,
        session_id=session_id,
        session_metadata=session_metadata,
        files=files,
        json_output=json_output,
    ) as events:
        yield events


@router.post(
    "/chat",
    response_class=EventSourceResponse,
    responses={
        400: {"description": "The configured model does not support the requested JSON output mode"},
        409: {"description": "Request identity or session context differs from the active session"},
    },
)
async def chat(
    events: Annotated[AsyncGenerator[dict[str, Any], None], Depends(chat_events)],
):
    """启动会话或插入活跃 loop；插入请求返回 command 确认，输出沿原 SSE 返回。"""
    async for event in events:
        yield event


@router.post(
    "/chat_sync",
    responses={400: {"description": "The configured model does not support the requested JSON output mode"}},
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
    return await execute_chat(
        user_id,
        content=content,
        sender_name=sender_name,
        channel=auth.channel,
        session_metadata=session_metadata,
        files=files,
        json_output=json_output,
    )
