import json
from collections.abc import AsyncIterator
from typing import Annotated
from urllib.parse import unquote

from fastapi import Form, Header, HTTPException, status

from .schema import InputContent
from .session_guard import SessionChatGuard

CHAT_GUARD = SessionChatGuard()


def sender_name_header(
    sender_name: Annotated[
        str,
        Header(
            description="发送者用户名；非 ASCII 字符可使用 UTF-8 URL 编码",
            alias="sender-name",
            min_length=1,
        ),
    ],
) -> str:
    """读取并规范化必填的发送者用户名请求头。"""
    try:
        decoded_sender_name = unquote(sender_name, errors="strict").strip()
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid UTF-8 encoding in sender-name header",
        )
    if not decoded_sender_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="sender-name header must not be blank",
        )
    return decoded_sender_name


def session_metadata(
    session_metadata: Annotated[
        str,
        Form(
            description='会话的元数据，JSON 格式字符串，如：{"sender_id": "xxxx","channel": "feishu"}',
        ),
    ] = "{}",
):
    """将会话元数据的 JSON 字符串解析为字典。"""
    try:
        return json.loads(session_metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid JSON format")


def content(
    content: Annotated[
        str,
        Form(
            description='消息内容列表，JSON 格式字符串。[{"text": "你好"}, {"text": "world"}]',
        ),
    ],
) -> InputContent:
    """将消息内容列表的 JSON 字符串解析为 InputContent 列表。"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid JSON format")


async def acquire_chat_session(
    session_id: Annotated[str, Header(description="会话的ID", alias="x-session-id")],
) -> AsyncIterator[None]:
    """占用会话直至响应结束，并拒绝同一会话的并发请求。"""
    if not await CHAT_GUARD.try_acquire(session_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Session '{session_id}' already has an active chat request",
        )
    try:
        yield
    finally:
        await CHAT_GUARD.release(session_id)
