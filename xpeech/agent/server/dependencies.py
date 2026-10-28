import json
from collections.abc import AsyncIterator
from typing import Annotated
from urllib.parse import unquote

from fastapi import Depends, Form, Header, HTTPException, status

from ...config.settings import settings
from ...exceptions import PathProtectionError
from ...utils.ids import DEFAULT_SESSION_ID, validate_session_id, validate_user_id
from .schema import InputContent
from .session_guard import SessionChatGuard

CHAT_GUARD = SessionChatGuard()


def json_output_enabled(
    json_output: Annotated[
        bool,
        Form(
            description='为大模型请求添加 response_format={"type":"json_object"}，不修改提示词；'
            "模型未配置支持 JSON 输出时返回 HTTP 400，成功响应仍使用 SSE 流"
        ),
    ] = False,
) -> bool:
    """在 SSE 响应开始前校验模型是否支持请求的 JSON 输出模式。"""
    if json_output and not settings.llm.support_json_output:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The configured model does not support JSON output",
        )
    return json_output


def user_id_header(
    user_id: Annotated[str, Header(description="用户 ID", alias="x-user-id")],
) -> str:
    """Validate the user ID from the request header."""
    try:
        return validate_user_id(user_id)
    except PathProtectionError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid user_id",
        ) from None


def session_id_header(
    session_id: Annotated[str, Header(description="会话 ID", alias="x-session-id")] = DEFAULT_SESSION_ID,
) -> str:
    """Validate the session ID from the request header."""
    try:
        return validate_session_id(session_id)
    except PathProtectionError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid session_id",
        ) from None


def sender_name_header(
    sender_name: Annotated[
        str,
        Header(
            description="发送者用户名；非 ASCII 字符可使用 UTF-8 URL 编码",
            alias="x-sender-name",
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
            detail="Invalid UTF-8 encoding in x-sender-name header",
        )
    if not decoded_sender_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="x-sender-name header must not be blank",
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
    user_id: Annotated[str, Depends(user_id_header)],
    session_id: Annotated[str, Depends(session_id_header)],
) -> AsyncIterator[None]:
    """占用会话直至响应结束，并拒绝同一会话的并发请求。"""
    if not await CHAT_GUARD.try_acquire(user_id, session_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Session '{session_id}' already has an active chat request",
        )
    try:
        yield
    finally:
        await CHAT_GUARD.release(user_id, session_id)
