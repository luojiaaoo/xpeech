import json
from collections.abc import AsyncIterator
from typing import Annotated
from urllib.parse import unquote

from fastapi import Depends, Form, Header, HTTPException, status

from ...config.settings import settings
from ...exceptions import PathProtectionError
from ...utils.ids import validate_session_id
from .auth import AuthenticatedUser, require_user_jwt
from .schema import InputContent
from .session_guard import SessionChatGuard

CHAT_GUARD = SessionChatGuard()
JSON_OUTPUT_TRIGGER = "::请严格以JSON格式回复::"


def internal_user_id(
    user_id: Annotated[str, Header(alias="x-user-id")],
) -> str:
    """Use the trusted internal caller's user parameter directly."""
    return user_id


def authenticated_user_id(
    auth: Annotated[AuthenticatedUser, Depends(require_user_jwt)],
) -> str:
    """Use only the verified JWT subject as the external user identity."""
    return auth.user_id


def session_id_header(
    session_id: Annotated[
        str,
        Header(
            description="必填会话 ID；所有通道均可传 default（主会话）或规范的小写连字符 UUIDv7",
            alias="x-session-id",
        ),
    ],
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
            description='会话的元数据，JSON 格式字符串，如：{"sender_id": "xxxx"}；'
            "渠道名由服务端根据 JWT kid 识别，不需要也不应出现在元数据中",
        ),
    ] = "{}",
):
    """将会话元数据的 JSON 字符串解析为字典，并移除其中的 channel 键。"""
    try:
        metadata = json.loads(session_metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid JSON format")
    if not isinstance(metadata, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="session_metadata must be a JSON object"
        )
    # channel 是服务端按 JWT kid 识别的独立字段，不接受调用方通过元数据传入。
    metadata.pop("channel", None)
    return metadata


def content(
    content: Annotated[
        str,
        Form(
            description='消息内容列表，JSON 格式字符串。[{"text": "你好"}, {"text": "world"}]。'
            f'在本次消息的 text 中加入完整标记 `{JSON_OUTPUT_TRIGGER}` 可强制模型输出 JSON 对象；'
            "前后的 `::` 均不可省略，缺少任一侧或仅出现普通 JSON 字样不会触发，仅本次请求生效。"
            "模型需配置 `llm.support_json_output = true`，否则在 SSE 开始前返回 HTTP 400。",
        ),
    ],
) -> InputContent:
    """将消息内容列表的 JSON 字符串解析为 InputContent 列表。"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid JSON format")


def json_output_enabled(
    request_content: Annotated[list[dict[str, str]], Depends(content)],
) -> bool:
    """仅根据本次用户文本的完整标记开启 JSON 输出，并校验模型是否支持。"""
    enabled = any(
        isinstance(block, dict) and isinstance(block.get("text"), str) and JSON_OUTPUT_TRIGGER in block["text"]
        for block in request_content
    )
    if enabled and not settings.llm.support_json_output:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The configured model does not support JSON output",
        )
    return enabled


async def acquire_chat_session(
    user_id: Annotated[str, Depends(authenticated_user_id)],
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
