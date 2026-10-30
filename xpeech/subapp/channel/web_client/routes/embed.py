import hashlib
import re
import secrets
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator

from ..dao import DuplicateUserIdError, WebClientDAO
from ..models import USER_ID_PATTERN, WebConfig, public_user
from ..session import EMBED_TOKEN_PREFIX, embedded_channel_auth, session_token


class EmbedSessionBody(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    channel_name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, value: str) -> str:
        value = value.strip()
        if re.fullmatch(USER_ID_PATTERN, value) is None:
            raise ValueError("user_id must follow the local account ID format")
        return value


def create_embed_router(
    config: WebConfig, dao: WebClientDAO, password_hash: Callable[[str], str],
) -> APIRouter:
    router = APIRouter(prefix="/api/auth/embed")

    def require_origin(request: Request) -> str:
        origin = request.headers.get("origin", "")
        if origin not in config.embed.allowed_origins:
            raise HTTPException(403, "不允许此站点嵌入")
        return origin

    async def revoke(request: Request) -> None:
        token = session_token(None, request.headers.get("authorization"))
        if token and token.startswith(EMBED_TOKEN_PREFIX):
            await dao.delete_session(hashlib.sha256(token.encode()).hexdigest())

    @router.post("/session")
    async def create_session(request: Request, body: EmbedSessionBody, response: Response):
        require_origin(request)
        embedded_channel_auth(body.channel_name)
        user = await dao.get_user_by_user_id(body.user_id)
        if user is None:
            try:
                user = await dao.create_user(
                    user_id=body.user_id,
                    username=body.user_id[:64],
                    password_hash=password_hash(secrets.token_urlsafe(32)),
                    is_admin=False,
                )
            except DuplicateUserIdError:
                user = await dao.get_user_by_user_id(body.user_id)
        if user is None or not user.is_active:
            raise HTTPException(403, "账号已停用")
        token = EMBED_TOKEN_PREFIX + secrets.token_urlsafe(32)
        await dao.create_session(
            token_hash=hashlib.sha256(token.encode()).hexdigest(),
            user_id=user.id,
            expires_at=datetime.now(UTC) + timedelta(seconds=config.embed.session_seconds),
        )
        await revoke(request)
        response.headers["Cache-Control"] = "no-store"
        return {"token": token, "expires_in": config.embed.session_seconds, "user": public_user(user)}

    @router.delete("/session", status_code=204)
    async def delete_session(request: Request):
        require_origin(request)
        await revoke(request)

    return router
