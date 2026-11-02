import math
from datetime import UTC, datetime
from typing import Annotated

import jwt
from fastapi import Header, HTTPException

from ....config.settings import settings
from ....utils.channel_jwt import UserJWTAuth


def embedded_channel(channel_name: str | None) -> str:
    """Resolve an embedded channel using server-side signing keys only."""
    if not channel_name or channel_name in {"internal", "feishu", "web_client"}:
        raise HTTPException(422, "嵌入 JWT 必须使用自定义 kid，不能使用内置通道")
    channel = settings.jwt.channels.get(channel_name)
    if channel is None:
        raise HTTPException(422, "JWT kid 未配置对应的通道")
    return channel_name


def session_token(cookie: str | None, authorization: str | None = None) -> str | None:
    """An explicit Authorization header always takes precedence over a browser cookie."""
    if authorization is not None:
        scheme, _, token = authorization.partition(" ")
        return token.strip() if scheme.lower() == "bearer" and token.strip() else None
    return cookie


def require_embedded_jwt(
    authorization: Annotated[str | None, Header()] = None,
) -> UserJWTAuth:
    """Verify each business JWT without reading or creating a login session."""
    try:
        token = session_token(None, authorization)
        if not token or len(token) > 16384:
            raise jwt.InvalidTokenError("Missing embedded JWT")
        auth = UserJWTAuth(token)
        if not auth.user_id.strip():
            raise jwt.InvalidTokenError("Empty user identity")
        embedded_channel(auth.channel)
        expires_at = auth.claims["exp"]
        if (
            isinstance(expires_at, bool)
            or not isinstance(expires_at, (int, float))
            or not math.isfinite(expires_at)
        ):
            raise jwt.InvalidTokenError("Invalid expiration")
        datetime.fromtimestamp(expires_at, UTC)
        return auth
    except (jwt.InvalidTokenError, HTTPException, ValueError, TypeError, OverflowError, OSError):
        raise HTTPException(401, "嵌入 JWT 无效或已过期") from None
