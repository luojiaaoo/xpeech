from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from ..config.settings import settings


@dataclass(frozen=True)
class ChannelJWTAuth:
    """The identity and signing configuration for one channel."""

    channel: str
    secret_key: str = field(repr=False)


def create_channel_access_token(
    auth: ChannelJWTAuth,
    *,
    now: datetime | None = None,
    lark_user_access_token: str | None = None,
) -> str:
    """Create a short-lived token signed with a channel's own key."""

    issued_at = now or datetime.now(UTC)
    claims: dict[str, Any] = {
        "exp": issued_at + timedelta(seconds=settings.jwt.access_token_expire_seconds),
    }
    if lark_user_access_token is not None:
        claims["lark_user_access_token"] = lark_user_access_token
    return jwt.encode(
        claims,
        auth.secret_key,
        algorithm=settings.jwt.algorithm,
        headers={"kid": auth.channel},
    )


def decode_channel_access_token(
    token: str,
    auth: ChannelJWTAuth,
) -> dict[str, Any]:
    """Validate a token with the specified channel's signing key."""

    if jwt.get_unverified_header(token).get("kid") != auth.channel:
        raise jwt.InvalidTokenError("Invalid channel JWT key ID")
    return jwt.decode(
        token,
        auth.secret_key,
        algorithms=[settings.jwt.algorithm],
        options={"require": ["exp"]},
    )
