import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any

import jwt

from ..config.settings import settings


def _channel_secret_key(channel: str) -> str:
    if not isinstance(channel, str):
        raise jwt.InvalidTokenError("Invalid channel JWT key ID")
    configured = settings.jwt.channels.get(channel)
    if configured is None:
        raise jwt.InvalidTokenError("Unknown channel JWT key ID")
    return configured.secret_key


class ChannelJWTAuth:
    """Channel credential decoded once at construction, with live expiration checks."""

    def __init__(self, token: str, secret_key: str | None = None):
        channel = jwt.get_unverified_header(token).get("kid")
        if not isinstance(channel, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}", channel) is None:
            raise jwt.InvalidTokenError("Invalid channel JWT key ID")
        if secret_key is None:
            secret_key = _channel_secret_key(channel)
        try:
            claims = jwt.decode(
                token, secret_key, algorithms=[settings.jwt.algorithm],
                options={"require": ["exp"], "verify_aud": False},
            )
        except (ValueError, TypeError, OverflowError):
            raise jwt.InvalidTokenError("Invalid channel JWT claims") from None
        self._token = token
        self._secret_key = secret_key
        self._channel = channel
        self._claims = MappingProxyType(claims)

    @property
    def token(self) -> str:
        return self._token

    @property
    def secret_key(self) -> str:
        return self._secret_key

    @property
    def channel(self) -> str:
        return self._channel

    @property
    def exp(self) -> int:
        return int(self.claims["exp"])

    @property
    def is_expired(self) -> bool:
        try:
            jwt.decode(
                self.token, self.secret_key, algorithms=[settings.jwt.algorithm],
                options={"require": ["exp"], "verify_aud": False},
            )
        except jwt.ExpiredSignatureError:
            return True
        return False

    @property
    def claims(self) -> Mapping[str, Any]:
        return self._claims

    def __repr__(self) -> str:
        return f"ChannelJWTAuth(channel={self.channel!r}, exp={self.exp!r})"

    @staticmethod
    def create_token(
        channel: str,
        *,
        secret_key: str | None = None,
        now: datetime | None = None,
        lark_user_access_token: str | None = None,
    ) -> str:
        """Sign a channel JWT using the shared algorithm and lifetime settings."""
        if not isinstance(channel, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}", channel) is None:
            raise ValueError("Invalid channel JWT key ID")
        if secret_key is None:
            secret_key = _channel_secret_key(channel)
        issued_at = now or datetime.now(UTC)
        claims: dict[str, Any] = {
            "exp": int((issued_at + timedelta(seconds=settings.jwt.access_token_expire_seconds)).timestamp()),
        }
        if lark_user_access_token is not None:
            claims["lark_user_access_token"] = lark_user_access_token
        return jwt.encode(claims, secret_key, algorithm=settings.jwt.algorithm, headers={"kid": channel})
