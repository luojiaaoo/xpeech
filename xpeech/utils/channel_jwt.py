import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any

import jwt

from ..config.settings import settings
from ..exceptions import PathProtectionError
from .ids import validate_user_id


def _channel_secret_key(channel: str) -> str:
    if not isinstance(channel, str):
        raise jwt.InvalidTokenError("Invalid channel JWT key ID")
    configured = settings.jwt.channels.get(channel)
    if configured is None:
        raise jwt.InvalidTokenError("Unknown channel JWT key ID")
    return configured.secret_key


@dataclass(frozen=True)
class ChannelTokenParams:
    """Parameters for issuing a channel JWT."""

    channel: str


@dataclass(frozen=True)
class UserTokenParams(ChannelTokenParams):
    """Parameters for issuing a channel JWT bound to one user."""

    user_id: str


class ChannelJWTAuth:
    """Channel credential decoded and validated once at construction."""

    def __init__(self, token: str):
        channel = jwt.get_unverified_header(token).get("kid")
        if not isinstance(channel, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}", channel) is None:
            raise jwt.InvalidTokenError("Invalid channel JWT key ID")
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
    def claims(self) -> Mapping[str, Any]:
        return self._claims

    def __repr__(self) -> str:
        return f"ChannelJWTAuth(channel={self.channel!r}, exp={self.exp!r})"


class UserJWTAuth(ChannelJWTAuth):
    """Channel credential bound only to its verified user subject."""

    def __init__(self, token: str):
        super().__init__(token)
        try:
            self._user_id = validate_user_id(self.claims.get("sub"))
        except PathProtectionError:
            raise jwt.InvalidTokenError("Invalid or missing user JWT subject") from None

    @property
    def user_id(self) -> str:
        return self._user_id


def create_token(params: ChannelTokenParams, *, secret_key4verify: str | None = None) -> str:
    """Sign a channel JWT, optionally verifying the supplied channel secret first."""
    channel = params.channel
    if not isinstance(channel, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}", channel) is None:
        raise ValueError("Invalid channel JWT key ID")
    secret_key = _channel_secret_key(channel)
    if secret_key4verify is not None and not secrets.compare_digest(
        secret_key4verify.encode("utf-8"), secret_key.encode("utf-8"),
    ):
        raise jwt.InvalidTokenError("Incorrect JWT secret")
    issued_at = datetime.now(UTC)
    claims: dict[str, Any] = {
        "exp": int((issued_at + timedelta(seconds=settings.jwt.access_token_expire_seconds)).timestamp()),
    }
    if isinstance(params, UserTokenParams):
        claims["sub"] = validate_user_id(params.user_id)
    return jwt.encode(claims, secret_key, algorithm=settings.jwt.algorithm, headers={"kid": channel})
