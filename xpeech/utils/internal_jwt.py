from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from ..config.settings import settings


def create_internal_access_token(
    *,
    secret_key: str | None = None,
    now: datetime | None = None,
) -> str:
    """Create a short-lived token for calls between trusted Xpeech services."""

    signing_key = secret_key if secret_key is not None else settings.jwt.secret_key
    issued_at = now or datetime.now(UTC)
    claims = {
        "exp": issued_at + timedelta(seconds=settings.jwt.access_token_expire_seconds),
    }
    return jwt.encode(
        claims,
        signing_key,
        algorithm=settings.jwt.algorithm,
    )


def decode_internal_access_token(
    token: str,
    *,
    secret_key: str | None = None,
) -> dict[str, Any]:
    """Validate a token issued for internal service-to-service calls."""

    signing_key = secret_key if secret_key is not None else settings.jwt.secret_key
    return jwt.decode(
        token,
        signing_key,
        algorithms=[settings.jwt.algorithm],
        options={"require": ["exp"]},
    )
