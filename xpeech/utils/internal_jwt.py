from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from ..config.settings import settings


def create_internal_access_token() -> str:
    """Create a short-lived token for calls between trusted Xpeech services."""

    issued_at = datetime.now(UTC)
    claims: dict[str, Any] = {
        "exp": issued_at + timedelta(seconds=settings.jwt.access_token_expire_seconds),
    }
    return jwt.encode(
        claims,
        settings.jwt.secret_key,
        algorithm=settings.jwt.algorithm,
    )


def decode_internal_access_token(token: str) -> dict[str, Any]:
    """Validate a token issued for internal service-to-service calls."""

    return jwt.decode(
        token,
        settings.jwt.secret_key,
        algorithms=[settings.jwt.algorithm],
        options={"require": ["exp"]},
    )
