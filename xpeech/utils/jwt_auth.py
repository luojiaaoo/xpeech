from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from ..config.settings import JWTConfig, settings


def create_access_token(
    *,
    config: JWTConfig | None = None,
    now: datetime | None = None,
) -> str:
    """Create a short-lived JWT for one request to the Xpeech API."""

    jwt_config = config or settings.jwt
    issued_at = now or datetime.now(UTC)
    expires_at = issued_at + timedelta(seconds=jwt_config.access_token_expire_seconds)
    return jwt.encode(
        {"exp": expires_at},
        jwt_config.secret_key,
        algorithm=jwt_config.algorithm,
    )


def decode_access_token(token: str, *, config: JWTConfig | None = None) -> dict[str, Any]:
    """Validate a JWT signature, registered claims, and expiration."""

    jwt_config = config or settings.jwt
    return jwt.decode(
        token,
        jwt_config.secret_key,
        algorithms=[jwt_config.algorithm],
        options={"require": ["exp"]},
    )
