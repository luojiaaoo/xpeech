from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Response

from .....utils.channel_jwt import UserJWTAuth
from ..session import require_embedded_jwt


def create_embed_router() -> APIRouter:
    router = APIRouter(prefix="/api/auth/embed")

    @router.get("")
    def verify_identity(
        auth: Annotated[UserJWTAuth, Depends(require_embedded_jwt)], response: Response,
    ):
        """Validate the supplied JWT without exchanging or persisting credentials."""
        response.headers["Cache-Control"] = "no-store"
        return {
            "user_id": auth.user_id,
            "expires_in": max(0, auth.claims["exp"] - datetime.now(UTC).timestamp()),
        }

    return router
