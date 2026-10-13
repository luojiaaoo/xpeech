import secrets
from typing import Annotated, Any

import jwt
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from ...config.settings import settings
from ...utils.jwt_auth import create_access_token, decode_access_token
from .context import user_access_token_var

router = APIRouter(tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    description="Swagger UI 会使用 JWT 密钥换取一个有效期 60 秒的访问令牌。",
)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


@router.post("/token", response_model=TokenResponse, include_in_schema=False)
async def issue_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    response: Response,
) -> TokenResponse:
    """Validate the shared JWT key and issue a one-minute token for Swagger UI."""

    secret_is_valid = secrets.compare_digest(
        form_data.password.encode("utf-8"),
        settings.jwt.secret_key.encode("utf-8"),
    )
    if not secret_is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect JWT secret",
            headers={"WWW-Authenticate": "Bearer"},
        )

    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return TokenResponse(
        access_token=create_access_token(),
        token_type="bearer",
    )


def _authenticate_token(token: str | None) -> dict[str, Any]:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        claims = decode_access_token(token)
        user_access_token = claims.get("user_access_token")
        if user_access_token is not None and (
            not isinstance(user_access_token, str) or "\x00" in user_access_token
        ):
            raise jwt.InvalidTokenError("Invalid user_access_token claim")
        return claims
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_jwt(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> dict[str, Any]:
    """Authenticate API callers with a Bearer token."""

    claims = _authenticate_token(token)
    user_access_token_var.set(claims.get("user_access_token"))
    return claims
