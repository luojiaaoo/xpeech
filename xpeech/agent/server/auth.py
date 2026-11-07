from typing import Annotated, Any

import jwt
from fastapi import APIRouter, Depends, Form, HTTPException, Response, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials, OAuth2PasswordBearer
from pydantic import BaseModel

from ...config.settings import settings
from ...exceptions import PathProtectionError
from ...utils.channel_jwt import ChannelJWTAuth, ChannelTokenParams, UserJWTAuth, UserTokenParams, create_token
from ...utils.internal_jwt import decode_internal_access_token

router = APIRouter(tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    description=(
        "Swagger UI 的 client_id 填写 channel，client_secret 填写对应的 secret_key，"
        "Client credentials location 选择 Authorization header。username 填写用户 ID，password 留空。"
        "换取的访问令牌有效期为 "
        f"{settings.jwt.access_token_expire_seconds} 秒。"
    ),
)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


@router.post("/token", response_model=TokenResponse, include_in_schema=False)
async def issue_access_token(
    response: Response,
    credentials: Annotated[HTTPBasicCredentials, Depends(HTTPBasic())],
    username: Annotated[str, Form(description="用户 ID；留空时只签发 channel JWT")] = "",
) -> TokenResponse:
    """Validate a channel key and optionally bind the issued token to a user."""

    try:
        token = create_token(
            (
                UserTokenParams(channel=credentials.username, user_id=username)
                if username else ChannelTokenParams(channel=credentials.username)
            ),
            secret_key4verify=credentials.password,
        )
    except (jwt.InvalidTokenError, ValueError, PathProtectionError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect JWT secret",
            headers={"WWW-Authenticate": "Basic"},
        ) from None

    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return TokenResponse(
        access_token=token,
        token_type="bearer",
    )


def _authenticate_channel_token[Auth: ChannelJWTAuth](
    token: str | None, auth_type: type[Auth],
) -> Auth:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        auth = auth_type(token)
        lark_user_access_token = auth.claims.get("lark_user_access_token")
        if lark_user_access_token is not None and not isinstance(lark_user_access_token, str):
            raise jwt.InvalidTokenError("Invalid lark_user_access_token claim")
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth


async def require_channel_jwt(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> ChannelJWTAuth:
    """Authenticate business API callers using a channel JWT with kid."""

    return _authenticate_channel_token(token, ChannelJWTAuth)


async def require_user_jwt(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> UserJWTAuth:
    """Authenticate an external user exclusively through the signed subject."""
    return _authenticate_channel_token(token, UserJWTAuth)


async def require_internal_jwt(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> dict[str, Any]:
    """Authenticate calls made by trusted Xpeech services only."""

    try:
        claims = decode_internal_access_token(token)
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return claims
