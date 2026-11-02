from typing import Annotated, Any

import jwt
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials, OAuth2PasswordBearer
from pydantic import BaseModel

from ...config.settings import settings
from ...utils.channel_jwt import ChannelJWTAuth, ChannelTokenParams, create_token
from ...utils.internal_jwt import decode_internal_access_token
from .context import lark_user_access_token_var

router = APIRouter(tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    description=(
        "Swagger UI 的 client_id 填写 channel，client_secret 填写对应的 secret_key，"
        "Client credentials location 选择 Authorization header。username 和 password 留空。"
        "换取的访问令牌有效期为 "
        f"{settings.jwt.access_token_expire_seconds} 秒。"
    ),
)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


class AuthenticatedChannel(ChannelJWTAuth):
    """Verified channel identity and its server-configured channel tags."""

    @property
    def channel_tags(self) -> list[str]:
        channel = settings.jwt.channels.get(self.channel)
        return channel.channel_tags if channel is not None else []


@router.post("/token", response_model=TokenResponse, include_in_schema=False)
async def issue_access_token(
    response: Response,
    credentials: Annotated[HTTPBasicCredentials, Depends(HTTPBasic())],
) -> TokenResponse:
    """Validate a channel key and issue a channel token for Swagger UI."""

    try:
        token = create_token(
            ChannelTokenParams(channel=credentials.username),
            secret_key4verify=credentials.password,
        )
    except (jwt.InvalidTokenError, ValueError):
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


def _authenticate_channel_token(token: str | None) -> AuthenticatedChannel:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        auth = AuthenticatedChannel(token)
        lark_user_access_token = auth.claims.get("lark_user_access_token")
        if lark_user_access_token is not None and not isinstance(lark_user_access_token, str):
            raise jwt.InvalidTokenError("Invalid lark_user_access_token claim")
        return auth
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_channel_jwt(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> AuthenticatedChannel:
    """Authenticate business API callers using a channel JWT with kid."""

    auth = _authenticate_channel_token(token)
    lark_user_access_token_var.set(auth.claims.get("lark_user_access_token"))
    return auth


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
    lark_user_access_token_var.set(None)
    return claims
