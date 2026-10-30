import secrets
from dataclasses import dataclass
from typing import Annotated, Any

import jwt
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from ...config.settings import settings
from ...utils.channel_jwt import (
    ChannelJWTAuth,
    create_channel_access_token,
    decode_channel_access_token,
)
from ...utils.internal_jwt import decode_internal_access_token
from .context import lark_user_access_token_var

router = APIRouter(tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    description=(
        "Swagger UI 使用 channel 名称和对应 JWT 密钥换取一个有效期 "
        f"{settings.jwt.access_token_expire_seconds} 秒的访问令牌。"
    ),
)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


@dataclass(frozen=True)
class AuthenticatedChannel:
    """Verified channel identity and its server-configured channel tags."""

    name: str
    channel_tags: list[str]


def _channel_keys() -> dict[str, str]:
    keys = {
        "feishu": settings.jwt.feishu_secret_key,
        "web_client": settings.jwt.web_client_secret_key,
    }
    keys.update(
        {
            channel_name: channel.secret_key
            for channel_name, channel in settings.jwt.channels.items()
        }
    )
    return keys


@router.post("/token", response_model=TokenResponse, include_in_schema=False)
async def issue_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    response: Response,
) -> TokenResponse:
    """Validate a channel key and issue a channel token for Swagger UI."""

    secret_key = _channel_keys().get(form_data.username)
    if secret_key is None or not secrets.compare_digest(
        form_data.password.encode("utf-8"),
        secret_key.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect JWT secret",
            headers={"WWW-Authenticate": "Bearer"},
        )

    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return TokenResponse(
        access_token=create_channel_access_token(
            ChannelJWTAuth(form_data.username, secret_key)
        ),
        token_type="bearer",
    )


def _authenticate_channel_token(
    token: str | None,
    *,
    web_client_only: bool = False,
) -> dict[str, Any]:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        kid = jwt.get_unverified_header(token).get("kid")
        channel_keys = _channel_keys()
        if not isinstance(kid, str) or kid not in channel_keys:
            raise jwt.InvalidTokenError("Invalid channel JWT key ID")
        claims = decode_channel_access_token(
            token,
            ChannelJWTAuth(kid, channel_keys[kid]),
        )
        lark_user_access_token = claims.get("lark_user_access_token")
        if lark_user_access_token is not None:
            if kid != "feishu":
                raise jwt.InvalidTokenError(
                    "lark_user_access_token is only valid in a Feishu channel token"
                )
            if not isinstance(lark_user_access_token, str):
                raise jwt.InvalidTokenError("Invalid lark_user_access_token claim")
        if web_client_only and kid != "web_client":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This endpoint requires the web_client channel",
            )
        return claims
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

    claims = _authenticate_channel_token(token)
    lark_user_access_token_var.set(claims.get("lark_user_access_token"))
    kid = jwt.get_unverified_header(token)["kid"]
    channel = settings.jwt.channels.get(kid)
    return AuthenticatedChannel(
        name=kid, channel_tags=channel.channel_tags if channel is not None else [],
    )


async def require_web_client_jwt(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> dict[str, Any]:
    """Authenticate statistics callers using the Web channel's JWT."""

    claims = _authenticate_channel_token(token, web_client_only=True)
    lark_user_access_token_var.set(None)
    return claims


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
