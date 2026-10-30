from fastapi import HTTPException

from ....config.settings import settings
from ....utils.channel_jwt import ChannelJWTAuth

EMBED_TOKEN_PREFIX = "embed_"


def embedded_channel_auth(channel_name: str | None) -> ChannelJWTAuth:
    """Resolve an embedded channel using server-side signing keys only."""
    if not channel_name or channel_name in {"internal", "feishu", "web_client"}:
        raise HTTPException(422, "嵌入会话必须指定自定义 channel_name，不能使用内置通道")
    channel = settings.jwt.channels.get(channel_name)
    if channel is None:
        raise HTTPException(422, "channel_name 未配置对应的 JWT 通道")
    return ChannelJWTAuth(channel_name, channel.secret_key)


def session_token(cookie: str | None, authorization: str | None = None) -> str | None:
    """An explicit Authorization header always takes precedence over a browser cookie."""
    if authorization is not None:
        scheme, _, token = authorization.partition(" ")
        return token.strip() if scheme.lower() == "bearer" and token.strip() else None
    return cookie
