# xpeech/agent/server/middleware.py
from uuid import uuid4

import jwt
from fastapi import HTTPException
from fastapi.security.utils import get_authorization_scheme_param
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ...exceptions import PathProtectionError
from ...utils.ids import validate_user_id
from ...utils.internal_jwt import decode_internal_access_token
from .auth import AuthenticatedChannel, _authenticate_channel_token
from .context import (
    lark_user_access_token_var,
    request_id_var,
    session_id_var,
    user_id_var,
)


def _request_identity(headers: Headers) -> tuple[str | None, str | None, str | None]:
    """Read context from verified credentials; route dependencies enforce access."""
    scheme, token = get_authorization_scheme_param(headers.get("authorization"))
    if scheme.lower() != "bearer":
        return None, None, None
    try:
        if "kid" in jwt.get_unverified_header(token):
            auth = _authenticate_channel_token(token, AuthenticatedChannel)
            subject = auth.claims.get("sub")
            user_id = validate_user_id(subject) if subject is not None else None
            return user_id, auth.channel, auth.claims.get("lark_user_access_token")
        else:
            decode_internal_access_token(token)
            return headers.get("x-user-id"), None, None
    except (jwt.InvalidTokenError, HTTPException, PathProtectionError):
        return None, None, None


class ContextASGIMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):

        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        user_id, channel, lark_token = _request_identity(headers)
        session_id = headers.get("x-session-id")
        request_id = str(uuid4())

        user_id_token = user_id_var.set(user_id)
        session_id_token = session_id_var.set(session_id)
        request_id_token = request_id_var.set(request_id)
        lark_user_access_token = lark_user_access_token_var.set(lark_token)

        async def send_wrapper(message: Message):

            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                if user_id is not None:
                    response_headers["x-user-id"] = user_id
                if channel is not None:
                    response_headers["x-channel"] = channel
                if session_id is not None:
                    response_headers["x-session-id"] = session_id
                response_headers["x-request-id"] = request_id

            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            lark_user_access_token_var.reset(lark_user_access_token)
            request_id_var.reset(request_id_token)
            session_id_var.reset(session_id_token)
            user_id_var.reset(user_id_token)
