# xpeech/agent/server/middleware.py
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .context import request_id_var, session_id_var


class ContextASGIMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):

        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        session_id = headers.get("x-session-id") or str(uuid4())
        request_id = str(uuid4())

        session_id_token = session_id_var.set(session_id)
        request_id_token = request_id_var.set(request_id)

        async def send_wrapper(message: Message):

            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                response_headers["x-session-id"] = session_id
                response_headers["x-request-id"] = request_id

            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            request_id_var.reset(request_id_token)
            session_id_var.reset(session_id_token)
