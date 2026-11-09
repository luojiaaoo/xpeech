"""Request context is initialized and restored by the ASGI middleware."""

from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from starlette.datastructures import Headers

from xpeech.agent.server import context
from xpeech.agent.server.middleware import ContextASGIMiddleware
from xpeech.utils.channel_jwt import ChannelTokenParams, UserTokenParams, create_token
from xpeech.utils.internal_jwt import create_internal_access_token


def current_context():
    return (
        context.get_user_id(), context.get_session_id(), context.get_request_id(),
        context.get_lark_user_access_token(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("credential", ["user", "channel", "internal", "invalid"])
@pytest.mark.parametrize("raise_error", [False, True])
@pytest.mark.parametrize("session_id", ["00000000-0000-7000-8000-000000000000", None, ""])
async def test_request_context_is_available_and_restored(credential, raise_error, session_id):
    expected_user = None
    if credential == "user":
        token = create_token(UserTokenParams("feishu", user_id="alice"))
        expected_user = "alice"
    elif credential == "channel":
        token = create_token(ChannelTokenParams("feishu"))
    elif credential == "internal":
        token = create_internal_access_token()
        expected_user = "trusted/internal-user"
    else:
        token = "invalid-token"

    before = current_context()
    observed = []
    send = AsyncMock()

    async def downstream(scope, receive, send):
        observed.append(current_context())
        assert context.get_user_id() == expected_user
        # 用户访问令牌改由表单参数依赖注入，中间件不再携带。
        assert context.get_lark_user_access_token() is None
        assert context.get_session_id() == session_id
        assert UUID(context.get_request_id()).version == 4
        if raise_error:
            raise RuntimeError("downstream failed")
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = ContextASGIMiddleware(downstream)
    scope = {
        "type": "http",
        "headers": [
            (b"authorization", f"Bearer {token}".encode()),
            (b"x-user-id", b"trusted/internal-user"),
        ],
    }
    if session_id is not None:
        scope["headers"].append((b"x-session-id", session_id.encode()))
    if raise_error:
        with pytest.raises(RuntimeError, match="downstream failed"):
            await middleware(scope, AsyncMock(), send)
    else:
        await middleware(scope, AsyncMock(), send)
        response_headers = Headers(raw=send.await_args_list[0].args[0]["headers"])
        if session_id is None:
            assert "x-session-id" not in response_headers
        else:
            assert response_headers["x-session-id"] == session_id
        assert response_headers["x-request-id"] == observed[0][2]
        if expected_user is None:
            assert "x-user-id" not in response_headers
        else:
            assert response_headers["x-user-id"] == expected_user
        if credential in {"user", "channel"}:
            assert response_headers["x-channel"] == "feishu"
        else:
            assert "x-channel" not in response_headers
    assert current_context() == before
