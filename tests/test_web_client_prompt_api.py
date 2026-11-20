from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError, TimeoutError as RedisTimeoutError

from xpeech.config.settings import RedisConfig
from xpeech.subapp.channel.web_client.routes.inject_prompt import create_inject_prompt_router

def require_user():
    raise HTTPException(401, "请先登录")


@pytest.fixture
def prompt_api(monkeypatch):
    client = AsyncMock()
    client.__aenter__.return_value = client
    prompts = {}

    async def store(key, value, *, ex, nx):
        if key in prompts:
            return None
        prompts[key] = value
        return True

    async def consume(key):
        return prompts.pop(key, None)

    client.set.side_effect = store
    client.getdel.side_effect = consume
    monkeypatch.setattr("xpeech.subapp.channel.web_client.routes.inject_prompt.Redis", MagicMock(return_value=client))

    def create():
        app = FastAPI()
        app.include_router(create_inject_prompt_router(RedisConfig(), require_user))
        return TestClient(app)

    return create, client


def test_anonymous_create_then_authenticated_one_shot_read(prompt_api):
    create, redis = prompt_api
    with create() as api:
        response = api.post("/api/auth/inject-prompt", json={"user_prefix": "  注入内容\n"})
        assert response.status_code == 201
        state = response.json()["state"]
        assert re.fullmatch(r"[A-Za-z0-9_-]{43}", state)
        assert response.json() == {
            "state": state, "expires_in": 60, "login_url": f"/?state={state}",
        }
        assert response.headers["cache-control"] == "no-store"
        redis.set.assert_awaited_once_with(f"xpeech:inject_prompt:{state}", "注入内容", ex=60, nx=True)
        assert api.get("/api/auth/inject-prompt", params={"state": state}).status_code == 401
        redis.getdel.assert_not_awaited()
        api.app.dependency_overrides[require_user] = lambda: MagicMock()
        response = api.get("/api/auth/inject-prompt", params={"state": state})
        assert response.status_code == 200
        assert response.json() == {"user_prefix": "注入内容"}
        assert api.get("/api/auth/inject-prompt", params={"state": state}).status_code == 404


def test_each_write_generates_a_new_state(prompt_api):
    create, redis = prompt_api
    with create() as api:
        # expires_in 由服务端固定为 60 秒，即使调用方传入其他值也被忽略
        body = {"user_prefix": "content", "expires_in": 86400}
        first = api.post("/api/auth/inject-prompt", json=body)
        second = api.post("/api/auth/inject-prompt", json=body)
        assert first.status_code == second.status_code == 201
        assert first.json()["state"] != second.json()["state"]
        assert first.json()["expires_in"] == second.json()["expires_in"] == 60
        assert all(call.kwargs["ex"] == 60 for call in redis.set.await_args_list)
        assert redis.set.await_count == 2


def test_state_collision_is_regenerated_without_replacing_prompt(prompt_api, monkeypatch):
    create, redis = prompt_api
    state = "existing-state-token-1234567890"
    next_state = "new-state-token-1234567890"
    monkeypatch.setattr(
        "xpeech.subapp.channel.web_client.routes.inject_prompt.secrets.token_urlsafe",
        MagicMock(side_effect=[state, state, next_state]),
    )
    with create() as api:
        body = {"user_prefix": "original"}
        assert api.post("/api/auth/inject-prompt", json=body).status_code == 201
        body["user_prefix"] = "replacement"
        response = api.post("/api/auth/inject-prompt", json=body)
        assert response.status_code == 201
        assert response.json()["state"] == next_state
        api.app.dependency_overrides[require_user] = lambda: MagicMock()
        assert api.get("/api/auth/inject-prompt", params={"state": state}).json() == {"user_prefix": "original"}
        assert api.get("/api/auth/inject-prompt", params={"state": next_state}).json() == {"user_prefix": "replacement"}
        assert redis.set.await_args_list[0].kwargs["ex"] == 60


def test_repeated_state_collisions_return_503(prompt_api):
    create, redis = prompt_api
    redis.set.side_effect = None
    redis.set.return_value = None
    with create() as api:
        response = api.post("/api/auth/inject-prompt", json={"user_prefix": "content"})
        assert response.status_code == 503
        assert redis.set.await_count == 3


@pytest.mark.parametrize("values", [
    {"user_prefix": ""}, {"user_prefix": " \n\t"},
])
def test_invalid_prompt_request_is_rejected_before_redis(prompt_api, values):
    create, redis = prompt_api
    with create() as api:
        body = {"user_prefix": "content", **values}
        assert api.post("/api/auth/inject-prompt", json=body).status_code == 422
        redis.set.assert_not_awaited()


@pytest.mark.parametrize("error,code", [
    (ConnectionError("private-connection-data"), 502), (RedisTimeoutError("private-connection-data"), 504),
])
def test_prompt_write_failure_is_reported_without_secrets(prompt_api, error, code):
    create, redis = prompt_api
    redis.set.side_effect = error
    with create() as api:
        response = api.post("/api/auth/inject-prompt", json={"user_prefix": "private-content"})
        assert response.status_code == code
        assert "private" not in response.text
        redis.__aexit__.assert_awaited_once()
import re
