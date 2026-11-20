import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from loguru import logger
from redis.exceptions import ConnectionError, TimeoutError as RedisTimeoutError

from xpeech.config.settings import RedisConfig, Settings
from xpeech.subapp.channel.web_client.models import INJECT_PROMPT_KEY_PREFIX
from xpeech.subapp.channel.web_client.routes.inject_prompt import resolve_injected_prompt


@pytest.fixture
def redis_client(monkeypatch):
    client = AsyncMock()
    client.__aenter__.return_value = client
    constructor = MagicMock(return_value=client)
    monkeypatch.setattr("xpeech.subapp.channel.web_client.routes.inject_prompt.Redis", constructor)
    return client, constructor


def test_redis_connection_validation():
    for values in ({"port": 0}, {"port": 65536}, {"db": -1}, {"timeout": 0}, {"timeout": float("inf")}):
        with pytest.raises(ValueError):
            RedisConfig(**values)
    assert "private-password" not in repr(RedisConfig(password="private-password"))


def test_redis_ignores_dotenv_and_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "REDIS_HOST=cache\nREDIS_PORT=7891\nREDIS_DB=2\nREDIS_USERNAME=prompt-user\n"
        "REDIS_PASSWORD=private-password\nREDIS_SSL=true\nREDIS_TIMEOUT=3.5\n"
        "POSTGRES_PASSWORD=unrelated-setting\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("REDIS_HOST", "environment-cache")
    monkeypatch.setenv("REDIS_PORT", "7892")
    monkeypatch.setenv("REDIS_USERNAME", "environment-user")
    monkeypatch.setenv("REDIS_PASSWORD", "environment-password")
    config = RedisConfig()
    assert (config.host, config.port, config.db, config.username) == ("redis", 6379, 0, "xpeech")
    assert config.password.get_secret_value() == ""
    assert config.ssl is False
    assert config.timeout == 10.0
    assert RedisConfig(port=7893).port == 7893


def test_redis_default_port(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("REDIS_PORT", raising=False)
    assert RedisConfig().port == 6379


def test_redis_connection_and_credentials_are_loaded_only_from_toml(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1] / "conf.toml.example"
    toml = source.read_text(encoding="utf-8").replace('username = "xpeech"', 'username = "toml-user"')
    toml = toml.replace('password = "replace-with-a-database-password"', 'password = "toml-password"')
    (tmp_path / "conf.toml").write_text(toml, encoding="utf-8")
    (tmp_path / ".env").write_text("REDIS_USERNAME=dotenv-user\nREDIS_PASSWORD=dotenv-password\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REDIS_USERNAME", "prompt-user")
    monkeypatch.setenv("REDIS_PASSWORD", "private-password")
    config = Settings().redis
    assert config.host == "redis"
    assert config.port == 6379
    assert config.username == "toml-user"
    assert config.password.get_secret_value() == "toml-password"


@pytest.mark.asyncio
async def test_redis_prompt_consumed_once_with_custom_connection(redis_client):
    client, constructor = redis_client
    client.getdel.side_effect = ["  提示词\n", None]
    config = RedisConfig(host="cache", port=6380, db=2, username="user", password="secret", ssl=True, timeout=3)
    assert await resolve_injected_prompt(config, "state-token-1234567890") == "提示词"
    with pytest.raises(HTTPException) as raised:
        await resolve_injected_prompt(config, "state-token-1234567890")
    assert raised.value.status_code == 404
    client.getdel.assert_awaited_with(f"{INJECT_PROMPT_KEY_PREFIX}state-token-1234567890")
    assert client.__aexit__.await_count == 2
    options = constructor.call_args.kwargs
    assert {key: options[key] for key in ("host", "port", "db", "username", "password", "ssl")} == {
        "host": "cache", "port": 6380, "db": 2, "username": "user", "password": "secret", "ssl": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, "", " \n\t"])
async def test_missing_or_empty_redis_prompt_returns_404(redis_client, value):
    client, _ = redis_client
    client.getdel.return_value = value
    with pytest.raises(HTTPException) as raised:
        await resolve_injected_prompt(RedisConfig(), "state-token-1234567890")
    assert raised.value.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("error,code", [
    (ConnectionError("private-connection-data"), 502),
    (RedisTimeoutError("private-connection-data"), 504),
    (UnicodeDecodeError("utf8", b"\xff", 0, 1, "invalid"), 502),
])
async def test_redis_errors_are_safe_and_close_connection(redis_client, error, code):
    client, _ = redis_client
    client.getdel.side_effect = error
    messages = []
    sink_id = logger.add(messages.append, format="{message}", level="WARNING")
    try:
        with pytest.raises(HTTPException) as raised:
            await resolve_injected_prompt(RedisConfig(), "private-state")
    finally:
        logger.remove(sink_id)
    assert raised.value.status_code == code
    assert "private" not in "".join(messages)
    client.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_redis_overall_timeout_closes_connection(redis_client):
    client, _ = redis_client

    async def slow_getdel(key):
        await asyncio.Event().wait()

    client.getdel.side_effect = slow_getdel
    config = RedisConfig(timeout=0.01)
    with pytest.raises(HTTPException) as raised:
        await resolve_injected_prompt(config, "state-token-1234567890")
    assert raised.value.status_code == 504
    client.__aexit__.assert_awaited_once()


