import httpx
import pytest

from xpeech.config.settings import settings
from xpeech.utils import helper


@pytest.fixture
def fake_vllm_tokenizer():
    """Use the real tokenizer HTTP helper with a mocked HTTP client in these tests."""


@pytest.mark.asyncio
async def test_vllm_tokenize_uses_configured_service(monkeypatch):
    requests = []

    class FakeAsyncClient:
        def __init__(self, *, base_url, timeout):
            requests.append({"base_url": base_url, "timeout": timeout})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, path, *, json):
            requests.append({"path": path, "json": json})
            return httpx.Response(
                200,
                json={"count": 7, "tokens": [1, 2, 3, 4, 5, 6, 7]},
                request=httpx.Request("POST", "http://tokenizer.test/tokenize"),
            )

    monkeypatch.setattr(helper.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(settings.llm.tokenizer, "api_base", "http://tokenizer.test:8000")
    monkeypatch.setattr(settings.llm.tokenizer, "model", "test")
    monkeypatch.setattr(settings.llm.tokenizer, "timeout", 42.0)

    messages = [{"role": "user", "content": "Hello"}]
    count = await helper.token_counter(messages)

    assert count == 7
    assert requests == [
        {"base_url": "http://tokenizer.test:8000", "timeout": 42.0},
        {"path": "/tokenize", "json": {
            "model": "test",
            "messages": messages,
            "add_generation_prompt": True,
            "return_token_strs": False,
        }},
    ]


@pytest.mark.asyncio
async def test_vllm_tokenize_does_not_fall_back_on_http_error(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, _path, *, json):
            return httpx.Response(
                503,
                json={"error": {"message": "tokenizer unavailable"}},
                request=httpx.Request("POST", "http://tokenizer.test/tokenize"),
            )

    monkeypatch.setattr(helper.httpx, "AsyncClient", FakeAsyncClient)

    with pytest.raises(httpx.HTTPStatusError):
        await helper._vllm_tokenize({"model": "test", "messages": []})


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [-1, True, 1.5, "7", None])
async def test_token_counter_rejects_invalid_vllm_count(monkeypatch, count):
    async def tokenize(_payload):
        return {"count": count}

    monkeypatch.setattr(helper, "_vllm_tokenize", tokenize)

    with pytest.raises(RuntimeError, match="invalid token count"):
        await helper.token_counter([])
