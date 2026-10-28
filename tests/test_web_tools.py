import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from xpeech.agent.tool_executor import ToolExecutor
from xpeech.agent.tools import web
from xpeech.agent.tools.helper import as_tool
from xpeech.provider.schema import ToolCallRequest


def test_web_fetch_sets_network_timeouts_and_closes_response(monkeypatch):
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            calls.append("response_closed")

        def raise_for_status(self):
            calls.append("status_checked")

    class FakeSession:
        def __init__(self):
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            calls.append("session_closed")

        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse()

    class FakeMarkItDown:
        def convert(self, response):
            assert isinstance(response, FakeResponse)
            return SimpleNamespace(title="Example", text_content="body")

    monkeypatch.setattr(web.requests, "Session", FakeSession)
    monkeypatch.setattr(web, "MarkItDown", FakeMarkItDown)

    result = web._fetch_and_convert("https://example.com/page")

    assert result == "[TITLE: Example]\n\nbody"
    assert calls == [
        (
            "https://example.com/page",
            {"stream": True, "timeout": (10, 60)},
        ),
        "status_checked",
        "response_closed",
        "session_closed",
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"keyword": " "},
        {"keyword": "x" * 201},
        {"max_results": 0},
        {"max_results": 11},
        {"max_results": 1.5},
        {"max_results": True},
        {"engine": "bing"},
        {"timelimit": "invalid"},
        {"engine": "baidu", "timelimit": "w"},
    ],
)
def test_web_search_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValidationError):
        web.WebSearchArgs(**({"keyword": "test"} | kwargs))


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["baidu", "duckduckgo", "news"])
async def test_web_search_routes_and_normalizes_results(engine, monkeypatch):
    calls = []
    items = [
        {
            "title": f"标题 {index}",
            "url": f"https://example.test/{index}",
            "href": f"https://example.test/{index}",
            "abstract": "摘要",
            "body": "摘要",
            "date": "2026-10-28",
            "source": "示例媒体",
        }
        for index in range(3)
    ]

    def baidu_search(keyword, max_results):
        calls.append(("baidu", keyword, max_results))
        return items

    class DDGS:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            calls.append("closed")

        def text(self, query, **kwargs):
            calls.append(("text", query, kwargs))
            return iter(items)

        def news(self, query, **kwargs):
            calls.append(("news", query, kwargs))
            return iter(items)

    monkeypatch.setattr(web, "baidu_search", baidu_search)
    monkeypatch.setattr(web, "DDGS", DDGS)
    args = web.WebSearchArgs(
        keyword=" 测试 ", engine=engine, max_results=2, timelimit=None if engine == "baidu" else "w"
    )
    hits = json.loads(await web.web_search(args))
    expected = [
        {"title": f"标题 {index}", "url": f"https://example.test/{index}", "abstract": "摘要", "rank": str(index + 1)}
        for index in range(2)
    ]
    if engine == "news":
        for hit in expected:
            hit.update(date="2026-10-28", publisher="示例媒体")
    assert hits == expected
    if engine == "baidu":
        assert calls == [("baidu", "测试", 2)]
    else:
        assert calls == [
            ("init", {"timeout": 10}),
            (
                "news" if engine == "news" else "text",
                "测试",
                {
                    "max_results": 2,
                    "backend": "auto" if engine == "news" else "duckduckgo",
                    "region": "cn-zh",
                    "timelimit": "w",
                },
            ),
            "closed",
        ]


@pytest.mark.asyncio
async def test_existing_keyword_call_and_schema_remain_supported(monkeypatch):
    calls = []

    def ddgs_search(*args):
        calls.append(args)
        return [{"title": "Example", "href": "https://example.test", "body": "summary"}]

    monkeypatch.setattr(web, "_ddgs_search", ddgs_search)
    result = json.loads(await web.web_search(web.WebSearchArgs(keyword="test")))
    assert result[0]["url"] == "https://example.test"
    assert calls == [("text", "test", 5, None)]
    schema = as_tool(web.web_search)["function"]
    assert schema["name"] == "web_search"
    assert set(schema["parameters"]["properties"]) == {"keyword", "engine", "max_results", "timelimit"}


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["baidu", "duckduckgo", "news"])
async def test_web_search_empty_results(engine, monkeypatch):
    monkeypatch.setattr(web, "baidu_search", lambda *_args: None)
    monkeypatch.setattr(web, "_ddgs_search", lambda *_args: [])
    assert "未找到" in await web.web_search(web.WebSearchArgs(keyword="test", engine=engine))


@pytest.mark.asyncio
async def test_search_error_is_reported_as_failed_execution(tmp_path, monkeypatch):
    def fail(_args):
        raise OSError("offline")

    monkeypatch.setattr(web, "_collect_search_results", fail)
    executor = ToolExecutor(tmp_path, max_result_chars=10_000)
    results = await executor.execute(
        [ToolCallRequest(id="search-1", name="web_search", arguments={"keyword": "test"})],
        {"web_search": web.web_search},
    )
    assert not results[0].succeeded
    assert "DuckDuckGo搜索失败：offline" in results[0].value


@pytest.mark.asyncio
async def test_web_search_timeout(monkeypatch):
    async def hang(*_args):
        await asyncio.Event().wait()

    monkeypatch.setattr(web.asyncio, "to_thread", hang)
    monkeypatch.setattr(web, "_SEARCH_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(RuntimeError, match="DuckDuckGo搜索失败：搜索超时"):
        await web.web_search(web.WebSearchArgs(keyword="test"))
