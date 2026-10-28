import asyncio
import json

import pytest
from pydantic import ValidationError

from xpeech.agent.tool_executor import ToolExecutor
from xpeech.agent.tools import web
from xpeech.agent.tools.helper import as_tool
from xpeech.provider.schema import ToolCallRequest


@pytest.mark.asyncio
@pytest.mark.parametrize("include_images", [False, True])
async def test_web_fetch_includes_comments_and_optional_images(monkeypatch, include_images):
    calls = []

    class FakeResponse:
        url = "https://example.com/page"
        content = (
            b"<html><head><title>Example</title></head><body><article><h1>Example</h1>"
            b"<p>This article explains a useful method and its practical applications. "
            b"The main text describes the research results in enough detail for readers to understand.</p>"
            b'<p><img src="https://example.test/figure.png" alt="Research figure"></p>'
            b"<p>The results are explained further in the final section. This paragraph discusses "
            b"reproducible experiments and their practical limitations in detail.</p></article>"
            b'<div id="comments"><p>A reader comment discussing the results and suggesting an alternative method.</p>'
            b"</div></body></html>"
        )

        def __init__(self):
            self.headers = {"Content-Type": "text/html; charset=utf-8"}

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
            assert self.headers["User-Agent"].startswith("Mozilla/5.0")
            assert "python-requests" not in self.headers["User-Agent"]
            assert "zh-CN" in self.headers["Accept-Language"]
            calls.append((url, kwargs))
            return FakeResponse()

    monkeypatch.setattr(web.requests, "Session", FakeSession)

    args = web.WebFetchArgs(url="https://example.com/page", include_images=include_images)
    result = await web.web_fetch(args)

    assert result.startswith("[TITLE: Example]\n\n# Example")
    assert "A reader comment discussing the results" in result
    assert ("![Research figure](https://example.test/figure.png)" in result) == include_images
    assert "include_images" in as_tool(web.web_fetch)["function"]["parameters"]["properties"]
    assert calls == [
        (
            "https://example.com/page",
            {"stream": True, "allow_redirects": True, "timeout": (10, 60)},
        ),
        "status_checked",
        "response_closed",
        "session_closed",
    ]


def make_response(data: bytes, content_type: str, url: str = "https://example.test/download"):
    response = web.requests.Response()
    response._content = data
    response.headers["Content-Type"] = content_type
    response.encoding = "utf-8"
    response.url = url
    return response


@pytest.mark.parametrize(
    "data,content_type,url,expected",
    [
        (
            b"name,value\nalpha,42\n",
            "text/csv; charset=utf-8",
            "https://example.test/download",
            "name,value\nalpha,42\n",
        ),
        (
            b"# Markdown\n\n[link](https://example.test)",
            "text/markdown",
            "https://example.test/content",
            "# Markdown\n\n[link](https://example.test)",
        ),
        (b"plain <text>", "text/plain", "https://example.test/content", "plain <text>"),
    ],
)
def test_web_fetch_preserves_text(data, content_type, url, expected):
    result = web._response_to_markdown(make_response(data, content_type, url))
    assert result == expected


@pytest.mark.parametrize("content_type", ["application/pdf", "application/octet-stream", "image/png"])
def test_web_fetch_rejects_binary_content(content_type):
    with pytest.raises(ValueError, match="Unsupported content type"):
        web._response_to_markdown(make_response(b"binary data", content_type))


def test_web_fetch_html_preserves_title_links_and_removes_images_and_scripts():
    response = make_response(
        b"<html><head><title>Page title</title></head><body><h1>Heading</h1>"
        b'<p>See <a href="https://example.test">link</a></p><img src="image.png" alt="hidden image">'
        b"<script>hidden script</script><style>hidden style</style></body></html>",
        "text/html",
    )
    result = web._response_to_markdown(response)
    assert result.startswith("[TITLE: Page title]\n\n# Heading")
    assert "[link](https://example.test)" in result
    assert "hidden" not in result


def test_web_fetch_extracts_article_without_navigation_and_footer():
    response = make_response(
        (
            b"<html><head><title>Research update</title></head><body>"
            b"<nav><a href='/home'>Navigation menu</a><a href='/login'>Account login</a></nav>"
            b"<article><h1>Research update</h1>"
            b"<p>This article describes an important research result and its practical applications. "
            b"The main text should remain readable after the page layout is removed.</p>"
            b"<p>The results are explained in <a href='/details'>the detailed report</a>, "
            b"which includes methods, measurements, and reproducible examples.</p>"
            b"<table><tr><th>Method</th><th>Score</th></tr><tr><td>Baseline</td><td>42</td></tr></table>"
            b"</article><footer>Copyright notice and privacy policy</footer></body></html>"
        ),
        "text/html",
        "https://example.test/articles/update",
    )
    result = web._response_to_markdown(response)
    assert result.startswith("[TITLE: Research update]")
    assert "important research result" in result
    assert "[the detailed report](https://example.test/details)" in result
    assert "Baseline" in result and "42" in result
    assert "Navigation menu" not in result
    assert "Copyright notice" not in result


def test_web_fetch_reports_empty_extraction(monkeypatch):
    monkeypatch.setattr(web.trafilatura, "extract", lambda *_args, **_kwargs: None)
    response = make_response(b"<html><body></body></html>", "text/html")
    with pytest.raises(ValueError, match="No readable content"):
        web._response_to_markdown(response)


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
