import asyncio
import json
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from types import SimpleNamespace
from urllib.parse import urlencode
from xml.etree import ElementTree

import pytest
from pydantic import ValidationError

from xpeech.agent.tool_executor import ToolExecutor
from xpeech.agent.tools import web
from xpeech.agent.tools.helper import as_tool
from xpeech.provider.schema import ToolCallRequest

SEARCH_ARG_TYPES = {
    "baidu": web.BaiduSearchArgs,
    "bing": web.BingSearchArgs,
    "news": web.NewsSearchArgs,
}


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
        {"query": " "},
        {"query": "x" * 201},
        {"max_results": 0},
        {"max_results": 11},
        {"max_results": 1.5},
        {"max_results": True},
        {"timelimit": "invalid"},
    ],
)
def test_web_search_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValidationError):
        web.NewsSearchArgs(**({"query": "test"} | kwargs))


def rss_response(items):
    root = ElementTree.Element("rss")
    channel = ElementTree.SubElement(root, "channel")
    for item in items:
        entry = ElementTree.SubElement(channel, "item")
        for tag, key in [("title", "title"), ("link", "url"), ("description", "abstract")]:
            ElementTree.SubElement(entry, tag).text = item[key]
        if "date" in item:
            ElementTree.SubElement(entry, "pubDate").text = item["date"]
        if "source" in item:
            ElementTree.SubElement(entry, "{urn:bing-news}Source").text = item["source"]
    return nullcontext(SimpleNamespace(content=ElementTree.tostring(root), raise_for_status=lambda: None))


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["baidu", "bing", "news"])
async def test_web_search_routes_and_normalizes_results(engine, monkeypatch):
    calls = []
    published = datetime.now(UTC).replace(microsecond=0)
    items = [
        {
            "title": f"标题 {index}",
            "url": f"https://example.test/{index}",
            "href": f"https://example.test/{index}",
            "abstract": "摘要",
            "body": "摘要",
            "date": format_datetime(published),
            "source": "示例媒体",
        }
        for index in range(3)
    ]

    def baidu_search(keyword, max_results):
        calls.append(("baidu", keyword, max_results))
        return items

    monkeypatch.setattr(web, "_baidu_search", baidu_search)

    def get(url, **kwargs):
        calls.append(("bing", url, kwargs))
        return rss_response(items)

    monkeypatch.setattr(web.requests, "get", get)
    tool = getattr(web, f"{engine}_search")
    options = {"timelimit": "w"} if engine == "news" else {}
    args = SEARCH_ARG_TYPES[engine](query=" 测试 ", max_results=2, **options)
    hits = json.loads(await tool(args))
    expected = [
        {"title": f"标题 {index}", "url": f"https://example.test/{index}", "abstract": "摘要", "rank": str(index + 1)}
        for index in range(2)
    ]
    if engine == "news":
        for hit in expected:
            hit.update(date=published.isoformat(), publisher="示例媒体")
    assert hits == expected
    if engine == "baidu":
        assert calls == [("baidu", "测试", 2)]
    elif engine == "bing":
        assert calls == [("bing", "https://cn.bing.com/search", {
            "params": {"q": "测试", "format": "rss"},
            "headers": {"User-Agent": web._WEB_USER_AGENT},
            "timeout": (10, 10),
        })]
    else:
        assert calls == [("bing", "https://www.bing.com/news/search", {
            "params": {"q": "测试", "format": "rss", "qft": 'sortbydate="1"'},
            "headers": {"User-Agent": web._WEB_USER_AGENT},
            "timeout": (10, 10),
        })]


@pytest.mark.asyncio
async def test_split_search_tools_expose_independent_schemas_and_defaults(monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs["params"]))
        return rss_response([{"title": "Example", "url": "https://example.test", "abstract": "summary"}])

    monkeypatch.setattr(web.requests, "get", get)
    result = json.loads(await web.bing_search(web.BingSearchArgs(query="test")))
    assert result[0]["url"] == "https://example.test"
    assert calls == [("https://cn.bing.com/search", {"q": "test", "format": "rss"})]
    for tool, properties in [
        (web.baidu_search, {"query", "max_results"}),
        (web.bing_search, {"query", "max_results"}),
        (web.news_search, {"query", "max_results", "timelimit"}),
    ]:
        schema = as_tool(tool)["function"]
        assert schema["name"] == tool.__name__
        assert set(schema["parameters"]["properties"]) == properties


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["baidu", "bing", "news"])
async def test_web_search_empty_results(engine, monkeypatch):
    monkeypatch.setattr(web, "_baidu_search", lambda *_args: None)
    monkeypatch.setattr(web.requests, "get", lambda *_args, **_kwargs: rss_response([]))
    tool = getattr(web, f"{engine}_search")
    args = SEARCH_ARG_TYPES[engine](query="test")
    assert "未找到" in await tool(args)


@pytest.mark.asyncio
@pytest.mark.parametrize("engine,label", [("bing", "Bing"), ("news", "新闻")])
@pytest.mark.parametrize("content", [b"<html><body>Verify you are human</body></html>", b"not XML"])
async def test_search_rejects_invalid_feed_instead_of_reporting_empty_results(content, monkeypatch, engine, label):
    monkeypatch.setattr(web.requests, "get", lambda *_args, **_kwargs: nullcontext(
        SimpleNamespace(content=content, raise_for_status=lambda: None)
    ))
    with pytest.raises(RuntimeError, match=f"{label}搜索失败"):
        await getattr(web, f"{engine}_search")(SEARCH_ARG_TYPES[engine](query="test"))


@pytest.mark.asyncio
@pytest.mark.parametrize("engine,label", [("baidu", "百度"), ("bing", "Bing"), ("news", "新闻")])
async def test_search_error_is_reported_as_failed_execution(tmp_path, monkeypatch, engine, label):
    def fail(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(web, "_baidu_search", fail)
    monkeypatch.setattr(web.requests, "get", fail)
    executor = ToolExecutor(tmp_path, max_result_chars=10_000)
    name = f"{engine}_search"
    results = await executor.execute(
        [ToolCallRequest(id="search-1", name=name, arguments={"query": "test"})],
        {name: getattr(web, name)},
    )
    assert not results[0].succeeded
    assert f"{label}搜索失败：offline" in results[0].value


@pytest.mark.asyncio
async def test_news_filters_dates_before_limiting_and_unwraps_bing_links(monkeypatch):
    now = datetime.now(UTC).replace(microsecond=0)
    article_url = "https://example.test/article?a=1&b=2"
    items = [
        {"title": "old", "date": format_datetime(now - timedelta(days=40))},
        {"title": "undated", "date": "invalid date"},
        {"title": "recent", "date": format_datetime(now - timedelta(hours=1))},
        {"title": "other", "date": format_datetime(now - timedelta(hours=2))},
    ]
    for item in items:
        item.update(
            url="https://www.bing.com/news/apiclick.aspx?" + urlencode({"url": article_url}),
            abstract="summary",
            source="Example publisher",
        )
    monkeypatch.setattr(web.requests, "get", lambda *_args, **_kwargs: rss_response(items))

    result = json.loads(await web.news_search(web.NewsSearchArgs(query="test", timelimit="d", max_results=1)))

    assert result == [{
        "title": "recent",
        "url": article_url,
        "abstract": "summary",
        "date": (now - timedelta(hours=1)).isoformat(),
        "publisher": "Example publisher",
        "rank": "1",
    }]
    unfiltered = json.loads(await web.news_search(web.NewsSearchArgs(query="test")))
    assert len(unfiltered) == 4
    assert unfiltered[1]["date"] == "invalid date"


@pytest.mark.asyncio
@pytest.mark.parametrize("engine,label", [("baidu", "百度"), ("bing", "Bing"), ("news", "新闻")])
async def test_web_search_timeout(monkeypatch, engine, label):
    async def hang(*_args):
        await asyncio.Event().wait()

    monkeypatch.setattr(web.asyncio, "to_thread", hang)
    monkeypatch.setattr(web, "_SEARCH_TIMEOUT_SECONDS", 0.01)
    tool = getattr(web, f"{engine}_search")
    args = SEARCH_ARG_TYPES[engine](query="test")
    with pytest.raises(RuntimeError, match=f"{label}搜索失败：搜索超时"):
        await tool(args)
