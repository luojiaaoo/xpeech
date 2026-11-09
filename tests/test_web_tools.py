import asyncio
import json
from xml.etree import ElementTree

import httpx
import pytest
from pydantic import ValidationError

from xpeech.agent.tool_executor import ToolExecutor
from xpeech.agent.tools import web
from xpeech.agent.tools.helper import as_tool
from xpeech.provider.schema import ToolCallRequest

SEARCH_ARG_TYPES = {
    "baidu": web.BaiduSearchArgs,
    "bing": web.BingSearchArgs,
}


@pytest.fixture
def mock_http(monkeypatch):
    client_type = httpx.AsyncClient
    clients = []

    def install(handler):
        def make_client(**kwargs):
            client = client_type(transport=httpx.MockTransport(handler), **kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(web.httpx, "AsyncClient", make_client)
        return clients

    return install


@pytest.mark.asyncio
@pytest.mark.parametrize("include_images", [False, True])
async def test_web_fetch_includes_comments_and_optional_images(mock_http, include_images):
    calls = []

    def respond(request):
        calls.append(request)
        assert request.headers["User-Agent"].startswith("Mozilla/5.0")
        assert "zh-CN" in request.headers["Accept-Language"]
        assert "text/markdown" in request.headers["Accept"]
        assert request.extensions["timeout"]["connect"] == 10
        assert request.extensions["timeout"]["read"] == 60
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

        return make_response(content, "text/html; charset=utf-8")

    clients = mock_http(respond)

    args = web.WebFetchArgs(url="https://example.com/page", include_images=include_images)
    result = await web.web_fetch(args)

    assert result.startswith("[TITLE: Example]\n\n# Example")
    assert "A reader comment discussing the results" in result
    assert ("![Research figure](https://example.test/figure.png)" in result) == include_images
    assert "include_images" in as_tool(web.web_fetch)["function"]["parameters"]["properties"]
    assert [str(request.url) for request in calls] == ["https://example.com/page"]
    assert clients[0].is_closed


def make_response(data: bytes, content_type: str, url: str = "https://example.test/download"):
    return httpx.Response(
        200,
        content=data,
        headers={"Content-Type": content_type},
        request=httpx.Request("GET", url),
    )


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


@pytest.mark.parametrize("content_type", ["application/atom+xml", "application/atom+xml; charset=utf-8"])
def test_web_fetch_preserves_atom_feed(content_type):
    feed = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        '<title>订阅更新</title><entry><title>文章标题</title>'
        '<link href="https://example.test/article"/>'
        '<content type="html">&lt;p&gt;文章正文&lt;/p&gt;</content></entry></feed>'
    )
    result = web._response_to_markdown(make_response(feed.encode("utf-8"), content_type))
    assert result == feed


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
    ],
)
@pytest.mark.parametrize("engine", ["baidu", "bing"])
def test_web_search_rejects_invalid_arguments(kwargs, engine):
    with pytest.raises(ValidationError):
        SEARCH_ARG_TYPES[engine](**({"query": "test"} | kwargs))


def rss_response(items):
    root = ElementTree.Element("rss")
    channel = ElementTree.SubElement(root, "channel")
    for item in items:
        entry = ElementTree.SubElement(channel, "item")
        for tag, key in [("title", "title"), ("link", "url"), ("description", "abstract")]:
            ElementTree.SubElement(entry, tag).text = item[key]
    return make_response(ElementTree.tostring(root), "application/rss+xml")


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["baidu", "bing"])
async def test_web_search_routes_and_normalizes_results(engine, monkeypatch, mock_http):
    calls = []
    items = [
        {
            "title": f"标题 {index}",
            "url": f"https://example.test/{index}",
            "href": f"https://example.test/{index}",
            "abstract": "摘要",
            "body": "摘要",
        }
        for index in range(3)
    ]

    def baidu_search(keyword, max_results):
        calls.append(("baidu", keyword, max_results))
        return items

    monkeypatch.setattr(web, "_baidu_search", baidu_search)

    def respond(request):
        calls.append(("bing", str(request.url.copy_with(query=None)), dict(request.url.params)))
        assert request.headers["User-Agent"] == web._WEB_USER_AGENT
        assert request.extensions["timeout"]["connect"] == 10
        assert request.extensions["timeout"]["read"] == 10
        return rss_response(items)

    mock_http(respond)
    tool = getattr(web, f"{engine}_search")
    args = SEARCH_ARG_TYPES[engine](query=" 测试 ", max_results=2)
    hits = json.loads(await tool(args))
    expected = [
        {"title": f"标题 {index}", "url": f"https://example.test/{index}", "abstract": "摘要", "rank": str(index + 1)}
        for index in range(2)
    ]
    assert hits == expected
    if engine == "baidu":
        assert calls == [("baidu", "测试", 2)]
    else:
        assert calls == [("bing", "https://cn.bing.com/search", {"q": "测试", "format": "rss"})]


@pytest.mark.asyncio
async def test_split_search_tools_expose_independent_schemas_and_defaults(mock_http):
    calls = []

    def respond(request):
        calls.append((str(request.url.copy_with(query=None)), dict(request.url.params)))
        return rss_response([{"title": "Example", "url": "https://example.test", "abstract": "summary"}])

    mock_http(respond)
    result = json.loads(await web.bing_search(web.BingSearchArgs(query="test")))
    assert result[0]["url"] == "https://example.test"
    assert calls == [("https://cn.bing.com/search", {"q": "test", "format": "rss"})]
    for tool, properties in [
        (web.baidu_search, {"query", "max_results"}),
        (web.bing_search, {"query", "max_results"}),
    ]:
        schema = as_tool(tool)["function"]
        assert schema["name"] == tool.__name__
        assert set(schema["parameters"]["properties"]) == properties


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["baidu", "bing"])
async def test_web_search_empty_results(engine, monkeypatch, mock_http):
    monkeypatch.setattr(web, "_baidu_search", lambda *_args: None)
    mock_http(lambda _request: rss_response([]))
    tool = getattr(web, f"{engine}_search")
    args = SEARCH_ARG_TYPES[engine](query="test")
    assert "未找到" in await tool(args)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [
    b"<!doctype html><html><body>Bing homepage</body></html>",
    b"<html><body>Verify you are human</body></html>",
    b'<script type="text/javascript">var example = 1;</script><style>html{color:black}</style>',
    b"not XML",
    b"",
    b"<rss />",
])
async def test_bing_search_rejects_invalid_feed_instead_of_reporting_empty_results(content, mock_http):
    mock_http(lambda _request: make_response(content, "text/html"))
    with pytest.raises(RuntimeError, match="Bing搜索失败.*未返回有效 RSS"):
        await web.bing_search(web.BingSearchArgs(query="test"))


@pytest.mark.asyncio
@pytest.mark.parametrize("engine,label", [("baidu", "百度"), ("bing", "Bing")])
async def test_search_error_is_reported_as_failed_execution(tmp_path, monkeypatch, mock_http, engine, label):
    def fail(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(web, "_baidu_search", fail)
    mock_http(fail)
    name = f"{engine}_search"
    executor = ToolExecutor(tmp_path, max_result_chars=10_000, mapping_tool_call_funcs={name: getattr(web, name)})
    results = [
        result
        async for result in executor.execute(
            [ToolCallRequest(id="search-1", name=name, arguments={"query": "test"})],
        )
    ]
    assert not results[0].succeeded
    assert f"{label}搜索失败：offline" in results[0].value


@pytest.mark.asyncio
@pytest.mark.parametrize("engine,label", [("baidu", "百度"), ("bing", "Bing")])
async def test_web_search_timeout(monkeypatch, mock_http, engine, label):
    async def hang(*_args, **_kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(web.asyncio, "to_thread", hang)
    clients = mock_http(hang)
    monkeypatch.setattr(web, "_SEARCH_TIMEOUT_SECONDS", 0.01)
    tool = getattr(web, f"{engine}_search")
    args = SEARCH_ARG_TYPES[engine](query="test")
    with pytest.raises(RuntimeError, match=f"{label}搜索失败：搜索超时"):
        await tool(args)
    if engine == "bing":
        assert clients[0].is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["web_fetch", "bing_search"])
async def test_http_tools_follow_redirects(mock_http, tool_name):
    paths = []

    def respond(request):
        paths.append(request.url.path)
        if request.url.path != "/redirected/page":
            return httpx.Response(302, headers={"Location": "https://redirected.example.com/redirected/page"})
        if tool_name == "bing_search":
            return rss_response([{"title": "Example", "url": "https://example.test", "abstract": "summary"}])
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html"},
            content=(
                b"<html><head><title>Research update</title></head><body><article><h1>Research update</h1>"
                b"<p>This article describes an important research result and its practical applications. "
                b"The main text should remain readable after the page layout is removed.</p>"
                b'<p>The results are explained in <a href="details">the report</a>, '
                b"which includes methods, measurements, and reproducible examples.</p></article></body></html>"
            ),
        )

    clients = mock_http(respond)
    if tool_name == "web_fetch":
        result = await web.web_fetch(web.WebFetchArgs(url="https://example.com/page"))
        assert "[the report](https://redirected.example.com/details)" in result
        assert paths == ["/page", "/redirected/page"]
    else:
        result = await web.bing_search(web.BingSearchArgs(query="test"))
        assert json.loads(result)[0]["title"] == "Example"
        assert paths == ["/search", "/redirected/page"]
    assert clients[0].is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["web_fetch", "bing_search"])
async def test_http_tools_report_http_errors_and_close_client(mock_http, tool_name):
    clients = mock_http(lambda _request: httpx.Response(503))
    if tool_name == "web_fetch":
        with pytest.raises(httpx.HTTPStatusError, match="503"):
            await web.web_fetch(web.WebFetchArgs(url="https://example.com/page"))
    else:
        with pytest.raises(RuntimeError, match="Bing搜索失败.*503"):
            await web.bing_search(web.BingSearchArgs(query="test"))
    assert clients[0].is_closed
