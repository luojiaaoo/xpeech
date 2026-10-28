import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from xpeech.agent.tool_executor import ToolExecutor
from xpeech.agent.tools import research
from xpeech.agent.tools.research import ArxivDownloadArgs, ArxivSearchArgs, build_research_tools
from xpeech.exceptions import PathProtectionError
from xpeech.provider.litellm_provider import LiteLLMProvider
from xpeech.provider.schema import LLMParameters, ToolCallRequest


@pytest.fixture
def article():
    return SimpleNamespace(
        title="示例论文",
        get_short_id=lambda: "2103.03404v2",
        entry_id="https://arxiv.org/abs/2103.03404v2",
        authors=[SimpleNamespace(name="Author")],
        primary_category="cs.AI",
        categories=["cs.AI"],
        published=datetime(2021, 3, 5, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/2103.03404v2",
        links=[SimpleNamespace(href="https://arxiv.org/pdf/2103.03404v2")],
        summary="摘要",
        comment=None,
    )


@pytest.fixture
def pdf_client(monkeypatch):
    client_cls = httpx.AsyncClient
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, content=b"%PDF-1.4 example")

    monkeypatch.setattr(
        research.httpx,
        "AsyncClient",
        lambda **kwargs: client_cls(transport=httpx.MockTransport(respond), **kwargs),
    )
    return requests


@pytest.mark.parametrize(
    "query,max_results", [(" ", 10), ("x" * 201, 10), ("AI", 0), ("AI", 21), ("AI", 1.5), ("AI", True)]
)
def test_search_rejects_invalid_arguments(query, max_results):
    with pytest.raises(ValidationError):
        ArxivSearchArgs(query=query, max_results=max_results)


@pytest.mark.parametrize("ids", [[], ["2103.03404"] * 6, ["../secret"], ["2103.03404/../../secret"], [123]])
def test_download_rejects_invalid_ids(ids):
    with pytest.raises(ValidationError):
        ArxivDownloadArgs(id_list=ids, download_dir="papers")


@pytest.mark.asyncio
async def test_search_runs_through_provider_and_executor(tmp_path, monkeypatch, article):
    searches = []

    def client_results(self, *, search):
        searches.append(search)
        return iter([article])

    monkeypatch.setattr(research.arxiv.Client, "results", client_results)
    provider = LiteLLMProvider(
        api_key="test", api_base="https://example.test", default_model="test", parameters=LLMParameters()
    )
    for tool in build_research_tools(tmp_path):
        provider.register_tool()(tool)
    assert set(provider.default_tools) == {"arxiv_search", "arxiv_download"}
    search_schema = provider.default_tools["arxiv_search"].tool_json["function"]
    assert search_schema["parameters"]["properties"]["max_results"]["maximum"] == 20
    executor = ToolExecutor(tmp_path, max_result_chars=10_000)
    results = await executor.execute(
        [ToolCallRequest(id="search-1", name="arxiv_search", arguments={"query": " AI "})],
        {name: tool.func for name, tool in provider.default_tools.items()},
    )
    assert results[0].succeeded
    assert json.loads(results[0].value) == [research._article_payload(article)]
    assert searches[0].query == "AI"
    assert searches[0].max_results == 10


@pytest.mark.asyncio
async def test_search_empty_and_network_failure(monkeypatch):
    monkeypatch.setattr(research, "_search_collect", lambda *_args: [])
    assert "未找到" in await research.arxiv_search(ArxivSearchArgs(query="AI"))

    def fail(*_args):
        raise OSError("offline")

    monkeypatch.setattr(research, "_search_collect", fail)
    with pytest.raises(RuntimeError, match="arXiv 检索失败：offline"):
        await research.arxiv_search(ArxivSearchArgs(query="AI"))


@pytest.mark.asyncio
async def test_download_normalizes_ids_saves_and_reuses_pdf(tmp_path, monkeypatch, article, pdf_client):
    def client_results(self, *, search):
        assert search.id_list == ["2103.03404v2"]
        return iter([article])

    monkeypatch.setattr(research.arxiv.Client, "results", client_results)
    _, download = build_research_tools(tmp_path)
    args = ArxivDownloadArgs(id_list=[" arXiv:2103.03404v2 ", "2103.03404v2"], download_dir="papers")
    assert args.id_list == ["2103.03404v2"]
    result = json.loads(await download(args))
    target = tmp_path / "papers" / "2103.03404v2.pdf"
    assert result[0]["pdf_path"] == str(target)
    assert target.read_bytes() == b"%PDF-1.4 example"
    assert len(list(target.parent.iterdir())) == 1
    await download(args)
    assert len(pdf_client) == 1


@pytest.mark.asyncio
async def test_download_old_style_id_and_partial_failure(tmp_path, monkeypatch, article, pdf_client):
    def lookup(paper_id):
        if paper_id == "2103.03404":
            raise ValueError("arXiv 上未找到该论文")
        return {**research._article_payload(article), "id": paper_id}

    monkeypatch.setattr(research, "_lookup_article", lookup)
    _, download = build_research_tools(tmp_path)
    result = await download(ArxivDownloadArgs(id_list=["2103.03404", "hep-th/9901001"], download_dir=str(tmp_path)))
    assert "2103.03404 下载失败" in result
    assert (tmp_path / "hep-th_9901001.pdf").is_file()


@pytest.mark.asyncio
async def test_download_timeout_and_total_failure(tmp_path, monkeypatch):
    async def hang(*_args):
        await asyncio.Event().wait()

    monkeypatch.setattr(research, "_download_one", hang)
    monkeypatch.setattr(research, "_DOWNLOAD_TIMEOUT_SECONDS", 0.01)
    _, download = build_research_tools(tmp_path)
    with pytest.raises(RuntimeError, match="下载失败：2103.03404：下载超时"):
        await download(ArxivDownloadArgs(id_list=["2103.03404"], download_dir="papers"))


@pytest.mark.asyncio
@pytest.mark.parametrize("directory", ["../outside", "symlink"])
async def test_download_protects_workspace(tmp_path, directory):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "symlink").symlink_to(tmp_path, target_is_directory=True)
    _, download = build_research_tools(workspace)
    with pytest.raises(PathProtectionError):
        await download(ArxivDownloadArgs(id_list=["2103.03404"], download_dir=directory))
    assert not (tmp_path / "outside").exists()


@pytest.mark.asyncio
async def test_download_rejects_pdf_symlink_outside_workspace(tmp_path, monkeypatch, article):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.pdf"
    (workspace / "2103.03404v2.pdf").symlink_to(outside)
    monkeypatch.setattr(research, "_lookup_article", lambda _paper_id: research._article_payload(article))
    _, download = build_research_tools(workspace)
    with pytest.raises(RuntimeError, match="Path escapes workspace"):
        await download(ArxivDownloadArgs(id_list=["2103.03404v2"], download_dir="."))
    assert not outside.exists()


@pytest.mark.asyncio
async def test_interrupted_download_does_not_cache_partial_pdf(tmp_path, monkeypatch):
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"%PDF-1.4 partial"
            raise httpx.ReadError("connection lost")

    client_cls = httpx.AsyncClient
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, stream=BrokenStream()))
    monkeypatch.setattr(research.httpx, "AsyncClient", lambda **kwargs: client_cls(transport=transport, **kwargs))
    target = tmp_path / "paper.pdf"
    with pytest.raises(httpx.ReadError, match="connection lost"):
        await research._download_pdf("https://arxiv.org/pdf/2103.03404", target)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
