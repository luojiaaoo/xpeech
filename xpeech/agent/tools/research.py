"""arXiv 论文检索和 PDF 下载工具。"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
from pathlib import Path
from typing import Annotated, Any

import aiofiles
import arxiv
import httpx
from pydantic import BaseModel, Field, StringConstraints, field_validator

from .helper import safe_resolve_workspace_path

_MAX_RESULTS_LIMIT = 20
_MAX_ID_LIST = 5
_SEARCH_TIMEOUT_SECONDS = 30.0
_DOWNLOAD_TIMEOUT_SECONDS = 120.0
_HTTP_TIMEOUT_SECONDS = 30.0
_ARXIV_ID_PATTERN = re.compile(r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Za-z-]+)?/\d{7})(?:v\d+)?")


class ArxivSearchArgs(BaseModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)] = Field(
        description="搜索关键词或 arXiv 查询语法"
    )
    max_results: int = Field(default=10, ge=1, le=_MAX_RESULTS_LIMIT, strict=True, description="最多返回的论文篇数")


class ArxivDownloadArgs(BaseModel):
    id_list: list[str] = Field(
        min_length=1,
        max_length=_MAX_ID_LIST,
        description="arXiv 论文 ID 列表，一次最多 5 篇，例如 2103.03404、2103.03404v2、hep-th/9901001",
    )
    download_dir: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)] = Field(
        description="工作区内的 PDF 保存目录，支持相对路径、绝对路径和 ~/，不存在时自动创建"
    )

    @field_validator("id_list")
    @classmethod
    def normalize_ids(cls, values: list[str]) -> list[str]:
        normalized = []
        for raw in values:
            paper_id = raw.strip().removeprefix("arXiv:").removeprefix("arxiv:").strip()
            if not _ARXIV_ID_PATTERN.fullmatch(paper_id):
                raise ValueError(f"论文 ID 格式无效：{raw}")
            normalized.append(paper_id)
        return list(dict.fromkeys(normalized))


def _article_payload(result: arxiv.Result) -> dict[str, Any]:
    """把 arXiv Result 归一化成可 JSON 序列化的论文元数据。"""
    return {
        "title": result.title,
        "id": result.get_short_id(),
        "entry_id": result.entry_id,
        "authors": [author.name for author in result.authors],
        "primary_category": result.primary_category,
        "categories": list(result.categories),
        "published": result.published.isoformat() if result.published else None,
        "pdf_url": result.pdf_url,
        "links": [link.href for link in result.links],
        "summary": result.summary,
        "comment": result.comment,
    }


def _search_collect(query: str, max_results: int) -> list[dict[str, Any]]:
    client = arxiv.Client(page_size=_MAX_RESULTS_LIMIT)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )
    return [_article_payload(result) for result in client.results(search=search)]


def _lookup_article(paper_id: str) -> dict[str, Any]:
    client = arxiv.Client()
    result = next(client.results(search=arxiv.Search(id_list=[paper_id])), None)
    if result is None:
        raise ValueError("arXiv 上未找到该论文")
    payload = _article_payload(result)
    if not payload["pdf_url"]:
        raise ValueError("该论文没有 PDF 链接")
    return payload


async def _download_pdf(pdf_url: str, target: Path) -> Path:
    """下载完成后原子落盘，避免失败或超时留下的半成品被复用。"""
    if target.is_file() and target.stat().st_size > 0:
        return target
    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=f".{target.name}.", suffix=".part", delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        async with (
            httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS, follow_redirects=True) as client,
            client.stream("GET", pdf_url) as response,
        ):
            response.raise_for_status()
            async with aiofiles.open(temporary, "wb") as file:
                async for chunk in response.aiter_bytes():
                    await file.write(chunk)
        if temporary.stat().st_size == 0:
            raise ValueError("下载的 PDF 为空")
        temporary.replace(target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


async def _download_one(paper_id: str, download_dir: Path, workspace: Path) -> dict[str, Any]:
    payload = await asyncio.to_thread(_lookup_article, paper_id)
    safe_name = re.sub(r"[^\w.-]", "_", payload["id"])
    target = safe_resolve_workspace_path(download_dir / f"{safe_name}.pdf", workspace)
    payload["pdf_path"] = str(await _download_pdf(payload["pdf_url"], target))
    return payload


async def arxiv_search(args: ArxivSearchArgs) -> str:
    """在 arXiv 检索学术论文，返回标题、ID、作者、分类、发布时间、PDF 链接和摘要。支持关键词和 arXiv 查询语法，默认 10 篇，最多 20 篇。"""
    try:
        async with asyncio.timeout(_SEARCH_TIMEOUT_SECONDS):
            articles = await asyncio.to_thread(_search_collect, args.query, args.max_results)
    except TimeoutError as error:
        raise RuntimeError("arXiv 检索失败：检索超时，请稍后重试") from error
    except Exception as error:
        raise RuntimeError(f"arXiv 检索失败：{error}") from error
    if not articles:
        return f"未找到与「{args.query}」相关的 arXiv 论文。"
    return json.dumps(articles, ensure_ascii=False, indent=2)


def build_research_tools(workspace: str | Path):
    """构建绑定当前工作区的科研工具。"""
    base = Path(workspace).expanduser().resolve()

    async def arxiv_download(args: ArxivDownloadArgs) -> str:
        """按 arXiv 论文 ID 下载 PDF 到工作区内指定目录，返回论文元数据和本地路径。一次最多 5 篇；目录不存在时自动创建，单篇失败不影响其他论文。"""
        download_dir = safe_resolve_workspace_path(args.download_dir, base)
        download_dir.mkdir(parents=True, exist_ok=True)

        async def collect(paper_id: str) -> dict[str, Any] | str:
            try:
                async with asyncio.timeout(_DOWNLOAD_TIMEOUT_SECONDS):
                    return await _download_one(paper_id, download_dir, base)
            except TimeoutError:
                return "下载超时"
            except Exception as error:  # noqa: BLE001 - 单篇失败不能取消其他论文下载
                return str(error)

        results = await asyncio.gather(*(collect(paper_id) for paper_id in args.id_list))
        articles = [result for result in results if isinstance(result, dict)]
        errors = {paper_id: result for paper_id, result in zip(args.id_list, results) if isinstance(result, str)}
        if not articles:
            detail = "；".join(f"{paper_id}：{message}" for paper_id, message in errors.items())
            raise RuntimeError(f"下载失败：{detail}")
        payload = json.dumps(articles, ensure_ascii=False, indent=2)
        if errors:
            notes = "\n".join(
                f"（注意：{paper_id} 下载失败：{message}，以下结果不含该篇）" for paper_id, message in errors.items()
            )
            return f"{notes}\n{payload}"
        return payload

    return arxiv_search, arxiv_download
