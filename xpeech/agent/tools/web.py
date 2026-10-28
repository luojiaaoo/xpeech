import asyncio
import json
from typing import Annotated, Any, Literal, Self

import requests
from baidusearch.baidusearch import search as baidu_search
from ddgs import DDGS
from markitdown import MarkItDown
from pydantic import BaseModel, Field, StringConstraints, model_validator

from ...utils.security.network import validate_url_target

WEB_CONNECT_TIMEOUT_SECONDS = 10
WEB_READ_TIMEOUT_SECONDS = 60
_SEARCH_TIMEOUT_SECONDS = 20.0
_DDGS_TIMEOUT = 10
_DDGS_REGION = "cn-zh"


class WebFetchArgs(BaseModel):
    url: str = Field(description="URL to fetch")


class WebSearchArgs(BaseModel):
    keyword: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)] = Field(
        description="搜索关键词，最多 200 字符"
    )
    engine: Literal["baidu", "duckduckgo", "news"] = Field(
        default="duckduckgo",
        description="搜索引擎：baidu 适合中文内容，duckduckgo 适合国际内容，news 聚合新闻源",
    )
    max_results: int = Field(default=5, ge=1, le=10, strict=True, description="最多返回的结果条数，默认 5，范围 1-10")
    timelimit: Literal["d", "w", "m", "y"] | None = Field(
        default=None,
        description="时间过滤：d=一天内、w=一周内、m=一月内、y=一年内；省略则不限时间。百度不支持此参数",
    )

    @model_validator(mode="after")
    def validate_timelimit(self) -> Self:
        if self.engine == "baidu" and self.timelimit is not None:
            raise ValueError("百度搜索不支持 timelimit，请使用 duckduckgo 或 news")
        return self


async def web_fetch(args: WebFetchArgs) -> str:
    """Fetch URL and extract readable content (HTML → markdown) without images."""

    is_valid, error_msg = validate_url_target(args.url)
    if not is_valid:
        return f"URL validation failed: {error_msg}"
    return await asyncio.to_thread(_fetch_and_convert, args.url)


def _fetch_and_convert(url: str) -> str:
    with requests.Session() as session:
        session.headers.update({"Accept": "text/markdown, text/html;q=0.9, text/plain;q=0.8, */*;q=0.1"})
        with session.get(
            url,
            stream=True,
            timeout=(WEB_CONNECT_TIMEOUT_SECONDS, WEB_READ_TIMEOUT_SECONDS),
        ) as response:
            response.raise_for_status()
            result = MarkItDown().convert(response)
    return f"[TITLE: {result.title}]\n\n{result.text_content}"


def _ddgs_search(category: str, query: str, max_results: int, timelimit: str | None) -> list[dict[str, Any]]:
    """在线程中调用 DDGS 网页或新闻搜索。"""
    with DDGS(timeout=_DDGS_TIMEOUT) as ddgs:
        kwargs: dict[str, Any] = {
            "max_results": max_results,
            "backend": "duckduckgo" if category == "text" else "auto",
            "region": _DDGS_REGION,
        }
        if timelimit:
            kwargs["timelimit"] = timelimit
        method = ddgs.text if category == "text" else ddgs.news
        return list(method(query, **kwargs) or [])


def _collect_search_results(args: WebSearchArgs) -> list[dict[str, str]]:
    """归一化不同搜索源的字段，并限制条数、添加排名。"""
    if args.engine == "baidu":
        items = baidu_search(args.keyword, args.max_results) or []
        fields = {"title": "title", "url": "url", "abstract": "abstract"}
    elif args.engine == "news":
        items = _ddgs_search("news", args.keyword, args.max_results, args.timelimit)
        fields = {"title": "title", "url": "url", "abstract": "body", "date": "date", "publisher": "source"}
    else:
        items = _ddgs_search("text", args.keyword, args.max_results, args.timelimit)
        fields = {"title": "title", "url": "href", "abstract": "body"}
    return [
        {**{target: item.get(source, "") for target, source in fields.items()}, "rank": str(index)}
        for index, item in enumerate(items[: args.max_results], 1)
    ]


async def web_search(args: WebSearchArgs) -> str:
    """使用百度、DuckDuckGo 或聚合新闻源搜索，返回标题、链接、摘要和排名；新闻还包含日期、媒体来源。
    默认使用 DuckDuckGo，最多返回 10 条，支持时间过滤（百度除外）。需要正文时，用 web_fetch 读取结果链接。
    """
    label = {"baidu": "百度", "duckduckgo": "DuckDuckGo", "news": "新闻"}[args.engine]
    try:
        async with asyncio.timeout(_SEARCH_TIMEOUT_SECONDS):
            hits = await asyncio.to_thread(_collect_search_results, args)
    except TimeoutError as error:
        raise RuntimeError(f"{label}搜索失败：搜索超时，请稍后重试") from error
    except Exception as error:
        raise RuntimeError(f"{label}搜索失败：{error}") from error
    if not hits:
        return f"未找到与「{args.keyword}」相关的{'新闻' if args.engine == 'news' else '搜索结果'}。"
    return json.dumps(hits, ensure_ascii=False, indent=2)
