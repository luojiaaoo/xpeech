import asyncio
import json
from typing import Annotated
from xml.etree import ElementTree

import httpx
import trafilatura
from baidusearch.baidusearch import search as _baidu_search
from loguru import logger
from pydantic import BaseModel, Field, StringConstraints

from ...utils.security.network import validate_url_target

WEB_CONNECT_TIMEOUT_SECONDS = 10
WEB_READ_TIMEOUT_SECONDS = 60
_WEB_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
_SEARCH_TIMEOUT_SECONDS = 20.0
_RSS_RESPONSE_PREVIEW_CHARS = 200


class WebFetchArgs(BaseModel):
    url: str = Field(description="URL to fetch")
    include_images: bool = Field(
        default=False,
        description="是否在 Markdown 中保留图片链接和替代文字；需要图片信息时设为 true，不会下载图片或进行视觉识别",
    )


class BaiduSearchArgs(BaseModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)] = Field(
        description="搜索关键词，最多 200 字符"
    )
    max_results: int = Field(default=5, ge=1, le=10, strict=True, description="最多返回的结果条数，默认 5，范围 1-10")


class BingSearchArgs(BaseModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)] = Field(
        description="搜索关键词，最多 200 字符"
    )
    max_results: int = Field(default=5, ge=1, le=10, strict=True, description="最多返回的结果条数，默认 5，范围 1-10")


async def web_fetch(args: WebFetchArgs) -> str:
    """Fetch a webpage over HTTP and extract its main content and comments as Markdown, including links and tables.
    Set include_images=true when image links and alt text are useful for the task.
    Does not launch a browser or execute JavaScript. Returns plain text, Markdown, XML (including RSS/Atom), and JSON directly.
    """

    is_valid, error_msg = validate_url_target(args.url)
    if not is_valid:
        return f"URL validation failed: {error_msg}"
    return await _fetch_and_convert(args.url, include_images=args.include_images)


async def _fetch_and_convert(url: str, *, include_images: bool = False) -> str:
    async with httpx.AsyncClient(
        headers={
            "User-Agent": _WEB_USER_AGENT,
            "Accept": "text/markdown, text/html;q=0.9, text/plain;q=0.8, */*;q=0.1",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
        follow_redirects=True,
        timeout=httpx.Timeout(WEB_READ_TIMEOUT_SECONDS, connect=WEB_CONNECT_TIMEOUT_SECONDS),
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        return await asyncio.to_thread(_response_to_markdown, response, include_images=include_images)


def _response_to_markdown(response: httpx.Response, *, include_images: bool = False) -> str:
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if (content_type.startswith("text/") and content_type != "text/html") or content_type in {
        "application/xml",
        "application/rss+xml",
        "application/atom+xml",
        "application/json",
    }:
        return response.text
    if content_type and content_type not in {"text/html", "application/xhtml+xml"}:
        raise ValueError(
            f"Unsupported content type: {content_type}. Download documents and use read_doc_file instead."
        )

    url = str(response.url)
    tree = trafilatura.load_html(response.content)
    if tree is None:
        raise ValueError(f"No readable HTML found at {response.url}")
    title = (tree.findtext(".//title") or "").strip()
    if not title:
        metadata = trafilatura.extract_metadata(tree, default_url=url)
        title = metadata.title if metadata else None
    content = trafilatura.extract(
        tree,
        url=url,
        output_format="markdown",
        favor_recall=True,
        include_links=True,
        include_tables=True,
        include_comments=True,
        include_images=include_images,
    )
    if not content or not content.strip():
        raise ValueError(f"No readable content found at {response.url}")
    return f"[TITLE: {title}]\n\n{content}" if title else content


async def baidu_search(args: BaiduSearchArgs) -> str:
    """使用百度检索网页，适合中文互联网内容，返回标题、链接、摘要和排名。
    默认返回 5 条，最多 10 条，不支持时间过滤。需要正文时，用 web_fetch 读取结果链接。
    """
    try:
        async with asyncio.timeout(_SEARCH_TIMEOUT_SECONDS):
            items = await asyncio.to_thread(_baidu_search, args.query, args.max_results)
        hits = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "abstract": item.get("abstract", ""),
                "rank": str(index),
            }
            for index, item in enumerate((items or [])[:args.max_results], 1)
        ]
    except TimeoutError as error:
        raise RuntimeError("百度搜索失败：搜索超时，请稍后重试") from error
    except Exception as error:
        raise RuntimeError(f"百度搜索失败：{error}") from error
    if not hits:
        return f"未找到与「{args.query}」相关的搜索结果。"
    return json.dumps(hits, ensure_ascii=False, indent=2)


async def _fetch_bing_rss(url: str, params: dict[str, str], *, source: str) -> ElementTree.Element:
    async with httpx.AsyncClient(
        headers={"User-Agent": _WEB_USER_AGENT},
        follow_redirects=True,
        timeout=WEB_CONNECT_TIMEOUT_SECONDS,
    ) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        try:
            root = ElementTree.fromstring(response.content)
            if root.tag != "rss" or root.find("channel") is None:
                raise ValueError("Expected an RSS root with a channel element")
        except (ElementTree.ParseError, ValueError) as error:
            text = response.text
            logger.warning(
                "{} returned invalid RSS: status={} url={} content_type={} redirects={} "
                "body_bytes={} parse_error={} body_head={!r} body_tail={!r}",
                source,
                response.status_code,
                response.url,
                response.headers.get("Content-Type", ""),
                [(item.status_code, item.headers.get("Location", "")) for item in response.history],
                len(response.content),
                str(error),
                text[:_RSS_RESPONSE_PREVIEW_CHARS],
                text[-_RSS_RESPONSE_PREVIEW_CHARS:],
            )
            raise ValueError(f"{source} 未返回有效 RSS，可能被重定向到首页或验证页，请稍后重试") from error
    return root


async def bing_search(args: BingSearchArgs) -> str:
    """使用必应检索网页，返回标题、链接、摘要和排名。
    默认返回 5 条，最多 10 条，不支持时间过滤。需要正文时，用 web_fetch 读取结果链接。
    """
    try:
        async with asyncio.timeout(_SEARCH_TIMEOUT_SECONDS):
            root = await _fetch_bing_rss(
                "https://cn.bing.com/search",
                {"q": args.query, "format": "rss"},
                source="Bing",
            )
        hits = [
            {
                "title": item.findtext("title", ""),
                "url": item.findtext("link", ""),
                "abstract": item.findtext("description", ""),
                "rank": str(index),
            }
            for index, item in enumerate(root.findall("./channel/item")[:args.max_results], 1)
        ]
    except TimeoutError as error:
        raise RuntimeError("Bing搜索失败：搜索超时，请稍后重试") from error
    except Exception as error:
        raise RuntimeError(f"Bing搜索失败：{error}") from error
    if not hits:
        return f"未找到与「{args.query}」相关的搜索结果。"
    return json.dumps(hits, ensure_ascii=False, indent=2)
