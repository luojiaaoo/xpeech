import asyncio
from urllib.parse import urlencode

import requests
from markitdown import MarkItDown
from pydantic import BaseModel, Field
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import SSLError, Timeout

from ...utils.security.network import validate_url_target

WEB_CONNECT_TIMEOUT_SECONDS = 10
WEB_READ_TIMEOUT_SECONDS = 60


class WebFetchArgs(BaseModel):
    url: str = Field(description="URL to fetch")


class WebSearchArgs(BaseModel):
    keyword: str = Field(description="Keyword to research")


async def web_fetch(args: WebFetchArgs) -> str:
    """Fetch URL and extract readable content (HTML → markdown) without images."""

    is_valid, error_msg = validate_url_target(args.url)
    if not is_valid:
        return f"URL validation failed: {error_msg}"
    return await asyncio.to_thread(_fetch_and_convert, args.url)


def _fetch_and_convert(url: str) -> str:
    with requests.Session() as session:
        session.headers.update(
            {"Accept": "text/markdown, text/html;q=0.9, text/plain;q=0.8, */*;q=0.1"}
        )
        with session.get(
            url,
            stream=True,
            timeout=(WEB_CONNECT_TIMEOUT_SECONDS, WEB_READ_TIMEOUT_SECONDS),
        ) as response:
            response.raise_for_status()
            result = MarkItDown().convert(response)
    return f"[TITLE: {result.title}]\n\n{result.text_content}"


async def web_search(args: WebSearchArgs) -> str:
    """
    Use Bing to search the web for the given keyword.
    Retrieve the top general results and the top news results.
    For each result, return only the URL and the snippet/summary.
    Then, read the full content of each page using web_fetch.
    """

    general_url = "https://www.bing.com/search?" + urlencode({"q": args.keyword})
    news_url = "https://www.bing.com/news/search?" + urlencode({"q": args.keyword})
    async def fetch_search_page(url: str, label: str) -> str:
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                return await web_fetch(WebFetchArgs(url=url))
            except (SSLError, RequestsConnectionError, Timeout) as exc:
                if attempt == max_attempts - 1:
                    return f"{label} search temporarily unavailable after {max_attempts} attempts: {exc}"
                await asyncio.sleep(0.5 * (2**attempt))
        return f"{label} search temporarily unavailable"

    async with asyncio.TaskGroup() as tg:
        general_task = tg.create_task(fetch_search_page(general_url, "General"))
        news_task = tg.create_task(fetch_search_page(news_url, "News"))
    general_content, news_content = general_task.result(), news_task.result()
    return f"Search results for '{args.keyword}':\n\n- {general_content}\n- {news_content}" 
    
