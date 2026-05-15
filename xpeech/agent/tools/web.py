from ...utils.security.network import validate_url_target
from markitdown import MarkItDown
from pydantic import BaseModel, Field
import asyncio
from urllib.parse import urlencode


class WebFetchArgs(BaseModel):
    url: str = Field(description="URL to fetch")


class WebSearchArgs(BaseModel):
    keyword: str = Field(description="Keyword to research")


async def web_fetch(args: WebFetchArgs) -> str:
    """Fetch URL and extract readable content (HTML → markdown)."""

    is_valid, error_msg = validate_url_target(args.url)
    if not is_valid:
        return f"URL validation failed: {error_msg}"
    md = MarkItDown()
    result = await asyncio.to_thread(md.convert, args.url)
    return f"""[TITLE: {result.title}]\n\n{result.text_content}"""


async def web_search(args: WebSearchArgs) -> str:
    """
    Search the web for the given keyword and return the top general and news results.
    only retrieve the URL and the summary; you need to further read the content using web_fetch.
    """

    general_url = "https://www.bing.com/search?" + urlencode({"q": args.keyword})
    news_url = "https://www.bing.com/news/search?" + urlencode({"q": args.keyword})
    general_content = await web_fetch(WebFetchArgs(url=general_url))
    news_content = await web_fetch(WebFetchArgs(url=news_url))
    return f"Search results for '{args.keyword}':\n\n- {general_content}\n- {news_content}" 
    
