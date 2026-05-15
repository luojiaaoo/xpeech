from ...utils.security.network import validate_url_target
from markitdown import MarkItDown
from pydantic import BaseModel, Field


class WebFetchArgs(BaseModel):
    url: str = Field(description="URL to fetch")


async def web_fetch(args: WebFetchArgs) -> str:
    """Fetch URL and extract readable content (HTML → markdown)."""
    is_valid, error_msg = validate_url_target(args.url)
    if not is_valid:
        return f"URL validation failed: {error_msg}"
    md = MarkItDown()
    result = md.convert(args.url)
    return f"""[TITLE: {result.title}]\n\n{result.text_content}"""
