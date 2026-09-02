import shutil
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from pydantic import BaseModel, Field

from .helper import safe_resolve_workspace_path
from ...utils.helper import is_relative_path


class BrowserPreviewArgs(BaseModel):
    path: str = Field(description="Workspace directory or HTML file to copy for browser preview")


def build_browser_preview_tool(
    workspace: str | Path,
    browser_preview_path: str | Path,
    browser_preview_base_url: str,
):
    base = Path(workspace).expanduser().resolve()
    if not base.exists():
        raise ValueError(f"Invalid workspace: {workspace}")
    target_root = Path(browser_preview_path).expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    base_url = browser_preview_base_url.rstrip("/")

    def create_browser_preview(args: BrowserPreviewArgs) -> str:
        """
        Copy a workspace directory or HTML file and return a browser preview URL.

        For a directory, the returned URL opens navigation for the copied directory;
        it does not select a particular file. To open a specific file, append its path
        relative to the copied directory, for example `<preview_url>pages/index.html`.
        For a single HTML file, the returned URL is the complete file URL.

        Args:
            path: A directory containing assets and HTML files, or a single HTML file.

        Returns:
            For a directory, a navigation URL with usage guidance. For a single HTML
            file, the complete file URL.
        """
        source = safe_resolve_workspace_path(
            args.path,
            base,
            protect_read_only_file=False,
        )
        if not source.exists():
            raise FileNotFoundError(f"Path not found: {args.path}")

        preview_id = uuid4()
        destination = target_root / str(preview_id)
        preview_url = f"{base_url}/{preview_id}/"
        if source.is_dir():
            shutil.copytree(source, destination)
            return (
                f"Directory preview URL: {preview_url}\n"
                "This URL opens navigation for the copied directory. To open a specific "
                "file, append its URL-encoded path relative to the copied directory, for "
                f"example: {preview_url}pages/index.html"
            )
        elif source.is_file() and source.suffix.lower() in {".html", ".htm"}:
            destination.mkdir()
            shutil.copy2(source, destination / source.name)
            return preview_url + quote(source.name)
        else:
            raise ValueError("Path must be a directory or a single HTML file")

    return create_browser_preview
