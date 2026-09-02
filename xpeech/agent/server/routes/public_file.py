from html import escape
from pathlib import Path
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse, HTMLResponse

from ....config.settings import settings

router = APIRouter()

_NO_STORE_HEADERS = {"Cache-Control": "no-store"}


def _preview_url_path(preview_id: UUID, relative_path: Path, *, is_directory: bool) -> str:
    encoded_path = "/".join(quote(part, safe="") for part in relative_path.parts)
    suffix = f"{encoded_path}/" if encoded_path and is_directory else encoded_path
    return f"{settings.tool.browser_preview.route_path}/{preview_id}/{suffix}"


def _render_directory_navigation(preview_id: UUID, directory: Path, relative_path: Path) -> str:
    display_path = "/" if not relative_path.parts else f"/{relative_path.as_posix()}/"
    links: list[str] = []

    if relative_path.parts:
        parent_url = _preview_url_path(preview_id, relative_path.parent, is_directory=True)
        links.append(f'<li><a href="{escape(parent_url, quote=True)}">../</a></li>')

    entries = sorted(
        directory.iterdir(),
        key=lambda entry: (not entry.is_dir(), entry.name.casefold(), entry.name),
    )
    for entry in entries:
        is_directory = entry.is_dir()
        entry_url = _preview_url_path(
            preview_id,
            relative_path / entry.name,
            is_directory=is_directory,
        )
        label = f"{entry.name}/" if is_directory else entry.name
        links.append(
            f'<li><a href="{escape(entry_url, quote=True)}">{escape(label)}</a></li>'
        )

    if not entries:
        links.append("<li><em>This directory is empty.</em></li>")

    escaped_display_path = escape(display_path)
    return (
        "<!doctype html>"
        '<html lang="en">'
        "<head>"
        '<meta charset="utf-8">'
        f"<title>Index of {escaped_display_path}</title>"
        "</head>"
        "<body>"
        f"<h1>Index of {escaped_display_path}</h1>"
        f"<ul>{''.join(links)}</ul>"
        "</body>"
        "</html>"
    )


@router.get(
    f"{settings.tool.browser_preview.route_path}/{{preview_id}}/{{file_path:path}}",
    include_in_schema=False,
)
async def preview_file(preview_id: UUID, file_path: str):
    """返回浏览器预览目录内经过路径校验的文件或目录导航。"""
    relative_path = Path(file_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid preview path")

    preview_root = (settings.tool.browser_preview.browser_preview_path / str(preview_id)).resolve()
    preview_file_path = (preview_root / relative_path).resolve()
    if not preview_file_path.is_relative_to(preview_root):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid preview path")
    if not preview_file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview file not found")
    if preview_file_path.is_dir():
        return HTMLResponse(
            _render_directory_navigation(preview_id, preview_file_path, relative_path),
            headers=_NO_STORE_HEADERS,
        )
    if preview_file_path.is_file():
        return FileResponse(preview_file_path, headers=_NO_STORE_HEADERS)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview file not found")
