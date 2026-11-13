from typing import Annotated
from pathlib import Path

import anyio
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse

from ....config.settings import settings
from ....exceptions import PathProtectionError
from ...tools.helper import safe_resolve_workspace_path
from ..dependencies import authenticated_user_id

router = APIRouter()

SEARCH_EXCLUDED_NAMES = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", ".local_lib", "node_modules", "__pycache__",
    ".cache", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".nox", ".npm", ".yarn",
    ".pnpm-store", ".ds_store", "thumbs.db",
})


def _excluded_from_search(path: anyio.Path, workspace: anyio.Path) -> bool:
    return any(part.casefold() in SEARCH_EXCLUDED_NAMES for part in path.relative_to(workspace).parts)


async def _user_workspace(user_id: str) -> anyio.Path:
    root = await anyio.Path(settings.path.workspace_base_path).resolve()
    workspace = root / user_id
    resolved = await workspace.resolve()
    if resolved != workspace:
        raise HTTPException(403, "User workspace must not redirect to another directory")
    return resolved


async def _search_path_allowed(path: anyio.Path, workspace: anyio.Path) -> bool:
    if _excluded_from_search(path, workspace):
        return False
    try:
        resolved = await path.resolve()
        return resolved.is_relative_to(workspace) and not _excluded_from_search(resolved, workspace)
    except (OSError, RuntimeError):
        return False


@router.get("/files/search")
async def search_workspace_files(
    user_id: Annotated[str, Depends(authenticated_user_id)],
    query: Annotated[str, Query(min_length=1, max_length=512)],
):
    """Return up to 100 native glob matches inside the authenticated user's workspace."""
    workspace = await _user_workspace(user_id)
    if not query.strip():
        return {"entries": [], "truncated": False}
    relative = Path(query)
    if relative.is_absolute() or ".." in relative.parts:
        raise HTTPException(403, "Search path is outside the workspace")
    if any(part.casefold() in SEARCH_EXCLUDED_NAMES for part in relative.parts):
        return {"entries": [], "truncated": False}

    entries = []
    seen = set()
    try:
        async for path in workspace.glob(query, case_sensitive=False):
            if path == workspace or path in seen or not await _search_path_allowed(path, workspace):
                continue
            is_directory = await path.is_dir()
            if not is_directory and not await path.is_file():
                continue
            seen.add(path)
            entries.append({"name": path.name, "path": path.relative_to(workspace).as_posix(), "is_directory": is_directory})
            if len(entries) > 100:
                return {"entries": entries[:100], "truncated": True}
    except ValueError:
        raise HTTPException(422, "Invalid glob pattern") from None
    return {"entries": entries, "truncated": False}


@router.get("/files/list")
async def list_workspace_files(
    user_id: Annotated[str, Depends(authenticated_user_id)],
    path: str = "",
    query: Annotated[str, Query(max_length=255)] = "",
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Browse only the authenticated user's workspace; never follow escaping links."""
    workspace = await _user_workspace(user_id)
    try:
        directory = await (workspace / path).resolve()
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(422, "Invalid directory path") from None
    if not directory.is_relative_to(workspace):
        raise HTTPException(403, "Directory is outside the workspace")

    if directory == workspace and not await directory.exists():
        return {"entries": [], "truncated": False}
    if not await directory.is_dir():
        raise HTTPException(404, "Directory not found")
    entries = []
    try:
        async for child in directory.iterdir():
            if not child.name.casefold().startswith(query.casefold()):
                continue
            try:
                resolved = await child.resolve()
            except (OSError, RuntimeError):
                continue
            if not resolved.is_relative_to(workspace):
                continue
            is_directory = await child.is_dir()
            if not is_directory and not await child.is_file():
                continue
            entries.append({
                "name": child.name,
                "path": child.relative_to(workspace).as_posix(),
                "is_directory": is_directory,
            })
    except OSError:
        raise HTTPException(403, "Directory cannot be read") from None
    entries.sort(key=lambda item: (not item["is_directory"], item["name"].casefold()))
    return {"entries": entries[offset:offset + 100], "truncated": len(entries) > offset + 100}


@router.get("/files")
async def download_session_file(
    user_id: Annotated[str, Depends(authenticated_user_id)],
    path: Annotated[str, Query(description="File path returned by a send_file event.")],
):
    """下载指定会话工作区内的文件。"""
    workspace = await _user_workspace(user_id)
    try:
        file_path = await anyio.to_thread.run_sync(
            lambda: safe_resolve_workspace_path(Path(path), Path(workspace), protect_read_only_file=False),
        )
    except PathProtectionError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="File is outside the workspace")
    if not await anyio.Path(file_path).is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    return FileResponse(file_path, filename=file_path.name)
