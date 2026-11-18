from datetime import UTC, datetime
from fnmatch import fnmatchcase
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from typing import Annotated

import anyio
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse

from ....exceptions import PathProtectionError
from ...tools.helper import safe_resolve_workspace_path
from ..dependencies import authenticated_user_id, authenticated_workspace

router = APIRouter()

SEARCH_EXCLUDED_NAMES = frozenset({
    ".git", ".hg", ".svn", ".idea", ".vscode", ".venv", "venv", ".local_lib", "node_modules", "__pycache__",
    ".cache", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".nox", ".npm", ".yarn",
    ".pnpm-store", ".ds_store", "thumbs.db", "site-packages", "dist-packages",
})
SEARCH_EXCLUDED_PATHS = frozenset({("pkg", "mod"), ("pkg", "sum")})


def _excluded_from_search(path: anyio.Path, workspace: anyio.Path) -> bool:
    parts = tuple(part.casefold() for part in path.relative_to(workspace).parts)
    return (
        any(part in SEARCH_EXCLUDED_NAMES for part in parts)
        or any(pair in SEARCH_EXCLUDED_PATHS for pair in zip(parts, parts[1:]))
    )


async def _search_path_allowed(path: anyio.Path, workspace: anyio.Path) -> bool:
    if _excluded_from_search(path, workspace):
        return False
    try:
        resolved = await path.resolve()
        return resolved.is_relative_to(workspace) and not _excluded_from_search(resolved, workspace)
    except (OSError, RuntimeError):
        return False


async def _workspace_entry(path: anyio.Path, workspace: anyio.Path) -> dict | None:
    try:
        info = await path.stat()
    except (OSError, RuntimeError):
        return None
    is_directory = S_ISDIR(info.st_mode)
    if not is_directory and not S_ISREG(info.st_mode):
        return None
    return {
        "name": path.name,
        "path": path.relative_to(workspace).as_posix(),
        "is_directory": is_directory,
        "modified_at": datetime.fromtimestamp(info.st_mtime, UTC).isoformat(),
        "size": None if is_directory else info.st_size,
    }


@router.get("/files/search")
async def search_workspace_files(
    user_id: Annotated[str, Depends(authenticated_user_id)],
    query: Annotated[str, Query(min_length=1, max_length=512)],
):
    """Match entry names in the user's workspace, pruning internal directories before descent."""
    workspace = await authenticated_workspace(user_id)
    if not query.strip():
        return {"entries": [], "truncated": False}
    relative = Path(query)
    if relative.is_absolute() or ".." in relative.parts:
        raise HTTPException(403, "Search path is outside the workspace")
    pattern = query.casefold()
    entries = []
    async for directory, dirs, files in workspace.walk(top_down=True, follow_symlinks=False):
        # 原地修改 dirs，让 walk 跳过这些目录及其全部子项。
        dirs[:] = [name for name in dirs if not _excluded_from_search(directory / name, workspace)]
        for name in dirs + files:
            if name.casefold() in SEARCH_EXCLUDED_NAMES or not fnmatchcase(name.casefold(), pattern):
                continue
            path = directory / name
            if not await _search_path_allowed(path, workspace):
                continue
            entry = await _workspace_entry(path, workspace)
            if entry is None:
                continue
            entries.append(entry)
            if len(entries) > 100:
                return {"entries": entries[:100], "truncated": True}
    return {"entries": entries, "truncated": False}


@router.get("/files/list")
async def list_workspace_files(
    user_id: Annotated[str, Depends(authenticated_user_id)],
    path: str = "",
    query: Annotated[str, Query(max_length=255)] = "",
    offset: Annotated[int, Query(ge=0)] = 0,
    initial_load: bool = False,
):
    """Browse only the authenticated user's workspace; never follow escaping links."""
    workspace = await authenticated_workspace(user_id)
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
    if initial_load and _excluded_from_search(directory, workspace):
        return {"entries": [], "truncated": False}
    entries = []
    try:
        async for child in directory.iterdir():
            if initial_load and _excluded_from_search(child, workspace):
                continue
            if not child.name.casefold().startswith(query.casefold()):
                continue
            try:
                resolved = await child.resolve()
            except (OSError, RuntimeError):
                continue
            if not resolved.is_relative_to(workspace):
                continue
            if initial_load and _excluded_from_search(resolved, workspace):
                continue
            entry = await _workspace_entry(child, workspace)
            if entry is None:
                continue
            entries.append(entry)
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
    workspace = Path(await authenticated_workspace(user_id))
    try:
        file_path = await anyio.to_thread.run_sync(
            lambda: safe_resolve_workspace_path(Path(path), workspace, protect_read_only_file=False),
        )
    except PathProtectionError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="File is outside the workspace")
    if not await anyio.Path(file_path).is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    return FileResponse(file_path, filename=file_path.name)
