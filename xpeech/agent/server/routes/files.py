from typing import Annotated
from pathlib import Path
from fnmatch import fnmatchcase
from functools import cache
from glob import has_magic

import anyio
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse

from ....config.settings import settings
from ....exceptions import PathProtectionError
from ...tools.helper import safe_resolve_workspace_path
from ..dependencies import authenticated_user_id

router = APIRouter()


async def _user_workspace(user_id: str) -> anyio.Path:
    root = await anyio.Path(settings.path.workspace_base_path).resolve()
    workspace = root / user_id
    resolved = await workspace.resolve()
    if resolved != workspace:
        raise HTTPException(403, "User workspace must not redirect to another directory")
    return resolved


def _matches_path_glob(path: tuple[str, ...], pattern: tuple[str, ...]) -> bool:
    @cache
    def matches(path_index: int, pattern_index: int) -> bool:
        if pattern_index == len(pattern):
            return path_index == len(path)
        if pattern[pattern_index] == "**":
            return matches(path_index, pattern_index + 1) or (
                path_index < len(path) and matches(path_index + 1, pattern_index)
            )
        return (
            path_index < len(path)
            and fnmatchcase(path[path_index], pattern[pattern_index])
            and matches(path_index + 1, pattern_index + 1)
        )

    return matches(0, 0)


@router.get("/files/search")
async def search_workspace_files(
    user_id: Annotated[str, Depends(authenticated_user_id)],
    query: Annotated[str, Query(min_length=1, max_length=512)],
):
    """Search filename or relative-path globs, returning at most 100 matching references."""
    workspace = await _user_workspace(user_id)
    pattern = tuple(query.casefold().split("/"))
    if not pattern[-1].strip():
        return {"entries": [], "truncated": False}
    if query.startswith("/") or ".." in pattern:
        raise HTTPException(403, "Search path is outside the workspace")
    parent_parts = []
    for part in query.split("/")[:-1]:
        if has_magic(part):
            break
        parent_parts.append(part)
    try:
        directory = await workspace.joinpath(*parent_parts).resolve()
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(422, "Invalid search path") from None
    if not directory.is_relative_to(workspace):
        raise HTTPException(403, "Directory is outside the workspace")

    entries = []
    pending = [directory]
    while pending:
        current = pending.pop()
        try:
            children = sorted([child async for child in current.iterdir()], key=lambda child: child.name.casefold())
        except OSError:
            continue
        subdirectories = []
        for child in children:
            try:
                if not (await child.resolve()).is_relative_to(workspace):
                    continue
                is_directory = await child.is_dir()
                if is_directory:
                    if not await child.is_symlink():
                        subdirectories.append(child)
                elif not await child.is_file():
                    continue
            except (OSError, RuntimeError):
                continue
            relative = child.relative_to(workspace)
            if not (
                fnmatchcase(child.name.casefold(), pattern[0]) if len(pattern) == 1
                else _matches_path_glob(tuple(part.casefold() for part in relative.parts), pattern)
            ):
                continue
            entries.append({"name": child.name, "path": relative.as_posix(), "is_directory": is_directory})
            if len(entries) > 100:
                return {"entries": entries[:100], "truncated": True}
        pending.extend(reversed(subdirectories))
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
