from typing import Annotated
from pathlib import Path
import os

import anyio
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse

from ....config.settings import settings
from ....exceptions import PathProtectionError
from ...tools.helper import safe_resolve_workspace_path
from ..dependencies import authenticated_user_id

router = APIRouter()


@router.get("/files/search")
async def search_workspace_files(
    user_id: Annotated[str, Depends(authenticated_user_id)],
    query: Annotated[str, Query(min_length=1, max_length=512)],
):
    """Search name prefixes recursively, returning at most 100 matching references."""
    workspace = (settings.path.workspace_base_path / user_id).resolve()
    parent, _, prefix = query.rpartition("/")
    if not prefix.strip():
        return {"entries": [], "truncated": False}
    try:
        directory = (workspace / parent).resolve()
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(422, "Invalid search path") from None
    if not directory.is_relative_to(workspace):
        raise HTTPException(403, "Directory is outside the workspace")

    def search():
        entries = []
        for root, directories, files in os.walk(directory, followlinks=False):
            directories.sort(key=str.casefold)
            for name in [*directories, *sorted(files, key=str.casefold)]:
                if not name.casefold().startswith(prefix.casefold()):
                    continue
                path = Path(root) / name
                try:
                    if not path.resolve().is_relative_to(workspace):
                        continue
                    is_directory = path.is_dir()
                    if not is_directory and not path.is_file():
                        continue
                except (OSError, RuntimeError):
                    continue
                entries.append({"name": name, "path": path.relative_to(workspace).as_posix(), "is_directory": is_directory})
                if len(entries) > 100:
                    return {"entries": entries[:100], "truncated": True}
        return {"entries": entries, "truncated": False}

    return await anyio.to_thread.run_sync(search)


@router.get("/files/list")
async def list_workspace_files(
    user_id: Annotated[str, Depends(authenticated_user_id)],
    path: str = "",
    query: Annotated[str, Query(max_length=255)] = "",
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Browse only the authenticated user's workspace; never follow escaping links."""
    workspace = (settings.path.workspace_base_path / user_id).resolve()
    try:
        directory = (workspace / path).resolve()
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(422, "Invalid directory path") from None
    if not directory.is_relative_to(workspace):
        raise HTTPException(403, "Directory is outside the workspace")

    def scan():
        if not directory.exists() and directory == workspace:
            return {"entries": [], "truncated": False}
        if not directory.is_dir():
            raise HTTPException(404, "Directory not found")
        entries = []
        try:
            for child in directory.iterdir():
                if not child.name.casefold().startswith(query.casefold()):
                    continue
                try:
                    resolved = child.resolve()
                except (OSError, RuntimeError):
                    continue
                if not resolved.is_relative_to(workspace):
                    continue
                is_directory = child.is_dir()
                if not is_directory and not child.is_file():
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

    return await anyio.to_thread.run_sync(scan)


@router.get("/files")
async def download_session_file(
    user_id: Annotated[str, Depends(authenticated_user_id)],
    path: Annotated[str, Query(description="File path returned by a send_file event.")],
):
    """下载指定会话工作区内的文件。"""
    workspace = (settings.path.workspace_base_path / user_id).resolve()
    try:
        file_path = safe_resolve_workspace_path(
            Path(path),
            workspace,
            protect_read_only_file=False,
        )
    except PathProtectionError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="File is outside the workspace")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    return FileResponse(file_path, filename=file_path.name)
