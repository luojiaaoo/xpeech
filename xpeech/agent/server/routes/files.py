from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse

from ....config.settings import settings
from ....exceptions import PathProtectionError
from ...tools.helper import safe_resolve_workspace_path

router = APIRouter()
preview_router = APIRouter()


@router.get("/sessions/{session_id}/files")
async def download_session_file(
    session_id: str,
    path: Annotated[str, Query(description="File path returned by a send_file event.")],
):
    """下载指定会话工作区内的文件。"""
    workspace = (settings.path.workspace_base_path / session_id).resolve()
    try:
        file_path = safe_resolve_workspace_path(
            path,
            workspace,
            protect_builtin_skills=False,
        )
    except PathProtectionError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="File is outside the workspace")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    return FileResponse(file_path, filename=file_path.name)


@preview_router.get(
    f"{settings.tool.browser_preview.route_path}/{{preview_id}}/{{file_path:path}}",
    include_in_schema=False,
)
async def preview_file(preview_id: UUID, file_path: str):
    """返回浏览器预览目录内经过路径校验的文件。"""
    relative_path = Path(file_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid preview path")
    preview_file_path = settings.tool.browser_preview.browser_preview_path / str(preview_id) / relative_path
    if not preview_file_path.exists() or not preview_file_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview file not found")
    return FileResponse(preview_file_path, headers={"Cache-Control": "no-store"})
