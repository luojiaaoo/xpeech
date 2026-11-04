from typing import Annotated
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse

from ....config.settings import settings
from ....exceptions import PathProtectionError
from ...tools.helper import safe_resolve_workspace_path
from ..dependencies import authenticated_user_id

router = APIRouter()


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
