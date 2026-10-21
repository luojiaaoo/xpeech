from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse

from ....config.settings import settings
from ....exceptions import PathProtectionError
from ....utils.user_id import validate_user_id
from ....utils.workspace import safe_resolve_workspace_path

router = APIRouter()


@router.get("/sessions/{user_id}/files")
async def download_session_file(
    user_id: str,
    path: Annotated[str, Query(description="File path returned by a send_file event.")],
):
    """下载指定会话工作区内的文件。"""
    try:
        workspace = (settings.path.workspace_base_path / validate_user_id(user_id)).resolve()
    except PathProtectionError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Invalid user_id") from None
    try:
        file_path = safe_resolve_workspace_path(
            path,
            workspace,
            protect_read_only_file=False,
        )
    except PathProtectionError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="File is outside the workspace")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    return FileResponse(file_path, filename=file_path.name)
