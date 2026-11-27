from pathlib import Path
from typing import Annotated

import anyio
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from ....config.settings import settings
from ....exceptions import PathProtectionError
from ....utils.channel_jwt import UserJWTAuth
from ....utils.file_io import ensure_path
from ....utils.helper import resolve_path_without_symlinks
from ...skills.skill import BUILTIN_SKILLS_DIR, SkillInfo, SkillsLoader
from ..auth import require_user_jwt
from ..dependencies import authenticated_user_id, authenticated_workspace

router = APIRouter(prefix="/skills", tags=["skills"])

MAX_SKILL_FILE_SIZE = 25 * 1024 * 1024


def _import_skill_file(data: bytes, workspace: Path) -> SkillInfo:
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("SKILL.md must be UTF-8 encoded") from exc
    skill = SkillInfo(source="workspace", sandbox_filepath="", content=content)
    name = skill.metadata.name
    skills_dir = workspace / "skills"
    destination = skills_dir / name
    if destination.parent != skills_dir or destination.name == "..":
        raise ValueError("Skill destination must be a direct child of the skills directory")
    destination = resolve_path_without_symlinks(destination)
    if (BUILTIN_SKILLS_DIR / destination.name / "SKILL.md").is_file():
        raise ValueError(f"Built-in skill '{name}' cannot be overwritten")

    ensure_path(skills_dir)
    try:
        destination.mkdir()
    except FileExistsError:
        raise FileExistsError(name) from None
    skill_file = destination / "SKILL.md"
    skill.sandbox_filepath = str(skill_file)
    try:
        skill_file.write_bytes(data)
    except OSError:
        skill_file.unlink(missing_ok=True)
        destination.rmdir()
        raise
    return skill


@router.get("", response_model=list[SkillInfo])
async def list_skills(
    auth: Annotated[UserJWTAuth, Depends(require_user_jwt)],
    user_id: Annotated[str, Depends(authenticated_user_id)],
) -> list[SkillInfo]:
    """返回内置技能，以及 JWT 用户可用的 SKILL 列表。"""
    workspace = Path(await authenticated_workspace(user_id))
    user_organization = await settings.extra_user_info.resolve_user_organization(user_id)
    try:
        return await SkillsLoader(
            workspace, user_id=user_id, channel=auth.channel, user_organization=user_organization,
        ).list_skill_info()
    except PathProtectionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from None


@router.post("/import", response_model=SkillInfo, status_code=status.HTTP_201_CREATED)
async def import_skill(
    user_id: Annotated[str, Depends(authenticated_user_id)],
    file: Annotated[UploadFile, File(description="SKILL.md 文件")],
) -> SkillInfo:
    """将用户上传的 SKILL.md 导入指定用户工作区。"""
    workspace = Path(await authenticated_workspace(user_id))
    if file.filename != "SKILL.md":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded file must be named SKILL.md",
        )

    data = await file.read(MAX_SKILL_FILE_SIZE + 1)
    if len(data) > MAX_SKILL_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="SKILL.md exceeds the 25 MiB upload limit",
        )
    try:
        return await anyio.to_thread.run_sync(_import_skill_file, data, workspace)
    except PathProtectionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from None
    except FileExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Skill '{exc.args[0]}' already exists",
        ) from None
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from None
