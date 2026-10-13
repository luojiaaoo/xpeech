from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from ....config.settings import settings
from ....exceptions import PathProtectionError
from ....utils.helper import is_relative_path
from ....utils.user_id import MAX_USER_ID_LENGTH, validate_user_id
from ...skills.skill import SkillInfo, SkillsLoader

router = APIRouter(prefix="/skills", tags=["skills"])

MAX_SKILL_FILE_SIZE = 25 * 1024 * 1024


def _user_workspace(user_id: str) -> Path:
    """Resolve a user workspace without allowing path traversal."""
    try:
        validated_user_id = validate_user_id(user_id)
    except PathProtectionError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid user_id",
        ) from None

    workspace_root = settings.path.workspace_base_path.expanduser().resolve()
    workspace = (workspace_root / validated_user_id).resolve()
    if not is_relative_path(workspace, workspace_root):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid user_id",
        )
    return workspace


def _import_skill_file(data: bytes, workspace: Path) -> SkillInfo:
    loader = SkillsLoader(workspace)
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("SKILL.md must be UTF-8 encoded") from exc
    skill = SkillInfo(source="workspace", path="", content=content)
    name = skill.metadata.name
    skills_dir = (workspace / "skills").resolve()
    destination = (skills_dir / name).resolve()
    if destination.parent != skills_dir:
        raise ValueError("Skill destination must be a direct child of the skills directory")
    if (loader.builtin_skills / destination.name / "SKILL.md").is_file():
        raise ValueError(f"Built-in skill '{name}' cannot be overwritten")

    skills_dir.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError:
        raise FileExistsError(name) from None
    skill_file = destination / "SKILL.md"
    skill.path = str(skill_file)
    try:
        skill_file.write_bytes(data)
    except OSError:
        skill_file.unlink(missing_ok=True)
        destination.rmdir()
        raise
    return skill


@router.get("", response_model=list[SkillInfo])
async def list_skills(
    user_id: Annotated[
        str | None,
        Query(max_length=MAX_USER_ID_LENGTH, description="用户 ID；为空时仅返回内置技能"),
    ] = None,
) -> list[SkillInfo]:
    """返回内置技能，以及指定用户可用的 SKILL 列表。"""
    workspace = _user_workspace(user_id) if user_id else None
    return await SkillsLoader(workspace).list_skill_info()


@router.post("/import", response_model=SkillInfo, status_code=status.HTTP_201_CREATED)
def import_skill(
    user_id: Annotated[str, Query(min_length=1, max_length=MAX_USER_ID_LENGTH, description="用户 ID")],
    file: Annotated[UploadFile, File(description="SKILL.md 文件")],
) -> SkillInfo:
    """将用户上传的 SKILL.md 导入指定用户工作区。"""
    workspace = _user_workspace(user_id)
    if file.filename != "SKILL.md":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded file must be named SKILL.md",
        )

    data = file.file.read(MAX_SKILL_FILE_SIZE + 1)
    if len(data) > MAX_SKILL_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="SKILL.md exceeds the 25 MiB upload limit",
        )
    try:
        return _import_skill_file(data, workspace)
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
