"""User-scoped choices for web chat commands and skills."""

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ....utils.channel_jwt import UserJWTAuth
from ..auth import require_user_jwt
from ..dependencies import authenticated_user_id
from .skills import list_skills

router = APIRouter()


@router.get("/chat_command")
async def chat_command_choices(
    auth: Annotated[UserJWTAuth, Depends(require_user_jwt)],
    user_id: Annotated[str, Depends(authenticated_user_id)],
):
    skills = await list_skills(auth, user_id)
    return JSONResponse({
        "commands": [
            {"name": "help", "description": "查看可用指令"},
            {"name": "new", "description": "总结记忆并清空上下文，可在指令后继续提问"},
            {"name": "clear", "description": "清空上下文，不进行记忆总结"},
        ],
        "skills": [
            {"name": skill.metadata.name, "description": skill.metadata.description, "path": skill.path}
            for skill in skills if not skill.metadata.internal
        ],
    }, headers={"Cache-Control": "no-store"})
