from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ..dao import (
    DuplicateSessionIdError,
    ProtectedAdminDeletionError,
    ProtectedAdminIdentityError,
    User,
    WebClientDAO,
)
from ..models import UserBody, UserUpdateBody, public_user


AdminUserDependency = Callable[..., Awaitable[User] | User]
PasswordHash = Callable[[str], str]


def create_admin_router(
    dao: WebClientDAO,
    admin_user_dependency: AdminUserDependency,
    password_hash: PasswordHash,
) -> APIRouter:
    router = APIRouter(prefix="/api/admin")
    AdminUser = Annotated[User, Depends(admin_user_dependency)]

    @router.get("/users")
    async def list_users(_admin: AdminUser):
        return [public_user(user) for user in await dao.list_users()]

    @router.post("/users", status_code=201)
    async def create_user(body: UserBody, _admin: AdminUser):
        try:
            user = await dao.create_user(
                username=body.username,
                session_id=body.session_id,
                password_hash=password_hash(body.password),
                is_admin=body.is_admin,
            )
        except DuplicateSessionIdError:
            raise HTTPException(status.HTTP_409_CONFLICT, "会话 ID 已存在")
        return public_user(user)

    @router.patch("/users/{user_id}")
    async def update_user(
        user_id: int,
        body: UserUpdateBody,
        admin: AdminUser,
    ):
        values = body.model_dump(exclude_none=True)
        if user_id == admin.id and values.get("is_active") is False:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "不能停用当前管理员")
        if not values:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "没有可更新字段")
        try:
            user = await dao.update_user(
                user_id,
                username=values.get("username"),
                session_id=values.get("session_id"),
                password_hash=(
                    password_hash(values["password"])
                    if "password" in values
                    else None
                ),
                is_admin=values.get("is_admin"),
                is_active=values.get("is_active"),
            )
        except DuplicateSessionIdError:
            raise HTTPException(status.HTTP_409_CONFLICT, "会话 ID 已存在")
        except ProtectedAdminIdentityError:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "默认管理员的用户名和会话 ID 不可修改",
            )
        if user is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "用户不存在")
        return public_user(user)

    @router.delete("/users/{user_id}", status_code=204)
    async def delete_user(user_id: int, admin: AdminUser):
        if user_id == admin.id:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "不能删除当前管理员")
        try:
            deleted = await dao.delete_user(user_id)
        except ProtectedAdminDeletionError:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "默认管理员不可删除",
            )
        if not deleted:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "用户不存在")

    return router
