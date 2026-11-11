from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import registry
from sqlmodel import Field, SQLModel, delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from ....utils.database import create_tables_async

_web_client_registry = registry()


def _now() -> datetime:
    return datetime.now(UTC)


class _WebClientSQLModel(SQLModel, registry=_web_client_registry):
    """为 Web 客户端认证数据提供独立的 metadata。"""


class User(_WebClientSQLModel, table=True):
    """Web 客户端用户。"""

    __tablename__ = "users"

    id: int | None = Field(default=None, primary_key=True)
    user_id: str = Field(sa_column=Column(String, nullable=False, unique=True))
    username: str = Field(sa_column=Column(String, nullable=False))
    password_hash: str
    is_admin: bool = Field(default=False)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=_now, sa_type=DateTime(timezone=True))


class AuthenticationSession(_WebClientSQLModel, table=True):
    """Web 客户端登录会话。"""

    __tablename__ = "sessions"
    __table_args__ = (Index("idx_sessions_user_id", "user_id"),)

    token_hash: str = Field(primary_key=True)
    user_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    expires_at: datetime = Field(sa_type=DateTime(timezone=True))
    created_at: datetime = Field(default_factory=_now, sa_type=DateTime(timezone=True))


class DuplicateUserIdError(Exception):
    """用户 ID 违反唯一约束。"""


class ProtectedAdminIdentityError(Exception):
    """默认管理员的用户名和用户 ID 不允许修改。"""


class ProtectedAdminDeletionError(Exception):
    """默认管理员不允许删除。"""


async def create_db_and_tables(engine: AsyncEngine) -> None:
    """创建 Web 客户端认证表。"""
    await create_tables_async(engine, _WebClientSQLModel.metadata)


class WebClientDAO:
    """通过 SQLModel 读写 Web 客户端用户和登录会话。"""

    def __init__(
        self,
        engine: AsyncEngine,
    ) -> None:
        self._engine = engine

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    async def initialize(
        self,
        default_admin_password_hash_factory: Callable[[], str],
    ) -> None:
        """初始化表、清理过期会话，并在空库中创建默认管理员。"""
        await create_db_and_tables(self._engine)
        async with AsyncSession(self._engine) as session:
            await session.exec(
                delete(AuthenticationSession).where(
                    AuthenticationSession.expires_at <= _now()
                )
            )
            first_user_id = (await session.exec(select(User.id).limit(1))).first()
            if first_user_id is None:
                session.add(
                    User(
                        username="admin",
                        user_id="admin",
                        password_hash=default_admin_password_hash_factory(),
                        is_admin=True,
                    )
                )
            await session.commit()

    async def close(self) -> None:
        """释放数据库连接池。"""
        await self._engine.dispose()

    async def get_user_for_session(
        self,
        token_hash: str,
        *,
        now: datetime | None = None,
    ) -> User | None:
        """读取有效会话对应的启用用户。"""
        statement = (
            select(User)
            .join(
                AuthenticationSession,
                User.id == AuthenticationSession.user_id,
            )
            .where(
                AuthenticationSession.token_hash == token_hash,
                AuthenticationSession.expires_at > (now or _now()),
                User.is_active.is_(True),
            )
        )
        async with AsyncSession(self._engine) as session:
            return (await session.exec(statement)).one_or_none()

    async def get_user_by_user_id(self, user_id: str) -> User | None:
        """按唯一用户 ID 读取用户。"""
        async with AsyncSession(self._engine) as session:
            return (
                await session.exec(select(User).where(User.user_id == user_id))
            ).one_or_none()

    async def create_session(
        self,
        *,
        token_hash: str,
        user_id: int,
        expires_at: datetime,
    ) -> None:
        async with AsyncSession(self._engine) as session:
            session.add(
                AuthenticationSession(
                    token_hash=token_hash,
                    user_id=user_id,
                    expires_at=expires_at,
                )
            )
            await session.commit()

    async def delete_session(self, token_hash: str) -> None:
        async with AsyncSession(self._engine) as session:
            await session.exec(
                delete(AuthenticationSession).where(
                    AuthenticationSession.token_hash == token_hash
                )
            )
            await session.commit()

    async def change_password(
        self,
        user_id: int,
        *,
        password_hash: str,
        keep_token_hash: str,
    ) -> bool:
        """修改用户密码，并使当前登录之外的会话失效。"""
        async with AsyncSession(self._engine) as session:
            user = await session.get(User, user_id)
            if user is None:
                return False
            user.password_hash = password_hash
            session.add(user)
            await session.exec(
                delete(AuthenticationSession).where(
                    AuthenticationSession.user_id == user_id,
                    AuthenticationSession.token_hash != keep_token_hash,
                )
            )
            await session.commit()
            return True

    async def list_users(self) -> list[User]:
        async with AsyncSession(self._engine) as session:
            return list((await session.exec(select(User).order_by(User.id))).all())

    async def create_user(
        self,
        *,
        username: str,
        user_id: str,
        password_hash: str,
        is_admin: bool,
    ) -> User:
        user = User(
            username=username,
            user_id=user_id,
            password_hash=password_hash,
            is_admin=is_admin,
        )
        async with AsyncSession(self._engine) as session:
            session.add(user)
            try:
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                if "user_id" in str(error.orig):
                    raise DuplicateUserIdError(user_id) from error
                raise
            await session.refresh(user)
        return user

    async def update_user(
        self,
        user_pk: int,
        *,
        username: str | None = None,
        user_id: str | None = None,
        password_hash: str | None = None,
        is_admin: bool | None = None,
        is_active: bool | None = None,
    ) -> User | None:
        async with AsyncSession(self._engine) as session:
            user = await session.get(User, user_pk)
            if user is None:
                return None
            if user.user_id == "admin" and (
                (username is not None and username != user.username)
                or (user_id is not None and user_id != "admin")
            ):
                raise ProtectedAdminIdentityError
            if username is not None:
                user.username = username
            if user_id is not None:
                user.user_id = user_id
            if password_hash is not None:
                user.password_hash = password_hash
            if is_admin is not None:
                user.is_admin = is_admin
            if is_active is not None:
                user.is_active = is_active
            if is_active is False:
                await session.exec(
                    delete(AuthenticationSession).where(
                        AuthenticationSession.user_id == user_pk
                    )
                )
            session.add(user)
            try:
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                if "user_id" in str(error.orig):
                    raise DuplicateUserIdError(user_id or "") from error
                raise
            await session.refresh(user)
            return user

    async def delete_user(self, user_id: int) -> bool:
        """删除用户；关联的登录会话由外键级联删除。"""
        async with AsyncSession(self._engine) as session:
            user = await session.get(User, user_id)
            if user is None:
                return False
            if user.user_id == "admin":
                raise ProtectedAdminDeletionError
            await session.delete(user)
            await session.commit()
            return True
