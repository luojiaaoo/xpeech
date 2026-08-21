from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Column, ForeignKey, Index, Integer, String, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import registry
from sqlmodel import Field, SQLModel, delete, select
from sqlmodel.ext.asyncio.session import AsyncSession


_web_client_registry = registry()


def _now() -> datetime:
    return datetime.now(UTC)


class _WebClientSQLModel(SQLModel, registry=_web_client_registry):
    """为 Web 客户端认证数据提供独立的 metadata。"""


class User(_WebClientSQLModel, table=True):
    """Web 客户端用户。"""

    __tablename__ = "users"

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(sa_column=Column(String, nullable=False, unique=True))
    username: str = Field(sa_column=Column(String(collation="NOCASE"), nullable=False))
    password_hash: str
    is_admin: bool = Field(default=False)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=_now)


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
    expires_at: datetime
    created_at: datetime = Field(default_factory=_now)


class DuplicateSessionIdError(Exception):
    """会话 ID 违反唯一约束。"""


class ProtectedAdminIdentityError(Exception):
    """默认管理员的用户名和会话 ID 不允许修改。"""


def create_web_client_engine(database_path: Path) -> AsyncEngine:
    """为指定 Web 客户端数据库创建异步 SQLite engine。"""
    resolved_path = database_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{resolved_path.as_posix()}",
        echo=False,
        connect_args={"timeout": 30},
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    return engine


async def create_db_and_tables(engine: AsyncEngine) -> None:
    """创建 Web 客户端认证表。"""
    async with engine.begin() as connection:
        await connection.run_sync(_WebClientSQLModel.metadata.create_all)

    # TODO(legacy-user-migration): BEGIN
    # 旧用户迁移完成后，删除本标记包围的全部代码。
    async with engine.connect() as connection:
        await connection.exec_driver_sql("PRAGMA foreign_keys = OFF")
        await connection.commit()
        try:
            async with connection.begin():
                columns = await connection.exec_driver_sql("PRAGMA table_info(users)")
                column_names = [row[1] for row in columns]
                if "session_id" not in column_names:
                    await connection.exec_driver_sql(
                        "ALTER TABLE users ADD COLUMN session_id VARCHAR"
                    )
                    await connection.exec_driver_sql(
                        "UPDATE users SET session_id = 'web_' || username"
                    )
                    column_names.append("session_id")

                await connection.exec_driver_sql(
                    "UPDATE users SET session_id = 'admin' "
                    "WHERE username = 'admin' COLLATE NOCASE AND session_id = 'web_admin'"
                )

                username_is_unique = False
                indexes = await connection.exec_driver_sql("PRAGMA index_list(users)")
                for index in indexes:
                    if not index[2]:
                        continue
                    index_name = str(index[1]).replace('"', '""')
                    index_columns = await connection.exec_driver_sql(
                        f'PRAGMA index_info("{index_name}")'
                    )
                    if [row[2] for row in index_columns] == ["username"]:
                        username_is_unique = True
                        break

                session_after_username = column_names.index(
                    "session_id"
                ) > column_names.index("username")
                if username_is_unique or session_after_username:
                    await connection.exec_driver_sql(
                        """
                        CREATE TABLE users_new (
                            id INTEGER NOT NULL PRIMARY KEY,
                            session_id VARCHAR NOT NULL UNIQUE,
                            username VARCHAR COLLATE NOCASE NOT NULL,
                            password_hash VARCHAR NOT NULL,
                            is_admin BOOLEAN NOT NULL,
                            is_active BOOLEAN NOT NULL,
                            created_at DATETIME NOT NULL
                        )
                        """
                    )
                    await connection.exec_driver_sql(
                        """
                        INSERT INTO users_new (
                            id, session_id, username, password_hash,
                            is_admin, is_active, created_at
                        )
                        SELECT
                            id, session_id, username, password_hash,
                            is_admin, is_active, created_at
                        FROM users
                        """
                    )
                    await connection.exec_driver_sql("DROP TABLE users")
                    await connection.exec_driver_sql(
                        "ALTER TABLE users_new RENAME TO users"
                    )
        finally:
            await connection.exec_driver_sql("PRAGMA foreign_keys = ON")
            foreign_keys_enabled = (
                await connection.exec_driver_sql("PRAGMA foreign_keys")
            ).scalar_one()
            violations = (
                await connection.exec_driver_sql("PRAGMA foreign_key_check")
            ).all()
            await connection.commit()
            if foreign_keys_enabled != 1:
                raise RuntimeError("Failed to re-enable Web client database foreign keys")
            if violations:
                raise RuntimeError(f"Web client database foreign key violations: {violations}")
    # TODO(legacy-user-migration): END


class WebClientDAO:
    """通过 SQLModel 读写 Web 客户端用户和登录会话。"""

    def __init__(
        self,
        database_path: Path | None = None,
        *,
        engine: AsyncEngine | None = None,
    ) -> None:
        if engine is None and database_path is None:
            raise ValueError("database_path 和 engine 必须提供一个")
        if engine is not None:
            self._engine = engine
        else:
            assert database_path is not None
            self._engine = create_web_client_engine(database_path)

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
                        session_id="admin",
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

    async def get_user_by_session_id(self, session_id: str) -> User | None:
        """按唯一会话 ID 读取用户。"""
        async with AsyncSession(self._engine) as session:
            return (
                await session.exec(select(User).where(User.session_id == session_id))
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

    async def list_users(self) -> list[User]:
        async with AsyncSession(self._engine) as session:
            return list((await session.exec(select(User).order_by(User.id))).all())

    async def create_user(
        self,
        *,
        username: str,
        session_id: str,
        password_hash: str,
        is_admin: bool,
    ) -> User:
        user = User(
            username=username,
            session_id=session_id,
            password_hash=password_hash,
            is_admin=is_admin,
        )
        async with AsyncSession(self._engine) as session:
            session.add(user)
            try:
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                if "session_id" in str(error.orig):
                    raise DuplicateSessionIdError(session_id) from error
                raise
            await session.refresh(user)
        return user

    async def update_user(
        self,
        user_id: int,
        *,
        username: str | None = None,
        session_id: str | None = None,
        password_hash: str | None = None,
        is_admin: bool | None = None,
        is_active: bool | None = None,
    ) -> User | None:
        async with AsyncSession(self._engine) as session:
            user = await session.get(User, user_id)
            if user is None:
                return None
            if user.session_id == "admin" and (
                (username is not None and username != user.username)
                or (session_id is not None and session_id != "admin")
            ):
                raise ProtectedAdminIdentityError
            if username is not None:
                user.username = username
            if session_id is not None:
                user.session_id = session_id
            if password_hash is not None:
                user.password_hash = password_hash
            if is_admin is not None:
                user.is_admin = is_admin
            if is_active is not None:
                user.is_active = is_active
            if is_active is False:
                await session.exec(
                    delete(AuthenticationSession).where(
                        AuthenticationSession.user_id == user_id
                    )
                )
            session.add(user)
            try:
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                if "session_id" in str(error.orig):
                    raise DuplicateSessionIdError(session_id or "") from error
                raise
            await session.refresh(user)
            return user
