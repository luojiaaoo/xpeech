"""SQLModel storage for Feishu credentials and pending device authorizations."""

import asyncio
import time
from functools import partial
from pathlib import Path
from typing import Any

from sqlalchemy import MetaData, Table, event, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import registry
from sqlmodel import Field, Session, SQLModel, create_engine, delete, false, select

from ...utils.sqlite import configure_sqlite_connection

_token_registry = registry()


class _TokenSQLModel(SQLModel, registry=_token_registry):
    """Keep token tables separate from web authentication and scheduler tables."""


class TokenSession(_TokenSQLModel, table=True):
    __tablename__ = "sessions"

    user_id: str = Field(primary_key=True)
    open_id: str | None = None
    user_name: str | None = None
    access_token: str | None = None
    refresh_token: str | None = None
    scope: str = ""
    access_expires_at: int = 0
    refresh_expires_at: int = 0
    status: str = "unauthorized"
    updated_at: int = 0


class DeviceAuthorization(_TokenSQLModel, table=True):
    __tablename__ = "device_authorizations"

    user_id: str = Field(primary_key=True)
    device_code: str
    user_code: str = ""
    verification_uri: str
    verification_uri_complete: str = ""
    scope: str
    interval_seconds: int = 5
    expires_at: int
    status: str = "pending"
    last_error: str = ""
    updated_at: int


class TokenStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.parent.chmod(0o700)
        self.engine = create_engine(f"sqlite:///{self.path.as_posix()}")
        event.listen(
            self.engine,
            "connect",
            partial(configure_sqlite_connection, secure_delete=True),
        )
        try:
            self._migrate_legacy_identities()
            _TokenSQLModel.metadata.create_all(self.engine)
            self.path.chmod(0o600)
        except Exception:
            self.engine.dispose()
            raise
        self.locks: dict[str, asyncio.Lock] = {}

    def _migrate_legacy_identities(self) -> None:
        """Rebuild legacy tables with model metadata while retaining all credentials."""
        with Session(self.engine) as session:
            connection = session.connection()
            inspector = inspect(connection)
            for model in (TokenSession, DeviceAuthorization):
                name = model.__tablename__
                if not inspector.has_table(name):
                    continue
                columns = {column["name"] for column in inspector.get_columns(name)}
                if "session_id" not in columns:
                    continue
                legacy = Table(name, MetaData(), autoload_with=connection)
                # Start SQLite's write transaction before transactional DDL so
                # a failed copy rolls back the original table and its contents.
                session.exec(delete(legacy).where(false()))
                rows = session.exec(select(*legacy.c)).mappings().all()
                migrated = []
                for row in rows:
                    values = dict(row)
                    if model is TokenSession and "user_id" in columns:
                        values["open_id"] = values.pop("user_id")
                    values["user_id"] = values.pop("session_id")
                    migrated.append(model.model_validate(values))
                legacy.drop(connection)
                model.__table__.create(connection)
                session.add_all(migrated)
                session.flush()
            session.commit()

    def close(self) -> None:
        self.engine.dispose()

    def lock(self, user_id: str) -> asyncio.Lock:
        return self.locks.setdefault(user_id, asyncio.Lock())

    def row(self, user_id: str) -> TokenSession | None:
        with Session(self.engine) as session:
            return session.get(TokenSession, user_id)

    def ensure(self, user_id: str) -> TokenSession:
        with Session(self.engine, expire_on_commit=False) as session:
            row = session.get(TokenSession, user_id)
            if row is not None:
                return row
            row = TokenSession(user_id=user_id, updated_at=int(time.time()))
            session.add(row)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                existing = session.get(TokenSession, user_id)
                if existing is None:
                    raise
                return existing
            return row

    def update_session(
        self,
        user_id: str,
        values: dict[str, Any],
        *,
        clear_authorization: bool = False,
        create: bool = False,
    ) -> None:
        with Session(self.engine) as session:
            row = session.get(TokenSession, user_id)
            if row is None and create:
                row = TokenSession(user_id=user_id)
            if row is not None:
                row.sqlmodel_update(values)
                session.add(row)
            if clear_authorization:
                session.exec(delete(DeviceAuthorization).where(DeviceAuthorization.user_id == user_id))
            session.commit()

    def authorization(
        self,
        user_id: str,
        *,
        pending_only: bool = False,
        valid_at: int | None = None,
    ) -> DeviceAuthorization | None:
        statement = select(DeviceAuthorization).where(DeviceAuthorization.user_id == user_id)
        if pending_only:
            statement = statement.where(DeviceAuthorization.status == "pending")
        if valid_at is not None:
            statement = statement.where(DeviceAuthorization.expires_at > valid_at)
        with Session(self.engine) as session:
            return session.exec(statement).first()

    def save_authorization(self, authorization: DeviceAuthorization) -> None:
        with Session(self.engine) as session:
            session.merge(authorization)
            row = session.get(TokenSession, authorization.user_id)
            if row is not None:
                row.status = "pending"
                row.updated_at = authorization.updated_at
                session.add(row)
            session.commit()

    def update_authorization(self, user_id: str, values: dict[str, Any]) -> None:
        with Session(self.engine) as session:
            row = session.get(DeviceAuthorization, user_id)
            if row is not None:
                row.sqlmodel_update(values)
                session.add(row)
                session.commit()

    def pending_user_ids(self, now: int) -> list[str]:
        with Session(self.engine) as session:
            return list(session.exec(
                select(DeviceAuthorization.user_id).where(DeviceAuthorization.expires_at > now)
            ).all())

    def expired_refresh_sessions(self, now: int) -> list[TokenSession]:
        with Session(self.engine) as session:
            return list(session.exec(
                select(TokenSession).where(
                    TokenSession.refresh_token.is_not(None),
                    TokenSession.refresh_token != "",
                    TokenSession.refresh_expires_at <= now,
                )
            ).all())

    def due_refresh_sessions(self, now: int, due_before: int) -> list[TokenSession]:
        with Session(self.engine) as session:
            return list(session.exec(
                select(TokenSession).where(
                    TokenSession.refresh_token.is_not(None),
                    TokenSession.refresh_token != "",
                    TokenSession.refresh_expires_at > now,
                    TokenSession.access_expires_at <= due_before,
                )
            ).all())

    def scopes(self, user_id: str, requested: list[str]) -> list[str]:
        self.ensure(user_id)
        # Feishu owns previously granted scopes; only request the supplied scopes.
        values = {scope.strip() for scope in requested if scope.strip()}
        values.add("offline_access")
        return sorted(values)
