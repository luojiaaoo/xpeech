import uuid
import hashlib
import time

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import event, inspect
from sqlalchemy.orm import registry
from sqlmodel import Field, Session, SQLModel, create_engine


@pytest_asyncio.fixture(autouse=True)
async def isolated_token_manager(tmp_path, monkeypatch):
    import asyncio

    import xpeech.subapp.lark_token_manager.app as manager

    store = manager.TokenStore(tmp_path / "manager.db")
    monkeypatch.setattr(manager, "store", store)
    monkeypatch.setattr(manager, "_poll_tasks", {})
    try:
        yield manager
    finally:
        tasks = list(manager._poll_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        store.close()


@pytest.fixture
def legacy_token_database(tmp_path):
    path = tmp_path / "tokens.db"
    legacy_registry = registry()

    class LegacyModel(SQLModel, registry=legacy_registry):
        pass

    class LegacySession(LegacyModel, table=True):
        __tablename__ = "sessions"

        session_id: str = Field(primary_key=True)
        user_id: str | None = None
        user_name: str | None = None
        access_token: str | None = None
        refresh_token: str | None = None
        scope: str = ""
        access_expires_at: int = 0
        refresh_expires_at: int = 0
        status: str = "unauthorized"
        updated_at: int = 0

    class LegacyAuthorization(LegacyModel, table=True):
        __tablename__ = "device_authorizations"

        session_id: str = Field(primary_key=True)
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

    engine = create_engine(f"sqlite:///{path.as_posix()}")
    try:
        LegacySession.metadata.create_all(engine)
        with Session(engine) as database:
            database.add(LegacySession(
                session_id="employee-1", user_id="ou_123", access_token="access", refresh_token="refresh",
                scope="offline_access", access_expires_at=123, refresh_expires_at=456, status="authorized",
            ))
            database.add(LegacyAuthorization(
                session_id="employee-1", device_code="device", verification_uri="https://example.com/device",
                scope="offline_access", expires_at=9999999999, updated_at=1,
            ))
            database.commit()
    finally:
        engine.dispose()
    return path, LegacySession, LegacyAuthorization


def test_token_store_migrates_local_and_feishu_identities(legacy_token_database):
    from xpeech.subapp.lark_token_manager.dao import TokenStore

    path, _, _ = legacy_token_database

    for _ in range(2):
        migrated = TokenStore(path)
        try:
            with migrated.engine.connect() as connection:
                assert connection.exec_driver_sql("PRAGMA journal_mode").scalar_one() == "wal"
                assert connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one() == 30000
                assert connection.exec_driver_sql("PRAGMA secure_delete").scalar_one() == 1
            row = migrated.row("employee-1")
            assert row.user_id == "employee-1"
            assert row.open_id == "ou_123"
            assert row.access_token == "access"
            assert row.refresh_token == "refresh"
            assert row.scope == "offline_access"
            assert row.access_expires_at == 123
            assert row.refresh_expires_at == 456
            assert row.status == "authorized"
            pending = migrated.authorization("employee-1")
            assert pending.device_code == "device"
            assert pending.scope == "offline_access"
            assert pending.expires_at == 9999999999
        finally:
            migrated.close()


def test_token_store_rolls_back_failed_legacy_migration(legacy_token_database):
    from xpeech.subapp.lark_token_manager.dao import TokenStore

    path, legacy_session, legacy_authorization = legacy_token_database

    def fail_copy(session, *_args):
        if session.new:
            raise RuntimeError("simulated migration failure")

    event.listen(Session, "before_flush", fail_copy)
    try:
        with pytest.raises(RuntimeError, match="simulated migration failure"):
            TokenStore(path)
    finally:
        event.remove(Session, "before_flush", fail_copy)

    engine = create_engine(f"sqlite:///{path.as_posix()}")
    try:
        assert "session_id" in {column["name"] for column in inspect(engine).get_columns("sessions")}
        with Session(engine) as session:
            assert session.get(legacy_session, "employee-1").access_token == "access"
            assert session.get(legacy_authorization, "employee-1").device_code == "device"
    finally:
        engine.dispose()

    store = TokenStore(path)
    try:
        assert store.row("employee-1").open_id == "ou_123"
    finally:
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("open_id", ["ou_123", None])
async def test_authorization_status_and_logout_use_open_id(tmp_path, monkeypatch, open_id):
    import xpeech.subapp.lark_token_manager.app as manager

    store = manager.TokenStore(tmp_path / "tokens.db")
    monkeypatch.setattr(manager, "store", store)
    try:
        store.ensure("employee-1")
        manager._save_authorized(
            "employee-1",
            "offline_access",
            {"access_token": "access", "refresh_token": "refresh"},
            {"open_id": open_id, "union_id": "on_other", "user_id": "other", "name": "Alice"},
        )
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://manager") as client:
            response = await client.get("/v1/status", params={"user_id": "employee-1"})
            assert response.status_code == 200
            assert response.json()["open_id"] == open_id
            assert store.row("employee-1").user_id == "employee-1"

            response = await client.post("/v1/logout", json={"user_id": "employee-1"})
            assert response.status_code == 200
            assert store.row("employee-1").open_id is None
            assert store.row("employee-1").access_token is None
    finally:
        store.close()


@pytest.mark.asyncio
async def test_token_manager_uses_device_flow_and_returns_nonblocking_url(monkeypatch):
    import xpeech.subapp.lark_token_manager.app as manager
    requested_scopes = []

    async def request_device_code(scopes):
        requested_scopes.append(scopes)
        return {
            "device_code": "device-" + uuid.uuid4().hex,
            "user_code": "ABCD-EFGH",
            "verification_uri": "https://feishu.cn/device",
            "verification_uri_complete": "https://feishu.cn/device?code=ABCD-EFGH",
            "expires_in": 120,
            "interval": 30,
        }

    monkeypatch.setattr(manager, "_request_device_code", request_device_code)
    monkeypatch.setattr(manager, "_poll_device_token", lambda code: manager.asyncio.sleep(999))

    session = "test-" + uuid.uuid4().hex
    transport = httpx.ASGITransport(app=manager.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://manager") as client:
        status = await client.get("/v1/status", params={"user_id": session})
        assert status.status_code == 200
        assert status.json()["status"] == "unauthorized"

        started = await client.post(
            "/v1/authorization/start",
            json={"user_id": session, "scopes": ["docs:doc:readonly"]},
        )
        assert started.status_code == 200
        body = started.json()
        assert body["status"] == "pending"
        assert body["authorization_url"].startswith("https://feishu.cn/device")
        assert "access_token" not in body
        assert "redirect_uri" not in body

        repeated = await client.post(
            "/v1/authorization/start",
            json={"user_id": session, "scopes": ["docs:doc:readonly"]},
        )
        assert repeated.json()["authorization_url"] == body["authorization_url"]
        assert requested_scopes[-1] == ["docs:doc:readonly", "offline_access"]

        next_started = await client.post(
            "/v1/authorization/start",
            json={"user_id": session, "scopes": ["drive:drive:readonly"]},
        )
        assert next_started.status_code == 200
        assert requested_scopes[-1] == ["drive:drive:readonly", "offline_access"]


def _insert_authorized_session(manager, user_id: str, *, access_token: str, refresh_token: str, access_expires_at: int, refresh_expires_at: int) -> None:
    manager.store.ensure(user_id)
    manager.store.update_session(user_id, {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "scope": "offline_access",
        "access_expires_at": access_expires_at,
        "refresh_expires_at": refresh_expires_at,
        "status": "authorized",
    })


@pytest.mark.asyncio
async def test_refresh_marks_expired_refresh_token_without_calling_feishu(monkeypatch):
    import xpeech.subapp.lark_token_manager.app as manager

    session = "expired-refresh-" + uuid.uuid4().hex
    _insert_authorized_session(
        manager,
        session,
        access_token="old-access",
        refresh_token="expired-refresh",
        access_expires_at=int(time.time()) - 1,
        refresh_expires_at=int(time.time()) - 1,
    )

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("expired refresh token must not be sent to Feishu")

    monkeypatch.setattr(manager, "_form_post", fail_if_called)
    assert await manager._refresh(manager.store.row(session)) is False
    row = manager.store.row(session)
    assert row.status == "reauthorization_required"
    assert row.access_token is None
    assert row.refresh_token is None


@pytest.mark.asyncio
async def test_refresh_handles_business_error_code_in_http_200(monkeypatch):
    import xpeech.subapp.lark_token_manager.app as manager

    session = "refresh-error-" + uuid.uuid4().hex
    _insert_authorized_session(
        manager,
        session,
        access_token="old-access",
        refresh_token="refresh-token",
        access_expires_at=int(time.time()) - 1,
        refresh_expires_at=int(time.time()) + 3600,
    )

    async def return_expired_error(*args, **kwargs):
        return {"code": 20037, "msg": "refresh token expired"}

    monkeypatch.setattr(manager, "_form_post", return_expired_error)
    assert await manager._refresh(manager.store.row(session)) is False
    row = manager.store.row(session)
    assert row.status == "reauthorization_required"
    assert row.access_token is None
    assert row.refresh_token is None


@pytest.mark.asyncio
async def test_refresh_rotates_both_tokens_and_uses_returned_ttls(monkeypatch):
    import xpeech.subapp.lark_token_manager.app as manager

    session = "refresh-success-" + uuid.uuid4().hex
    _insert_authorized_session(
        manager,
        session,
        access_token="old-access",
        refresh_token="old-refresh",
        access_expires_at=int(time.time()) - 1,
        refresh_expires_at=int(time.time()) + 3600,
    )

    async def return_rotated_tokens(*args, **kwargs):
        assert args[1]["refresh_token"] == "old-refresh"
        return {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 7200,
            "refresh_token_expires_in": 30 * 86400,
        }

    monkeypatch.setattr(manager, "_form_post", return_rotated_tokens)
    before = int(time.time())
    assert await manager._refresh(manager.store.row(session)) is True
    row = manager.store.row(session)
    assert row.access_token == "new-access"
    assert row.refresh_token == "new-refresh"
    assert before + 7200 <= row.access_expires_at <= int(time.time()) + 7200
    assert before + 30 * 86400 <= row.refresh_expires_at <= int(time.time()) + 30 * 86400


@pytest.mark.asyncio
async def test_token_invalidate_only_invalidates_matching_current_token():
    import xpeech.subapp.lark_token_manager.app as manager

    session = "invalidate-" + uuid.uuid4().hex
    access_token = "access-to-invalidate"
    _insert_authorized_session(
        manager,
        session,
        access_token=access_token,
        refresh_token="refresh-token",
        access_expires_at=int(time.time()) + 3600,
        refresh_expires_at=int(time.time()) + 7200,
    )
    transport = httpx.ASGITransport(app=manager.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://manager") as client:
        wrong = await client.post(
            "/v1/token/invalidate",
            json={"user_id": session, "token_type": "user", "token_hash": hashlib.sha256(b"other").hexdigest()},
        )
        assert wrong.status_code == 200
        assert wrong.json() == {"invalidated": False}
        assert manager.store.row(session).access_token == access_token

        invalidated = await client.post(
            "/v1/token/invalidate",
            json={"user_id": session, "token_type": "user", "token_hash": hashlib.sha256(access_token.encode()).hexdigest()},
        )
        assert invalidated.status_code == 200
        assert invalidated.json() == {"invalidated": True}
        row = manager.store.row(session)
        assert row.access_token is None
        assert row.status == "authorized"


@pytest.mark.asyncio
async def test_refresh_scan_refreshes_only_due_sessions(isolated_token_manager, monkeypatch):
    manager = isolated_token_manager
    now = int(time.time())
    for user_id, access_expiry, refresh_expiry in (
        ("expired", now - 10, now - 1),
        ("due", now + 10, now + 3600),
        ("fresh", now + 3600, now + 7200),
    ):
        _insert_authorized_session(
            manager, user_id, access_token=f"{user_id}-access", refresh_token=f"{user_id}-refresh",
            access_expires_at=access_expiry, refresh_expires_at=refresh_expiry,
        )
    calls = []

    async def rotate(_url, payload):
        calls.append(payload["refresh_token"])
        return {"access_token": "rotated-access", "refresh_token": "rotated-refresh"}

    monkeypatch.setattr(manager, "_form_post", rotate)
    await manager.refresh_due()
    assert calls == ["due-refresh"]
    assert manager.store.row("expired").status == "reauthorization_required"
    assert manager.store.row("due").access_token == "rotated-access"
    assert manager.store.row("fresh").access_token == "fresh-access"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["expired", "denied", "success"])
async def test_device_poll_persists_completion_and_cleans_pending_authorization(
    isolated_token_manager, monkeypatch, outcome,
):
    manager = isolated_token_manager
    store = manager.store
    now = int(time.time())
    store.ensure("employee")
    store.save_authorization(manager.DeviceAuthorization(
        user_id="employee", device_code="device", verification_uri="https://example.com/device",
        scope="offline_access", expires_at=now - 1 if outcome == "expired" else now + 3600, updated_at=now,
    ))

    async def skip_sleep(_seconds):
        pass

    async def poll(_device_code):
        return {"access_token": "access", "refresh_token": "refresh"} if outcome == "success" else {"error": "access_denied"}

    async def user_info(_url, _headers):
        return {"open_id": "ou_employee", "name": "Employee"}

    monkeypatch.setattr(manager.asyncio, "sleep", skip_sleep)
    monkeypatch.setattr(manager, "_poll_device_token", poll)
    monkeypatch.setattr(manager, "_json_get", user_info)
    await manager._device_poll("employee")
    assert store.authorization("employee") is None
    row = store.row("employee")
    if outcome == "success":
        assert row.status == "authorized"
        assert row.open_id == "ou_employee"
        assert row.access_token == "access"
    else:
        assert row.status == "authorization_required"


def test_token_store_reopens_current_schema_and_keeps_metadata_isolated(tmp_path):
    from xpeech.subapp.lark_token_manager.dao import DeviceAuthorization, TokenSession, TokenStore

    path = tmp_path / "current.db"
    store = TokenStore(path)
    store.ensure("employee")
    store.update_session("employee", {"open_id": "ou_employee", "access_token": "existing"})
    store.save_authorization(DeviceAuthorization(
        user_id="employee", device_code="device", verification_uri="https://example.com/device",
        scope="offline_access", expires_at=100, updated_at=1,
    ))
    store.close()

    store = TokenStore(path)
    try:
        assert store.row("employee").access_token == "existing"
        assert store.row("employee").open_id == "ou_employee"
        assert store.authorization("employee").device_code == "device"
        assert store.pending_user_ids(99) == ["employee"]
        assert store.pending_user_ids(100) == []
        assert set(inspect(store.engine).get_table_names()) == {"sessions", "device_authorizations"}
        assert TokenSession.metadata is DeviceAuthorization.metadata
        assert TokenSession.metadata is not SQLModel.metadata
    finally:
        store.close()
