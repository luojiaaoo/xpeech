import uuid
import hashlib
import time

import httpx
import pytest


def test_token_store_migrates_local_and_feishu_identities(tmp_path):
    from xpeech.subapp.lark_token_manager.app import TokenStore

    path = tmp_path / "tokens.db"
    store = TokenStore(path)
    store.ensure("employee-1")
    store.db.execute(
        "UPDATE sessions SET open_id='ou_123', access_token='access', refresh_token='refresh', scope='offline_access', access_expires_at=123, refresh_expires_at=456, status='authorized' WHERE user_id='employee-1'"
    )
    store.db.execute(
        "INSERT INTO device_authorizations(user_id, device_code, verification_uri, scope, expires_at, updated_at) VALUES ('employee-1', 'device', 'https://example.com/device', 'offline_access', 9999999999, 1)"
    )
    store.db.execute("ALTER TABLE sessions RENAME COLUMN user_id TO session_id")
    store.db.execute("ALTER TABLE sessions RENAME COLUMN open_id TO user_id")
    store.db.execute("ALTER TABLE device_authorizations RENAME COLUMN user_id TO session_id")
    store.db.commit()
    store.db.close()

    for _ in range(2):
        migrated = TokenStore(path)
        try:
            row = migrated.row("employee-1")
            assert row["user_id"] == "employee-1"
            assert row["open_id"] == "ou_123"
            assert row["access_token"] == "access"
            assert row["refresh_token"] == "refresh"
            assert row["scope"] == "offline_access"
            assert row["access_expires_at"] == 123
            assert row["refresh_expires_at"] == 456
            assert row["status"] == "authorized"
            pending = migrated.db.execute("SELECT * FROM device_authorizations WHERE user_id='employee-1'").fetchone()
            assert pending["device_code"] == "device"
            assert pending["scope"] == "offline_access"
            assert pending["expires_at"] == 9999999999
        finally:
            migrated.db.close()


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
            assert store.row("employee-1")["user_id"] == "employee-1"

            response = await client.post("/v1/logout", json={"user_id": "employee-1"})
            assert response.status_code == 200
            assert store.row("employee-1")["open_id"] is None
            assert store.row("employee-1")["access_token"] is None
    finally:
        store.db.close()


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
    manager.store.db.execute(
        "UPDATE sessions SET access_token=?, refresh_token=?, scope=?, access_expires_at=?, refresh_expires_at=?, status='authorized' WHERE user_id=?",
        (access_token, refresh_token, "offline_access", access_expires_at, refresh_expires_at, user_id),
    )
    manager.store.db.commit()


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
    assert row["status"] == "reauthorization_required"
    assert row["access_token"] is None
    assert row["refresh_token"] is None


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
    assert row["status"] == "reauthorization_required"
    assert row["access_token"] is None
    assert row["refresh_token"] is None


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
    assert row["access_token"] == "new-access"
    assert row["refresh_token"] == "new-refresh"
    assert before + 7200 <= row["access_expires_at"] <= int(time.time()) + 7200
    assert before + 30 * 86400 <= row["refresh_expires_at"] <= int(time.time()) + 30 * 86400


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
        assert manager.store.row(session)["access_token"] == access_token

        invalidated = await client.post(
            "/v1/token/invalidate",
            json={"user_id": session, "token_type": "user", "token_hash": hashlib.sha256(access_token.encode()).hexdigest()},
        )
        assert invalidated.status_code == 200
        assert invalidated.json() == {"invalidated": True}
        row = manager.store.row(session)
        assert row["access_token"] is None
        assert row["status"] == "authorized"
