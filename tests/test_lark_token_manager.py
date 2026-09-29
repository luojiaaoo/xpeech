import uuid
import hashlib
import time

import httpx
import pytest


@pytest.mark.asyncio
async def test_token_manager_uses_device_flow_and_returns_nonblocking_url(monkeypatch):
    import xpeech.lark_token_manager as manager
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
        status = await client.get("/v1/status", params={"session_id": session})
        assert status.status_code == 200
        assert status.json()["status"] == "unauthorized"

        started = await client.post(
            "/v1/authorization/start",
            json={"session_id": session, "scopes": ["docs:doc:readonly"]},
        )
        assert started.status_code == 200
        body = started.json()
        assert body["status"] == "pending"
        assert body["authorization_url"].startswith("https://feishu.cn/device")
        assert "access_token" not in body
        assert "redirect_uri" not in body

        repeated = await client.post(
            "/v1/authorization/start",
            json={"session_id": session, "scopes": ["docs:doc:readonly"]},
        )
        assert repeated.json()["authorization_url"] == body["authorization_url"]
        assert requested_scopes[-1] == ["docs:doc:readonly", "offline_access"]

        next_started = await client.post(
            "/v1/authorization/start",
            json={"session_id": session, "scopes": ["drive:drive:readonly"]},
        )
        assert next_started.status_code == 200
        assert requested_scopes[-1] == ["drive:drive:readonly", "offline_access"]


def _insert_authorized_session(manager, session_id: str, *, access_token: str, refresh_token: str, access_expires_at: int, refresh_expires_at: int) -> None:
    manager.store.ensure(session_id)
    manager.store.db.execute(
        "UPDATE sessions SET access_token=?, refresh_token=?, scope=?, access_expires_at=?, refresh_expires_at=?, status='authorized' WHERE session_id=?",
        (access_token, refresh_token, "offline_access", access_expires_at, refresh_expires_at, session_id),
    )
    manager.store.db.commit()


@pytest.mark.asyncio
async def test_refresh_marks_expired_refresh_token_without_calling_feishu(monkeypatch):
    import xpeech.lark_token_manager as manager

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
    import xpeech.lark_token_manager as manager

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
    import xpeech.lark_token_manager as manager

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
    import xpeech.lark_token_manager as manager

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
            json={"session_id": session, "token_type": "user", "token_hash": hashlib.sha256(b"other").hexdigest()},
        )
        assert wrong.status_code == 200
        assert wrong.json() == {"invalidated": False}
        assert manager.store.row(session)["access_token"] == access_token

        invalidated = await client.post(
            "/v1/token/invalidate",
            json={"session_id": session, "token_type": "user", "token_hash": hashlib.sha256(access_token.encode()).hexdigest()},
        )
        assert invalidated.status_code == 200
        assert invalidated.json() == {"invalidated": True}
        row = manager.store.row(session)
        assert row["access_token"] is None
        assert row["status"] == "authorized"
