"""Feishu device-authorisation and token refresh service.

The service owns long-lived credentials. ``lark-cli`` only receives a
short-lived access token, while device-code polling continues in this process.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

from ...config.settings import settings
from .dao import DeviceAuthorization, TokenSession, TokenStore

DEVICE_AUTH_URL = "https://accounts.feishu.cn/oauth/v1/device_authorization"
TOKEN_URL = "https://open.feishu.cn/open-apis/authen/v2/oauth/token"
USERINFO_URL = "https://open.feishu.cn/open-apis/authen/v1/user_info"
TENANT_URL = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
# Feishu user tokens are short-lived; keep refresh behaviour predictable and
# independent of deployment configuration.
DEFAULT_ACCESS_TOKEN_TTL_SECONDS = 2 * 60 * 60
DEFAULT_REFRESH_TOKEN_TTL_SECONDS = 30 * 24 * 60 * 60
REFRESH_SKEW_SECONDS = 300
REFRESH_SCAN_INTERVAL_SECONDS = 60


class StartRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    scopes: list[str] = Field(default_factory=list)


class ResolveRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    token_type: str = "user"
    scopes: list[str] = Field(default_factory=list)


class LogoutRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)


class InvalidateRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    token_hash: str = Field(min_length=64, max_length=64)
    token_type: str = "user"


class MigrateRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    app_id: str
    access_token: str
    refresh_token: str = ""
    scope: str = ""
    expires_at: int = 0
    refresh_token_expires_at: int = 0


store = TokenStore(settings.feishu.lark_cli.database_path)
_poll_tasks: dict[str, asyncio.Task[None]] = {}


async def _form_post(url: str, payload: dict[str, Any], *, headers: dict[str, str] | None = None, allow_error: bool = False) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/x-www-form-urlencoded", **(headers or {})}
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(url, data=payload, headers=request_headers)
    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Feishu returned non-JSON response ({response.status_code})") from exc
    if response.status_code >= 400 and not allow_error:
        raise RuntimeError(str(body))
    return body if isinstance(body, dict) else {}


async def _json_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(url, json=payload)
    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Feishu returned non-JSON response ({response.status_code})") from exc
    if response.status_code >= 400 or (isinstance(body, dict) and body.get("code", 0) not in (0, None)):
        raise RuntimeError(str(body))
    return body if isinstance(body, dict) else {}


async def _json_get(url: str, headers: dict[str, str]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(url, headers=headers)
    body = response.json()
    if response.status_code >= 400 or body.get("code", 0) not in (0, None):
        raise RuntimeError(str(body))
    return body


async def _request_device_code(scopes: list[str]) -> dict[str, Any]:
    basic = base64.b64encode(f"{settings.feishu.app_id}:{settings.feishu.app_secret}".encode()).decode()
    body = await _form_post(DEVICE_AUTH_URL, {"client_id": settings.feishu.app_id, "scope": " ".join(scopes)}, headers={"Authorization": f"Basic {basic}"})
    data = body.get("data", body)
    if not isinstance(data, dict) or not data.get("device_code"):
        raise RuntimeError("device authorization response did not contain device_code")
    return data


async def _poll_device_token(device_code: str) -> dict[str, Any]:
    return await _form_post(TOKEN_URL, {"grant_type": "urn:ietf:params:oauth:grant-type:device_code", "device_code": device_code, "client_id": settings.feishu.app_id, "client_secret": settings.feishu.app_secret}, allow_error=True)


def _token_data(body: dict[str, Any]) -> dict[str, Any]:
    data = body.get("data", body)
    return data if isinstance(data, dict) else {}


def _error_code(body: dict[str, Any]) -> str:
    data = _token_data(body)
    return str(body.get("error") or body.get("code") or data.get("error") or data.get("code") or "")


def _positive_ttl(data: dict[str, Any], keys: tuple[str, ...], default: int) -> int:
    """Read a provider TTL while tolerating absent/null/malformed responses."""
    for key in keys:
        value = data.get(key)
        try:
            ttl = int(value)
        except (TypeError, ValueError):
            continue
        if ttl > 0:
            return ttl
    return default


def _token_expiry(data: dict[str, Any], now: int) -> tuple[int, int]:
    access_ttl = _positive_ttl(data, ("expires_in", "expire_in"), DEFAULT_ACCESS_TOKEN_TTL_SECONDS)
    refresh_ttl = _positive_ttl(
        data,
        ("refresh_token_expires_in", "refresh_expires_in"),
        DEFAULT_REFRESH_TOKEN_TTL_SECONDS,
    )
    return now + access_ttl, now + refresh_ttl


def _mark_reauthorization_required(user_id: str) -> None:
    # Clear both credentials after a terminal refresh failure.  This prevents
    # a background scan from retrying a rotated/expired refresh token forever.
    store.update_session(user_id, {
        "access_token": None,
        "refresh_token": None,
        "access_expires_at": 0,
        "refresh_expires_at": 0,
        "status": "reauthorization_required",
        "updated_at": int(time.time()),
    })


def _save_authorized(user_id: str, scope: str, data: dict[str, Any], user: dict[str, Any]) -> None:
    now = int(time.time())
    granted = " ".join(sorted(set(filter(None, str(data.get("scope") or scope).split()))))
    access_expires_at, refresh_expires_at = _token_expiry(data, now)
    store.update_session(user_id, {
        "open_id": user.get("open_id"),
        "user_name": user.get("name", ""),
        "access_token": data.get("access_token", ""),
        "refresh_token": data.get("refresh_token", ""),
        "scope": granted,
        "access_expires_at": access_expires_at,
        "refresh_expires_at": refresh_expires_at,
        "status": "authorized",
        "updated_at": now,
    }, clear_authorization=True)


def _end_device_authorization(user_id: str) -> None:
    store.update_session(user_id, {
        "status": "authorization_required",
        "updated_at": int(time.time()),
    }, clear_authorization=True)


async def _complete_device(user_id: str, scope: str, body: dict[str, Any]) -> None:
    data = _token_data(body)
    access = data.get("access_token", "")
    if not access:
        raise RuntimeError("device token response did not contain access_token")
    info = await _json_get(USERINFO_URL, {"Authorization": f"Bearer {access}"})
    user = info.get("data", info)
    async with store.lock(user_id):
        store.ensure(user_id)
        _save_authorized(user_id, scope, data, user if isinstance(user, dict) else {})


async def _device_poll(user_id: str) -> None:
    interval = 5
    try:
        while True:
            row = store.authorization(user_id)
            if not row:
                return
            now = int(time.time())
            if row.expires_at <= now:
                _end_device_authorization(user_id)
                return
            interval = max(1, int(row.interval_seconds or interval))
            await asyncio.sleep(min(interval, max(1, row.expires_at - now)))
            try:
                body = await _poll_device_token(row.device_code)
            except (RuntimeError, httpx.HTTPError) as exc:
                store.update_authorization(user_id, {
                    "last_error": str(exc)[:200],
                    "updated_at": int(time.time()),
                })
                interval = min(interval + 5, 60)
                continue
            error = _error_code(body)
            if body.get("access_token") or _token_data(body).get("access_token"):
                try:
                    await _complete_device(user_id, row.scope, body)
                except (RuntimeError, ValueError, KeyError, httpx.HTTPError, SQLAlchemyError):
                    _end_device_authorization(user_id)
                return
            if error == "authorization_pending":
                continue
            if error == "slow_down":
                interval = min(interval + 5, 60)
                continue
            _end_device_authorization(user_id)
            return
    finally:
        _poll_tasks.pop(user_id, None)


def _schedule_poll(user_id: str) -> None:
    task = _poll_tasks.get(user_id)
    if task is None or task.done():
        _poll_tasks[user_id] = asyncio.create_task(_device_poll(user_id))


async def _refresh(row: TokenSession) -> bool:
    now = int(time.time())
    if not row.refresh_token:
        return False
    if row.refresh_expires_at <= now:
        _mark_reauthorization_required(row.user_id)
        return False
    try:
        body = await _form_post(TOKEN_URL, {"grant_type": "refresh_token", "refresh_token": row.refresh_token, "client_id": settings.feishu.app_id, "client_secret": settings.feishu.app_secret})
        error = _error_code(body)
        if error and error not in {"0", "none"}:
            raise RuntimeError(f"refresh failed with Feishu error {error}: {body.get('msg') or body.get('message') or ''}")
        data = _token_data(body)
        access, refresh = data.get("access_token", ""), data.get("refresh_token", "")
        if not access or not refresh:
            raise RuntimeError("refresh response did not rotate both tokens")
        access_expires_at, refresh_expires_at = _token_expiry(data, now)
        store.update_session(row.user_id, {
            "access_token": access,
            "refresh_token": refresh,
            "scope": data.get("scope") or row.scope,
            "access_expires_at": access_expires_at,
            "refresh_expires_at": refresh_expires_at,
            "status": "authorized",
            "updated_at": now,
        })
        return True
    except (RuntimeError, ValueError, KeyError, httpx.HTTPError, SQLAlchemyError) as exc:
        if any(word in str(exc).lower() for word in ("invalid_grant", "invalid refresh", "expired", "revoked", "already used", "20026", "20037", "20064", "20073")):
            _mark_reauthorization_required(row.user_id)
        return False


async def refresh_due() -> None:
    current = int(time.time())
    now = current + REFRESH_SKEW_SECONDS
    # Expired refresh tokens are terminal.  Mark them once and exclude them
    # from the scan so the service does not issue a failed request every minute.
    expired_rows = store.expired_refresh_sessions(current)
    for expired in expired_rows:
        async with store.lock(expired.user_id):
            latest = store.row(expired.user_id)
            if latest and latest.refresh_token and latest.refresh_expires_at <= current:
                _mark_reauthorization_required(expired.user_id)
    rows = store.due_refresh_sessions(current, now)
    for row in rows:
        async with store.lock(row.user_id):
            latest = store.row(row.user_id)
            if latest and latest.refresh_expires_at > current and latest.access_expires_at <= now:
                await _refresh(latest)


async def refresh_loop() -> None:
    while True:
        await refresh_due()
        await asyncio.sleep(REFRESH_SCAN_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    refresh_task = asyncio.create_task(refresh_loop())
    for user_id in store.pending_user_ids(int(time.time())):
        _schedule_poll(user_id)
    try:
        yield
    finally:
        refresh_task.cancel()
        tasks = list(_poll_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(refresh_task, *tasks, return_exceptions=True)
        store.close()


app = FastAPI(title="Xpeech Feishu token manager", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/v1/authorization/start")
async def authorization_start(payload: StartRequest) -> dict[str, Any]:
    async with store.lock(payload.user_id):
        scopes = store.scopes(payload.user_id, payload.scopes)
        now = int(time.time())
        existing = store.authorization(payload.user_id, pending_only=True, valid_at=now)
        if existing and set(existing.scope.split()) == set(scopes):
            _schedule_poll(payload.user_id)
            return {"status": "pending", "authorization_url": existing.verification_uri_complete or existing.verification_uri, "verification_uri": existing.verification_uri, "user_code": existing.user_code, "expires_at": existing.expires_at, "scopes": scopes}
        data = await _request_device_code(scopes)
        expires_in = max(1, int(data.get("expires_in", data.get("expire_in", 240))))
        now = int(time.time())
        # The device authorization endpoint owns the lifetime.  Do not add a
        # second deployment-specific timeout that can disagree with Feishu.
        expires_at = now + expires_in
        interval = max(1, int(data.get("interval", 5)))
        store.save_authorization(DeviceAuthorization(
            user_id=payload.user_id,
            device_code=data["device_code"],
            user_code=data.get("user_code", ""),
            verification_uri=data.get("verification_uri", ""),
            verification_uri_complete=data.get("verification_uri_complete", data.get("verification_uri", "")),
            scope=" ".join(scopes),
            interval_seconds=interval,
            expires_at=expires_at,
            updated_at=now,
        ))
    _schedule_poll(payload.user_id)
    return {"status": "pending", "authorization_url": data.get("verification_uri_complete") or data.get("verification_uri"), "verification_uri": data.get("verification_uri", ""), "user_code": data.get("user_code", ""), "expires_at": expires_at, "scopes": scopes}


@app.post("/v1/token/resolve")
async def token_resolve(payload: ResolveRequest) -> dict[str, Any]:
    async with store.lock(payload.user_id):
        row = store.ensure(payload.user_id)
        now = int(time.time())
        if payload.token_type == "tenant":
            data = _token_data(await _json_post(TENANT_URL, {"app_id": settings.feishu.app_id, "app_secret": settings.feishu.app_secret}))
            return {"access_token": data["tenant_access_token"], "expires_at": now + int(data.get("expire", 7200)), "scope": ""}
        requested = {s.strip() for s in payload.scopes if s.strip()}
        granted = set(filter(None, row.scope.split()))
        # Credential Provider requests do not carry endpoint-specific scopes;
        # a valid user token is therefore usable until the upstream command
        # reports a missing scope and the user explicitly authorizes it.
        required = requested
        if row.access_token and row.access_expires_at > now + 30 and required <= granted:
            return {"access_token": row.access_token, "expires_at": row.access_expires_at, "scope": row.scope}
        if row.refresh_token and row.refresh_expires_at > now and required <= granted and await _refresh(row):
            row = store.row(payload.user_id)
            return {"access_token": row.access_token, "expires_at": row.access_expires_at, "scope": row.scope}
    start = await authorization_start(StartRequest(user_id=payload.user_id, scopes=payload.scopes))
    raise HTTPException(status_code=428, detail={"type": "authorization_required", **start})


@app.post("/v1/token/invalidate")
async def token_invalidate(payload: InvalidateRequest) -> dict[str, bool]:
    """Forget a token that the upstream API rejected before local expiry."""
    if payload.token_type != "user":
        return {"invalidated": False}

    async with store.lock(payload.user_id):
        row = store.row(payload.user_id)
        if not row or not row.access_token:
            return {"invalidated": False}
        current_hash = hashlib.sha256(row.access_token.encode()).hexdigest()
        if not hmac.compare_digest(current_hash, payload.token_hash):
            return {"invalidated": False}

        now = int(time.time())
        has_refresh = bool(row.refresh_token) and row.refresh_expires_at > now
        values = {
            "access_token": None,
            "access_expires_at": 0,
            "status": "authorized" if has_refresh else "reauthorization_required",
            "updated_at": now,
        }
        if not has_refresh:
            values.update(refresh_token=None, refresh_expires_at=0)
        store.update_session(payload.user_id, values)
        return {"invalidated": True}


@app.get("/v1/status")
async def status(user_id: str) -> dict[str, Any]:
    row = store.row(user_id)
    if not row:
        return {"status": "unauthorized", "scopes": []}
    pending = store.authorization(user_id, pending_only=True)
    result: dict[str, Any] = {"status": row.status, "open_id": row.open_id, "user_name": row.user_name, "scopes": row.scope.split(), "access_expires_at": row.access_expires_at, "refresh_expires_at": row.refresh_expires_at}
    if pending:
        result.update({"status": "pending", "authorization_url": pending.verification_uri_complete or pending.verification_uri, "verification_uri": pending.verification_uri, "user_code": pending.user_code, "authorization_expires_at": pending.expires_at, "requested_scopes": pending.scope.split()})
    return result


@app.post("/v1/logout")
async def logout(payload: LogoutRequest) -> dict[str, str]:
    task = _poll_tasks.pop(payload.user_id, None)
    if task:
        task.cancel()
    store.update_session(payload.user_id, {
        "open_id": None,
        "user_name": None,
        "access_token": None,
        "refresh_token": None,
        "scope": "",
        "access_expires_at": 0,
        "refresh_expires_at": 0,
        "status": "unauthorized",
        "updated_at": int(time.time()),
    }, clear_authorization=True)
    return {"status": "logged_out"}


@app.post("/v1/migrate")
async def migrate(payload: MigrateRequest) -> dict[str, str]:
    if payload.app_id != settings.feishu.app_id or not payload.access_token:
        raise HTTPException(status_code=400, detail="invalid legacy token")
    now = int(time.time())
    store.update_session(payload.user_id, {
        "access_token": payload.access_token,
        "refresh_token": payload.refresh_token,
        "scope": payload.scope,
        "access_expires_at": payload.expires_at,
        "refresh_expires_at": payload.refresh_token_expires_at,
        "status": "authorized",
        "updated_at": now,
    }, clear_authorization=True, create=True)
    return {"status": "migrated"}


def run(host: str = "0.0.0.0", port: int = 7883) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, workers=1)
