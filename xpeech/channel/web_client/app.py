from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import secrets
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, AsyncIterator
from urllib.parse import quote

import httpx
import uvicorn
from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

COOKIE_NAME = "xpeech_session"
SESSION_DAYS = 7
PBKDF2_ITERATIONS = 600_000


def _configured_system_name() -> str:
    from ...config.settings import settings

    return settings.llm.system_name.strip() or "AI 助手"


@dataclass(frozen=True)
class WebConfig:
    backend_url: str
    database_path: Path
    static_dir: Path
    secure_cookie: bool
    system_name: str


class LoginBody(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class UserBody(BaseModel):
    username: str = Field(min_length=1, max_length=64, pattern=r"^[\w.@+-]+$")
    password: str = Field(min_length=8, max_length=256)
    is_admin: bool = False


class UserUpdateBody(BaseModel):
    password: str | None = Field(default=None, min_length=8, max_length=256)
    is_admin: bool | None = None
    is_active: bool | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _password_hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def _password_matches(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt_hex, expected_hex = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(actual, bytes.fromhex(expected_hex))
    except (ValueError, TypeError):
        return False


class Database:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
                """
            )
            db.execute("DELETE FROM sessions WHERE expires_at <= ?", (_now(),))
            if db.execute("SELECT 1 FROM users LIMIT 1").fetchone() is None:
                username = os.getenv("XPEECH_WEB_ADMIN_USERNAME", "admin")
                password = os.getenv("XPEECH_WEB_ADMIN_PASSWORD", "admin123456")
                db.execute(
                    "INSERT INTO users(username, password_hash, is_admin, created_at) VALUES (?, ?, 1, ?)",
                    (username, _password_hash(password), _now()),
                )


def _public_user(row: sqlite3.Row) -> dict[str, object]:
    return {
        "id": row["id"],
        "username": row["username"],
        "is_admin": bool(row["is_admin"]),
        "is_active": bool(row["is_active"]),
        "created_at": row["created_at"],
    }


def _web_session_id(user: sqlite3.Row) -> str:
    return f"web_{user['username']}"


def create_app(config: WebConfig) -> FastAPI:
    database = Database(config.database_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        database.initialize()
        yield

    app = FastAPI(
        title=f"{config.system_name} Web",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.get("/api/config")
    async def public_config():
        return {"system_name": config.system_name}

    def current_user(token: Annotated[str | None, Cookie(alias=COOKIE_NAME)] = None):
        if not token:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "请先登录")
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with database.connect() as db:
            row = db.execute(
                """
                SELECT users.* FROM sessions
                JOIN users ON users.id = sessions.user_id
                WHERE sessions.token_hash = ? AND sessions.expires_at > ? AND users.is_active = 1
                """,
                (token_hash, _now()),
            ).fetchone()
        if row is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "登录已失效")
        return row

    def admin_user(user=Depends(current_user)):
        if not user["is_admin"]:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "需要管理员权限")
        return user

    @app.post("/api/auth/login")
    async def login(body: LoginBody, response: Response):
        with database.connect() as db:
            user = db.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (body.username,)
            ).fetchone()
            if user is None or not user["is_active"] or not _password_matches(body.password, user["password_hash"]):
                raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户名或密码错误")
            token = secrets.token_urlsafe(32)
            expires = datetime.now(UTC) + timedelta(days=SESSION_DAYS)
            db.execute(
                "INSERT INTO sessions(token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
                (hashlib.sha256(token.encode()).hexdigest(), user["id"], expires.isoformat(), _now()),
            )
        response.set_cookie(
            COOKIE_NAME,
            token,
            max_age=SESSION_DAYS * 86400,
            httponly=True,
            secure=config.secure_cookie,
            samesite="lax",
            path="/",
        )
        return _public_user(user)

    @app.post("/api/auth/logout", status_code=204)
    async def logout(
        response: Response,
        token: Annotated[str | None, Cookie(alias=COOKIE_NAME)] = None,
    ):
        if token:
            with database.connect() as db:
                db.execute("DELETE FROM sessions WHERE token_hash = ?", (hashlib.sha256(token.encode()).hexdigest(),))
        response.delete_cookie(COOKIE_NAME, path="/")

    @app.get("/api/auth/me")
    async def me(user=Depends(current_user)):
        return _public_user(user)

    @app.get("/api/admin/users")
    async def list_users(_=Depends(admin_user)):
        with database.connect() as db:
            rows = db.execute("SELECT * FROM users ORDER BY id").fetchall()
        return [_public_user(row) for row in rows]

    @app.post("/api/admin/users", status_code=201)
    async def create_user(body: UserBody, _=Depends(admin_user)):
        try:
            with database.connect() as db:
                cursor = db.execute(
                    "INSERT INTO users(username, password_hash, is_admin, created_at) VALUES (?, ?, ?, ?)",
                    (body.username, _password_hash(body.password), int(body.is_admin), _now()),
                )
                row = db.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
        except sqlite3.IntegrityError:
            raise HTTPException(status.HTTP_409_CONFLICT, "用户名已存在")
        return _public_user(row)

    @app.patch("/api/admin/users/{user_id}")
    async def update_user(
        user_id: int,
        body: UserUpdateBody,
        admin=Depends(admin_user),
    ):
        values = body.model_dump(exclude_none=True)
        if user_id == admin["id"] and values.get("is_active") is False:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "不能停用当前管理员")
        fields: list[str] = []
        params: list[object] = []
        if "password" in values:
            fields.append("password_hash = ?")
            params.append(_password_hash(values["password"]))
        for key in ("is_admin", "is_active"):
            if key in values:
                fields.append(f"{key} = ?")
                params.append(int(values[key]))
        if not fields:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "没有可更新字段")
        params.append(user_id)
        with database.connect() as db:
            if db.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone() is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "用户不存在")
            db.execute(f"UPDATE users SET {', '.join(fields)} WHERE id = ?", params)
            if values.get("is_active") is False:
                db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _public_user(row)

    def backend_headers(user: sqlite3.Row) -> dict[str, str]:
        return {"x-session-id": _web_session_id(user)}

    @app.post("/api/chat")
    async def proxy_chat(
        request: Request,
        content: Annotated[str, Form()],
        session_metadata: Annotated[str, Form()] = "{}",
        timestamp: Annotated[str | None, Form()] = None,
        files: Annotated[list[UploadFile], File()] = [],
        user=Depends(current_user),
    ):
        form_data = {"content": content, "session_metadata": session_metadata}
        if timestamp:
            form_data["timestamp"] = timestamp
        upload_data = []
        for file in files:
            upload_data.append(("files", (file.filename or "attachment", await file.read(), file.content_type)))
        client = httpx.AsyncClient(timeout=None)
        upstream = await client.send(
            client.build_request(
                "POST",
                f"{config.backend_url}/chat",
                headers=backend_headers(user),
                data=form_data,
                files=upload_data,
            ),
            stream=True,
        )
        if upstream.status_code >= 400:
            body = await upstream.aread()
            await upstream.aclose()
            await client.aclose()
            return Response(body, status_code=upstream.status_code, media_type=upstream.headers.get("content-type"))

        async def stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_bytes():
                    if await request.is_disconnected():
                        break
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(
            stream(),
            media_type=upstream.headers.get("content-type", "text/event-stream"),
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/answer_question")
    async def proxy_answer(
        answer: Annotated[str, Form()],
        user=Depends(current_user),
    ):
        async with httpx.AsyncClient(timeout=30) as client:
            upstream = await client.post(
                f"{config.backend_url}/answer_question",
                headers=backend_headers(user),
                data={"answer": answer},
            )
        return Response(upstream.content, status_code=upstream.status_code, media_type=upstream.headers.get("content-type"))

    @app.get("/api/files")
    async def proxy_file(
        path: str,
        user=Depends(current_user),
    ):
        session_id = _web_session_id(user)
        async with httpx.AsyncClient(timeout=None) as client:
            upstream = await client.get(
                f"{config.backend_url}/sessions/{quote(session_id, safe='')}/files",
                params={"path": path},
            )
        headers = {}
        if disposition := upstream.headers.get("content-disposition"):
            headers["Content-Disposition"] = disposition
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
            headers=headers,
        )

    if config.static_dir.exists():
        assets = config.static_dir / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            candidate = (config.static_dir / path).resolve()
            if path and candidate.is_relative_to(config.static_dir.resolve()) and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(config.static_dir / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        async def missing_frontend():
            return JSONResponse(
                {"detail": "前端尚未构建，请先在 web_client/frontend 运行 npm install && npm run build"},
                status_code=503,
            )

    return app


def run(
    host: str = "0.0.0.0",
    port: int = 7880,
    backend_url: str = "http://127.0.0.1:7878",
    database_path: str = "data/web/users.db",
    dev_frontend: bool = False,
) -> None:
    frontend = Path(__file__).parent / "frontend"
    static_dir = frontend / ("dist" if not dev_frontend else "dist")
    config = WebConfig(
        backend_url=backend_url.rstrip("/"),
        database_path=Path(database_path).resolve(),
        static_dir=static_dir.resolve(),
        secure_cookie=os.getenv("XPEECH_WEB_SECURE_COOKIE", "").lower() in {"1", "true", "yes"},
        system_name=_configured_system_name(),
    )
    uvicorn.run(create_app(config), host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Run the authenticated {_configured_system_name()} web client."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7880)
    parser.add_argument("--backend-url", default="http://127.0.0.1:7878")
    parser.add_argument("--database", default="data/web/users.db")
    args = parser.parse_args()
    run(args.host, args.port, args.backend_url, args.database)


if __name__ == "__main__":
    main()
