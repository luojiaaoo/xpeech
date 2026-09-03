import argparse
import hashlib
import hmac
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import Cookie, Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .dao import User, WebClientDAO
from .models import (
    InjectPromptWebConfig,
    OAuth2WebConfig,
    WebConfig,
)
from .routes.admin import create_admin_router
from .routes.auth import create_auth_router
from .routes.proxy import create_proxy_router

PBKDF2_ITERATIONS = 600_000
XPEECH_FAVICON = Path(__file__).resolve().parents[3] / "assets" / "favicon.ico"


def _configured_system_name() -> str:
    from ...config.settings import settings

    return settings.llm.system_name.strip() or "AI 助手"


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


def create_app(config: WebConfig) -> FastAPI:
    dao = WebClientDAO(config.database_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await dao.initialize(lambda: _password_hash("admin123456"))
        try:
            yield
        finally:
            await dao.close()

    app = FastAPI(
        title=f"{config.system_name} Web",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse(
            XPEECH_FAVICON,
            media_type="image/x-icon",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/api/config")
    async def public_config():
        return {
            "system_name": config.system_name,
            "inject_prompt": {"enabled": config.inject_prompt.enabled},
            "oauth2": (
                {
                    "enabled": True,
                    "provider_name": config.oauth2.provider_name,
                    "display_type": config.oauth2.display_type,
                }
                if config.oauth2 is not None
                else {
                    "enabled": False,
                    "provider_name": "OAuth2",
                    "display_type": "qrcode",
                }
            ),
        }

    async def current_user(
        token: Annotated[str | None, Cookie(alias=config.cookie_name)] = None,
    ) -> User:
        if not token:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "请先登录")
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        user = await dao.get_user_for_session(token_hash)
        if user is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "登录已失效")
        return user

    CurrentUser = Annotated[User, Depends(current_user)]

    def admin_user(user: CurrentUser) -> User:
        if not user.is_admin:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "需要管理员权限")
        return user

    app.include_router(
        create_auth_router(
            config,
            dao,
            current_user,
            _password_hash,
            _password_matches,
        )
    )
    app.include_router(create_admin_router(dao, admin_user, _password_hash))
    app.include_router(
        create_proxy_router(config.backend_url, current_user, admin_user)
    )

    if config.static_dir.exists():
        assets = config.static_dir / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            candidate = (config.static_dir / path).resolve()
            if (
                path
                and candidate.is_relative_to(config.static_dir.resolve())
                and candidate.is_file()
            ):
                return FileResponse(candidate)
            return FileResponse(config.static_dir / "index.html")

    else:

        @app.get("/", include_in_schema=False)
        async def missing_frontend():
            return JSONResponse(
                {
                    "detail": (
                        "前端尚未构建，请先在 web_client/frontend 运行 "
                        "npm install && npm run build"
                    )
                },
                status_code=503,
            )

    return app


def run(
    host: str = "0.0.0.0",
    port: int = 7939,
    backend_url: str = "http://127.0.0.1:7878",
    dev_frontend: bool = False,
) -> None:
    from ...config.settings import settings

    frontend = Path(__file__).parent / "frontend"
    static_dir = frontend / ("dist" if not dev_frontend else "dist")
    oauth2_settings = settings.web_client.oauth2
    oauth2_config = (
        OAuth2WebConfig(
            provider_name=oauth2_settings.provider_name,
            display_type=oauth2_settings.display_type,
            client_id=oauth2_settings.client_id,
            client_secret=oauth2_settings.client_secret,
            authorization_url=oauth2_settings.authorization_url,
            token_url=oauth2_settings.token_url,
            userinfo_url=oauth2_settings.userinfo_url,
            redirect_uri=oauth2_settings.redirect_uri,
            scopes=tuple(oauth2_settings.scopes),
            session_id_claim=oauth2_settings.session_id_claim,
            username_claim=oauth2_settings.username_claim,
            auto_create_users=oauth2_settings.auto_create_users,
            use_pkce=oauth2_settings.use_pkce,
            token_auth_method=oauth2_settings.token_auth_method,
            extra_authorization_params=dict(
                oauth2_settings.extra_authorization_params
            ),
        )
        if oauth2_settings.enabled
        else None
    )
    config = WebConfig(
        backend_url=backend_url.rstrip("/"),
        database_path=settings.web_client.database_path.resolve(),
        static_dir=static_dir.resolve(),
        system_name=_configured_system_name(),
        cookie_name=settings.web_client.cookie_name,
        oauth2=oauth2_config,
        inject_prompt=InjectPromptWebConfig(
            enabled=settings.web_client.inject_prompt.enabled,
            command_template=settings.web_client.inject_prompt.command_template,
        ),
    )
    uvicorn.run(create_app(config), host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Run the authenticated {_configured_system_name()} web client."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7939)
    parser.add_argument("--backend-url", default="http://127.0.0.1:7878")
    args = parser.parse_args()
    run(args.host, args.port, args.backend_url)


if __name__ == "__main__":
    main()
