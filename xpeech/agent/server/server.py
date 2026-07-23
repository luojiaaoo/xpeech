import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from ...config.settings import settings
from ...utils.logging import configure_logging
from ..tools.mcp_client import close_persistent_mcp_registrations
from .middleware import ContextASGIMiddleware

configure_logging()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        yield
    finally:
        await close_persistent_mcp_registrations()


app = FastAPI(
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

################ doc文档basic HTTP 登录
docs_security = HTTPBasic()


def authenticate_docs(
    credentials: Annotated[HTTPBasicCredentials, Depends(docs_security)],
) -> None:
    username_is_correct = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        settings.docs.username.encode("utf-8"),
    )
    password_is_correct = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        settings.docs.password.encode("utf-8"),
    )
    if not (username_is_correct and password_is_correct):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )


@app.get(
    app.openapi_url,
    include_in_schema=False,
    dependencies=[Depends(authenticate_docs)],
)
async def openapi_json():
    return JSONResponse(app.openapi())


@app.get("/docs", include_in_schema=False, dependencies=[Depends(authenticate_docs)])
async def swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
    )


@app.get(
    app.swagger_ui_oauth2_redirect_url,
    include_in_schema=False,
    dependencies=[Depends(authenticate_docs)],
)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@app.get("/redoc", include_in_schema=False, dependencies=[Depends(authenticate_docs)])
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - ReDoc",
    )


######### 头像
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(Path("assets") / "favicon.ico")


######### 健康扫描
@app.get("/health", include_in_schema=False)
async def health():
    return PlainTextResponse("ok")


######### 开发者工具
@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def well_known():
    return PlainTextResponse("", status_code=204)


app.add_middleware(ContextASGIMiddleware)
