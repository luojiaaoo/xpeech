from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse
from starlette.middleware.cors import CORSMiddleware

from ...utils.logging import configure_logging
from ..record import create_db_and_tables, record_engine
from ..tools.mcp_client import close_all_mcp_connections
from ..tools.shell.session import close_all_shell_sandboxes
from .middleware import ContextASGIMiddleware

configure_logging()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await create_db_and_tables()
    try:
        yield
    finally:
        try:
            await close_all_mcp_connections()
        finally:
            await close_all_shell_sandboxes()
            await record_engine.dispose()


app = FastAPI(
    lifespan=lifespan,
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
