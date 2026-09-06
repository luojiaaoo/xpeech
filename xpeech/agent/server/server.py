from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse

from ...utils.logging import configure_logging
from ..background import start_background_scheduler, stop_background_scheduler
from ..record import create_db_and_tables, record_engine
from ..tools.mcp_client import close_persistent_mcp_registrations
from .middleware import ContextASGIMiddleware

configure_logging()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await create_db_and_tables()
    try:
        start_background_scheduler()
        yield
    finally:
        stop_background_scheduler()
        await close_persistent_mcp_registrations()
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
