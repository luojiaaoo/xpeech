import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse
from loguru import logger

from ...utils.logging import configure_logging
from ..background import start_background_scheduler, stop_background_scheduler
from ..record import create_db_and_tables, record_engine
from ..delegation import ACTIVE_USER_RUNS, release_runtime_lease
from .middleware import ContextASGIMiddleware

configure_logging()


async def _close_runtime(lease):
    try:
        if lease.sandbox is not None:
            async with asyncio.timeout(20):
                await lease.sandbox.destroy()
    except Exception:
        # One unavailable sandbox must not prevent the remaining containers
        # or the database from being closed. The finite sandbox TTL is fallback.
        logger.exception("Failed to destroy runtime during gateway shutdown")
    finally:
        release_runtime_lease(lease)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await create_db_and_tables()
    try:
        start_background_scheduler()
        yield
    finally:
        stop_background_scheduler()
        await asyncio.gather(*(_close_runtime(lease) for lease in list(ACTIVE_USER_RUNS.values())))
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
