from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse

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


app = FastAPI(lifespan=lifespan)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("assets" / "favicon.ico")


@app.get("/health", include_in_schema=False)
async def health():
    return PlainTextResponse("ok")


app.add_middleware(ContextASGIMiddleware)
