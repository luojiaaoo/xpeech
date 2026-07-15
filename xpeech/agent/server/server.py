from contextlib import asynccontextmanager

from fastapi import FastAPI

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

app.add_middleware(ContextASGIMiddleware)
