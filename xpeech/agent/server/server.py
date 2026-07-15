from contextlib import asynccontextmanager

from fastapi import FastAPI

from ...config.settings import settings
from ...utils.logging import configure_logging
from ..tools.mcp_client import close_persistent_mcp_registrations, connect_persistent_mcp_registrations
from .middleware import ContextASGIMiddleware

configure_logging()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await connect_persistent_mcp_registrations(settings.tool.mcp_servers)
    try:
        yield
    finally:
        await close_persistent_mcp_registrations()


app = FastAPI(lifespan=lifespan)

app.add_middleware(ContextASGIMiddleware)
