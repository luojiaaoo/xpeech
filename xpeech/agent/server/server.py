from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse
from openai import AsyncOpenAI
from starlette.middleware.cors import CORSMiddleware

from ...config.settings import settings
from ...utils.database import get_async_database_engine
from ...utils.logging import configure_logging
from ...utils.redis_state import redis_pool
from ...utils.security.sandbox import close_all_sandboxes
from ..record import create_db_and_tables
from ..tools.mcp_client import close_all_mcp_connections
from .middleware import ContextASGIMiddleware

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = get_async_database_engine()
    async with AsyncExitStack() as cleanup:
        await cleanup.enter_async_context(redis_pool())
        cleanup.push_async_callback(engine.dispose)
        await create_db_and_tables(engine)
        cleanup.push_async_callback(close_all_sandboxes)
        cleanup.push_async_callback(close_all_mcp_connections)
        llm = settings.llm
        app.state.openai_client = await cleanup.enter_async_context(
            AsyncOpenAI(api_key=llm.api_key, base_url=llm.api_base, max_retries=8)
        )
        cleanup.callback(delattr, app.state, "openai_client")
        yield


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
