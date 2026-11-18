"""Feishu connection and persistent scheduling in one application process."""

import asyncio
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from ....agent.server.auth import require_internal_jwt
from ....agent.server.dependencies import internal_user_id
from ....agent.tools.schedule.schema import (
    FeishuScheduleCreateRequest,
    ScheduleCancelResponse,
    ScheduleListResponse,
    ScheduleResponse,
)
from ....config.settings import settings
from ....utils.database import get_database_engine
from ....utils.helper import ensure_async
from ....utils.logging import configure_logging
from .bridge import FeishuBridge
from .schedule import (
    FeishuScheduleRuntime,
    cancel_scheduled_task,
    describe_scheduled_task,
    get_background_scheduler,
    list_scheduled_tasks,
    schedule_feishu_task,
    start_background_scheduler,
    stop_background_scheduler,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    bridge = FeishuBridge(app.state.chat_url, settings.feishu.app_id, settings.feishu.app_secret)
    async with httpx.AsyncClient() as client:
        runtime = FeishuScheduleRuntime(app.state.chat_url, client, bridge.channel)
        await ensure_async(start_background_scheduler)(
            get_database_engine(), runtime=runtime, event_loop=asyncio.get_running_loop()
        )
        # APScheduler needs the running loop even when job-store work is offloaded.
        try:
            async with asyncio.TaskGroup() as group:
                connection = group.create_task(bridge.start())
                try:
                    yield
                finally:
                    await stop_background_scheduler()
                    connection.cancel()
        finally:
            await stop_background_scheduler()


app = FastAPI(lifespan=lifespan)
router = APIRouter(dependencies=[Depends(require_internal_jwt)])


@app.get("/health", include_in_schema=False)
async def health():
    try:
        get_background_scheduler()
    except RuntimeError:
        return JSONResponse({"status": "unavailable", "scheduler": "stopped"}, status_code=503)
    return {"status": "ok", "scheduler": "running"}


@router.post("/schedules", response_model=ScheduleResponse)
async def create_schedule(
    request: FeishuScheduleCreateRequest,
    user_id: Annotated[str, Depends(internal_user_id)],
):
    def create():
        job = schedule_feishu_task(user_id=user_id, **request.model_dump())
        return ScheduleResponse(task=describe_scheduled_task(job))

    try:
        return await ensure_async(create)()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/schedules", response_model=ScheduleListResponse)
async def list_schedules(user_id: Annotated[str, Depends(internal_user_id)]):
    return ScheduleListResponse(tasks=await ensure_async(list_scheduled_tasks)(user_id))


@router.delete("/schedules/{job_id}", response_model=ScheduleCancelResponse)
async def delete_schedule(job_id: str, user_id: Annotated[str, Depends(internal_user_id)]):
    try:
        await ensure_async(cancel_scheduled_task)(user_id=user_id, job_id=job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ScheduleCancelResponse(job_id=job_id)


app.include_router(router)


def run(chat_url: str, host: str = "0.0.0.0", port: int = 8090) -> None:
    import uvicorn

    configure_logging()
    app.state.chat_url = chat_url
    uvicorn.run(app, host=host, port=port, workers=1, lifespan="on")
