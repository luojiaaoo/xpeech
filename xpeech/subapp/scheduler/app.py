"""Single-process scheduler API. Only this service opens schedule.db."""

import asyncio
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.responses import PlainTextResponse
from pydantic import Field, model_validator

from ...agent.server.auth import require_jwt
from ...agent.server.dependencies import user_id_header
from ...agent.server.schema import BackgroundMessageChannel, FeishuBackgroundMessage
from ...agent.tools.schedule import FeishuScheduleArgs
from ...agent.tools.schedule.schema import ScheduledTask
from ...utils.helper import ensure_async
from ...utils.logging import configure_logging
from .background import (
    BACKGROUND_MESSAGE_QUEUES,
    cancel_scheduled_task,
    list_scheduled_tasks,
    schedule_feishu_task,
    start_background_scheduler,
    stop_background_scheduler,
)

BACKGROUND_MESSAGE_LONG_POLL_SECONDS = 50


class ScheduleCreateRequest(FeishuScheduleArgs):
    sender_name: str = Field(min_length=1)
    session_metadata: dict[str, str]

    @model_validator(mode="after")
    def validate_delivery_context(self):
        if self.session_metadata.get("channel") != "feishu":
            raise ValueError("feishu_schedule is only available for Feishu sessions")
        if not self.session_metadata.get("open_id", "").strip():
            raise ValueError("session_metadata.open_id must be a non-empty string")
        if not self.sender_name.strip():
            raise ValueError("sender_name must be a non-empty string")
        return self


@asynccontextmanager
async def lifespan(_app: FastAPI):
    start_background_scheduler()
    try:
        yield
    finally:
        stop_background_scheduler()


app = FastAPI(lifespan=lifespan)


@app.get("/health", include_in_schema=False)
async def health():
    return PlainTextResponse("ok")


@app.post("/schedules", dependencies=[Depends(require_jwt)])
async def create_schedule(
    request: ScheduleCreateRequest,
    user_id: Annotated[str, Depends(user_id_header)],
) -> dict[str, str | None]:
    try:
        job = await ensure_async(schedule_feishu_task)(
            user_id=user_id, **request.model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "job_id": job.id,
        "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
    }


@app.get("/schedules", dependencies=[Depends(require_jwt)])
async def list_schedules(
    user_id: Annotated[str, Depends(user_id_header)],
) -> list[ScheduledTask]:
    return await ensure_async(list_scheduled_tasks)(user_id)


@app.delete("/schedules/{job_id}", dependencies=[Depends(require_jwt)], status_code=204)
async def delete_schedule(
    job_id: str,
    user_id: Annotated[str, Depends(user_id_header)],
) -> Response:
    try:
        await ensure_async(cancel_scheduled_task)(user_id=user_id, job_id=job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)


@app.get(
    "/background_message",
    dependencies=[Depends(require_jwt)],
    response_model=FeishuBackgroundMessage,
    responses={204: {"description": "No background message available"}},
)
async def poll_background_message(
    channel: BackgroundMessageChannel,
) -> FeishuBackgroundMessage | Response:
    queue = BACKGROUND_MESSAGE_QUEUES[channel]
    try:
        return await asyncio.wait_for(queue.get(), timeout=BACKGROUND_MESSAGE_LONG_POLL_SECONDS)
    except TimeoutError:
        return Response(status_code=204)


def run(host: str = "0.0.0.0", port: int = 8090, *, backend_url: str) -> None:
    import uvicorn

    configure_logging()
    app.state.backend_url = backend_url
    uvicorn.run(app, host=host, port=port, workers=1, lifespan="on")
