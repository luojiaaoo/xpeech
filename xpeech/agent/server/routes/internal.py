"""Internal service-to-service endpoints for scheduled Agent execution."""

from typing import Annotated

from fastapi import APIRouter, Depends

from ...execute import execute_scheduled_chat
from ...tools.schedule.schema import ScheduledExecution, ScheduledExecutionResult
from ..dependencies import user_id_header

router = APIRouter(prefix="/internal")


@router.post("/scheduled-run", include_in_schema=False)
async def scheduled_run(
    request: ScheduledExecution,
    user_id: Annotated[str, Depends(user_id_header)],
) -> ScheduledExecutionResult:
    return await execute_scheduled_chat(user_id, request)
