"""Scoped container-to-gateway callbacks; no model or tool execution here."""

import asyncio
from typing import Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from ...background import cancel_scheduled_task, list_scheduled_tasks, schedule_feishu_task
from ...delegation import ACTIVE_RUNTIME_LEASES
from ...record import ConversationRecord, SqliteConversationRecordRepository

router = APIRouter(prefix="/internal/runtime", tags=["runtime"])


class CallbackRequest(BaseModel):
    action: Literal["record", "schedule", "schedule_list", "schedule_cancel"]
    payload: dict = Field(default_factory=dict)
    call_id: str = Field(min_length=1, max_length=128)


@router.post("/callback")
async def callback(request: CallbackRequest, x_xpeech_runtime_token: str = Header(default="")):
    lease = ACTIVE_RUNTIME_LEASES.get(x_xpeech_runtime_token)
    if lease is None or lease.message is None:
        raise HTTPException(status_code=401, detail="Invalid or expired runtime capability")
    async with lease.callback_lock:
        return await _execute_callback(request, lease)


async def _execute_callback(request, lease):
    message = lease.message
    previous = lease.completed_callbacks.get(request.call_id)
    if previous is not None:
        action, payload, result = previous
        if action != request.action or payload != request.payload:
            raise HTTPException(status_code=409, detail="Callback ID reused for another operation")
        return {"result": result}
    try:
        if request.action == "record":
            if lease.record_received:
                raise HTTPException(status_code=409, detail="Record already received")
            values = dict(request.payload)
            values.update(id=None, user_id=message.user_id, sender_name=message.sender_name)
            record = ConversationRecord.model_validate(values)
            await SqliteConversationRecordRepository().append(record)
            lease.record_received = True
            result = None
        elif request.action == "schedule":
            if message.session_metadata.get("channel") != "feishu":
                raise HTTPException(status_code=403, detail="Scheduling requires a Feishu session")
            if not message.session_metadata.get("open_id", "").strip():
                raise ValueError("Scheduling requires the original Feishu open_id")
            if not isinstance(request.payload.get("prompt"), str) or not request.payload["prompt"].strip():
                raise ValueError("prompt must be a non-empty string")
            from datetime import datetime

            run_at = request.payload.get("run_at")
            job = await asyncio.to_thread(
                schedule_feishu_task,
                prompt=request.payload["prompt"],
                run_at=datetime.fromisoformat(run_at) if run_at else None,
                cron=request.payload.get("cron"),
                user_id=message.user_id,
                sender_name=message.sender_name,
                session_metadata=message.session_metadata,
            )
            result = {"job_id": job.id, "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None}
        elif request.action == "schedule_list":
            tasks = await asyncio.to_thread(list_scheduled_tasks, message.user_id)
            result = [task.model_dump() for task in tasks]
        else:
            await asyncio.to_thread(cancel_scheduled_task, user_id=message.user_id, job_id=request.payload["job_id"])
            result = None
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    lease.completed_callbacks[request.call_id] = (request.action, request.payload, result)
    return {"result": result}
