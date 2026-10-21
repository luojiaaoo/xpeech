"""Authenticated HTTP/SSE boundary around the complete agent runtime."""

import asyncio
import json
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from xpeech.agent.loop import AgentLoop
from xpeech.agent.runtime_runner import AgentRunner
from xpeech.agent.server.context import user_id_var
from xpeech.agent.server.schema import InboundMessage
from xpeech.agent.tools.mcp_client import close_all_mcp_connections
from .callbacks import GatewayRecords


@asynccontextmanager
async def lifespan(app):
    try:
        yield
    finally:
        await close_all_mcp_connections()


def require_token(x_xpeech_runtime_token: str = Header(default="")) -> None:
    expected = os.environ.get("XPEECH_RUNTIME_TOKEN", "")
    if not expected or not secrets.compare_digest(expected, x_xpeech_runtime_token):
        raise HTTPException(status_code=401, detail="Invalid runtime capability")


app = FastAPI(lifespan=lifespan, dependencies=[Depends(require_token)])
_run_lock = asyncio.Lock()
HEARTBEAT_SECONDS = 10.0


class RunRequest(BaseModel):
    message: InboundMessage
    background: bool = False


class AnswerRequest(BaseModel):
    answer: str


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/answer")
async def answer(request: AnswerRequest):
    question = AgentLoop.SESSION_QUESTION_EVENT.get(os.environ["XPEECH_RUNTIME_USER_ID"])
    if question is None:
        return {"received": False}
    question.answer = request.answer
    question.event.set()
    return {"received": True}


@app.post("/run")
async def run(request: RunRequest):
    expected_user = os.environ["XPEECH_RUNTIME_USER_ID"]
    if request.message.user_id != expected_user:
        raise HTTPException(status_code=403, detail="Runtime belongs to another user")
    if _run_lock.locked():
        raise HTTPException(status_code=409, detail="Runtime already executing a loop")
    await _run_lock.acquire()

    queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue(maxsize=32)

    async def execute():
        context = user_id_var.set(expected_user)
        terminal = ("runtime_done", "{}")
        try:
            runner = await AgentRunner.create(expected_user)
            runner._agent_loop.records = GatewayRecords()
            async for event in runner.run(request.message, background=request.background):
                await queue.put(("event", event))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Container agent loop failed")
            terminal = ("runtime_error", "Agent loop failed; inspect sandbox logs")
        finally:
            AgentLoop.SESSION_QUESTION_EVENT.pop(expected_user, None)
            try:
                await close_all_mcp_connections()
            except Exception:
                logger.exception("Container MCP cleanup failed")
                terminal = ("runtime_error", "Agent resource cleanup failed; inspect sandbox logs")
            finally:
                user_id_var.reset(context)
                _run_lock.release()
        await queue.put(terminal)

    async def events():
        worker = asyncio.create_task(execute())
        try:
            while True:
                try:
                    kind, payload = await asyncio.wait_for(queue.get(), HEARTBEAT_SECONDS)
                except TimeoutError:
                    # Keep proxy connections alive while a model/tool runs or
                    # the loop waits for a user's answer. Do not cancel it.
                    yield ": keepalive\n\n"
                    continue
                if kind == "event":
                    yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                else:
                    await worker
                    yield f"event: {kind}\ndata: {payload}\n\n"
                    break
        finally:
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    return StreamingResponse(events(), media_type="text/event-stream")
