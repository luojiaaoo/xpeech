"""Track cancellable SSE producer tasks inside one backend process."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import Any

import anyio


@dataclass
class _ActiveChat:
    task: asyncio.Task[Any]
    stopping: bool = False


class ChatStreams:
    def __init__(self) -> None:
        self._active: dict[tuple[str, str], _ActiveChat] = {}

    async def stream(
        self, user_id: str, session_id: str, events: AsyncGenerator[dict[str, Any], None],
    ) -> AsyncIterator[dict[str, Any]]:
        """Keep Agent cancellation separate from FastAPI's SSE task group."""
        send, receive = anyio.create_memory_object_stream[dict[str, Any]](max_buffer_size=1)

        async def produce() -> None:
            async with send:
                try:
                    async for event in events:
                        await send.send(event)
                finally:
                    # 停止时必须等待 loop 的历史保存完成，不能依赖异步生成器的 GC。
                    with anyio.CancelScope(shield=True):
                        await events.aclose()

        task = asyncio.create_task(produce())
        key = (user_id, session_id)
        active = _ActiveChat(task)
        self._active[key] = active

        def discard(_task: asyncio.Task[Any]) -> None:
            if self._active.get(key) is active:
                self._active.pop(key)

        task.add_done_callback(discard)
        consumer = asyncio.current_task()

        def consumer_done(_task: asyncio.Task[Any]) -> None:
            # A consumer can finish while this generator is suspended at yield.
            if not task.done() and not task.cancelling():
                task.cancel()

        if consumer is not None:
            consumer.add_done_callback(consumer_done)
        try:
            async with receive:
                async for event in receive:
                    if task.cancelled():
                        break
                    yield event
            if not task.cancelled():
                await task
        finally:
            if consumer is not None:
                consumer.remove_done_callback(consumer_done)
            if not task.done() and not task.cancelling():
                task.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.gather(task, return_exceptions=True)
            task.remove_done_callback(discard)
            discard(task)

    async def stop(self, user_id: str, session_id: str) -> bool:
        """Cancel once and wait for the owner task to return its model permit."""
        active = self._active.get((user_id, session_id))
        if active is None or active.task.done():
            return False
        if not active.stopping:
            active.stopping = True
            if not active.task.cancelling():
                active.task.cancel()
        # Shield cleanup from cancellation of the independent stop request.
        await asyncio.shield(asyncio.gather(active.task, return_exceptions=True))
        return True


CHAT_STREAMS = ChatStreams()
