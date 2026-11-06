"""Standard-library-only command supervisor, executed inside the sandbox."""

import asyncio
import base64
import json
import os
import signal
import sys
import tempfile


async def execute(request: dict) -> dict:
    environment = os.environ.copy()
    for key, value in request["env"].items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        spawning = asyncio.create_task(
            asyncio.create_subprocess_exec(
                "bash",
                "-c",
                request["command"],
                stdin=asyncio.subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                start_new_session=True,
            )
        )
        timed_out = False
        try:
            process = await asyncio.shield(spawning)
            await asyncio.wait_for(process.wait(), request["timeout_ms"] / 1000)
        except (TimeoutError, asyncio.CancelledError) as error:
            timed_out = isinstance(error, TimeoutError)
            process = await spawning
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
        stdout.seek(0)
        stderr.seek(0)
        return {
            "stdout": base64.b64encode(stdout.read()).decode("ascii"),
            "stderr": base64.b64encode(stderr.read()).decode("ascii"),
            "returncode": process.returncode,
            "timed_out": timed_out,
        }


async def main() -> None:
    reader = asyncio.StreamReader(limit=16 * 1024 * 1024)
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_running_loop().connect_read_pipe(lambda: protocol, sys.stdin.buffer)
    tasks: dict[int, asyncio.Task] = {}

    async def respond(request):
        try:
            message = await execute(request)
        except Exception as error:  # noqa: BLE001 -- report one job's error without stopping other jobs.
            message = {"error": f"{type(error).__name__}: {error}"}
        message["id"] = request["id"]
        response = json.dumps(message).encode("utf-8")
        # No await between frame writes: concurrent jobs cannot mix responses.
        sys.stdout.buffer.write(str(len(response)).encode("ascii") + b"\n" + response)
        sys.stdout.buffer.flush()
        tasks.pop(request["id"], None)

    # The host waits for this handshake before accepting commands.
    print("ready", flush=True)
    while line := await reader.readline():
        request = json.loads(line)
        if request["op"] == "cancel":
            task = tasks.get(request["id"])
            if task is not None and not task.done() and not task.cancelling():
                task.cancel()
        else:
            tasks[request["id"]] = asyncio.create_task(respond(request))
            # Enter execute() before consuming a buffered cancellation request.
            await asyncio.sleep(0)


if __name__ == "__main__":
    asyncio.run(main())
