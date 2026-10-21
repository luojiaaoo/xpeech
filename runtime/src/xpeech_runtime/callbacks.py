"""Narrow gateway operations available to a sandbox-owned agent loop."""

import os
from uuid import uuid4

import httpx


async def gateway_call(action: str, payload: dict) -> object:
    url = os.environ["XPEECH_RUNTIME_CALLBACK_URL"].rstrip("/")
    token = os.environ["XPEECH_RUNTIME_TOKEN"]
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            url + "/callback",
            headers={"X-Xpeech-Runtime-Token": token},
            json={"action": action, "payload": payload, "call_id": uuid4().hex},
        )
    response.raise_for_status()
    return response.json()["result"]


class GatewayRecords:
    async def append(self, record) -> None:
        await gateway_call("record", record.model_dump(mode="json"))
