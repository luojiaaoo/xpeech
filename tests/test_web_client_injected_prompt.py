import shlex
import sys

import pytest
from fastapi import HTTPException
from loguru import logger

from xpeech.subapp.channel.web_client.models import InjectPromptWebConfig
from xpeech.subapp.channel.web_client.routes.auth import resolve_injected_prompt


@pytest.mark.asyncio
async def test_injected_prompt_logs_gateway_error_without_command_data():
    script = "import sys; print('private-stdout'); print('private-stderr', file=sys.stderr); sys.exit(7)"
    template = shlex.join([sys.executable, "-c", script, "$state"])
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}", level="WARNING")
    try:
        with pytest.raises(HTTPException) as raised:
            await resolve_injected_prompt(
                InjectPromptWebConfig(enabled=True, command_template=template),
                "private-state",
            )
    finally:
        logger.remove(sink_id)

    assert raised.value.status_code == 502
    assert raised.value.detail == "提示词命令执行失败"
    assert messages == ["inject_prompt.command_template failed with exit code 7 (reason=failed, status_code=502)\n"]
