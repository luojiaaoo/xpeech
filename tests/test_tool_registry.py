from pathlib import Path
from typing import Any

import pytest

from xpeech.agent.tools import registry
from xpeech.config.settings import ToolConfig


def named_tool(name: str):
    async def tool():
        return name

    tool.__name__ = name
    return tool


class FakeProvider:
    def __init__(self) -> None:
        self.support_image = False
        self.support_video = True
        self.registered: list[str] = []
        self.mcp_registrations: list[Any] = []

    def register_tool(self, tool_type: str = "function"):
        if tool_type == "mcp":
            async def register_mcp(registration):
                self.mcp_registrations.append(registration)
                return registration

            return register_mcp

        def register_function(tool):
            self.registered.append(tool.__name__)
            return tool

        return register_function


@pytest.mark.asyncio
async def test_registers_supported_default_and_mcp_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    file_tools = tuple(
        named_tool(name)
        for name in ("read_image", "read_video", "read_file", "write_file", "edit_file", "read_office_file")
    )
    monkeypatch.setattr(registry, "build_file_tools", lambda **_kwargs: file_tools)
    monkeypatch.setattr(registry, "build_shell_tools", lambda **_kwargs: named_tool("shell"))
    monkeypatch.setattr(registry, "web_fetch", named_tool("web_fetch"))
    monkeypatch.setattr(registry, "web_search", named_tool("web_search"))
    monkeypatch.setattr(registry, "build_browser_preview_tool", lambda **_kwargs: named_tool("browser_preview"))
    monkeypatch.setattr(registry, "build_file_message_tools", lambda **_kwargs: named_tool("send_file"))
    monkeypatch.setattr(registry, "ask_user_question", named_tool("ask_user_question"))

    registration = object()

    async def get_registration(*_args, **_kwargs):
        return registration

    monkeypatch.setattr(registry, "get_persistent_mcp_registration_from_config", get_registration)
    provider = FakeProvider()
    config = ToolConfig.model_validate({"mcpServers": {"demo": {"command": "demo"}}})

    await registry.register_default_tools(provider=provider, workspace=tmp_path, config=config)

    assert "read_image" not in provider.registered
    assert provider.registered == [
        "read_video",
        "read_file",
        "write_file",
        "edit_file",
        "shell",
        "web_fetch",
        "web_search",
        "browser_preview",
        "read_office_file",
        "send_file",
        "ask_user_question",
    ]
    assert provider.mcp_registrations == [registration]
