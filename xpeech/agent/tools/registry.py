from pathlib import Path
from typing import Any

from ...config.settings import ToolConfig, settings
from ...utils.channel_tags import channel_tags_allow_access
from .browser_preview import build_browser_preview_tool
from .document import build_document_tool
from .file_message import build_file_message_tools
from .filesystem import build_file_tools
from .mcp_client import get_session_mcp_registration_from_config
from .question import ask_user_question
from .research import build_research_tools
from .schedule import feishu_schedule, feishu_schedule_cancel, feishu_schedule_list
from .sessions import build_session_tools
from .shell import build_shell_tools
from .web import baidu_search, bing_search, news_search, web_fetch


async def register_tools(
    *,
    user_id: str,
    provider: Any,
    workspace: Path,
    config: ToolConfig,
    not_interactive: bool = False,
    channel_tags: list[str] | None = None,
) -> None:
    """向模型提供方注册内置工具及配置中的 MCP 工具。"""
    read_image, read_video, read_file, write_file, edit_file = build_file_tools(
        workspace=workspace,
        max_result_chars=config.max_result_chars,
    )
    if provider.support_image:
        provider.register_tool()(read_image)
    if provider.support_video:
        provider.register_tool()(read_video)
    for tool in (read_file, write_file, edit_file):
        provider.register_tool()(tool)

    provider.register_tool()(build_shell_tools(workspace=workspace))
    provider.register_tool()(web_fetch)
    for tool in (baidu_search, bing_search, news_search):
        provider.register_tool()(tool)
    for tool in build_research_tools(workspace=workspace):
        provider.register_tool()(tool)

    browser_preview = config.browser_preview
    provider.register_tool()(
        build_browser_preview_tool(
            workspace=workspace,
            browser_preview_path=browser_preview.browser_preview_path,
            browser_preview_base_url=browser_preview.browser_preview_base_url,
        )
    )
    provider.register_tool()(build_document_tool(workspace=workspace))
    provider.register_tool()(build_file_message_tools(workspace=workspace))
    if not not_interactive:
        provider.register_tool()(ask_user_question)
    for tool in build_session_tools(user_id=user_id, history_path=settings.path.session_history_path):
        provider.register_tool()(tool)
    if not not_interactive:
        for tool in (feishu_schedule, feishu_schedule_list, feishu_schedule_cancel):
            provider.register_tool()(tool)

    for server_name, server_config in config.mcp_servers.items():
        if not channel_tags_allow_access(server_config.channel_tags, channel_tags or []):
            continue
        registration = await get_session_mcp_registration_from_config(
            user_id,
            server_name,
            server_config,
            workspace=workspace,
        )
        await provider.register_tool("mcp")(registration)
