from functools import partial
from pathlib import Path
from typing import Any

from ...config.settings import ToolConfig, settings
from ...utils.filter_expr import FilterContext, filter_allows_access
from ...utils.security.dcg import shell_requires_confirmation
from ..skills.skill import SkillsLoader
from .ask_user_question import ask_user_question
from .browser_preview import build_browser_preview_tool
from .document import build_document_tool
from .execution_request import build_execution_request_tool
from .file_message import build_file_message_tools
from .filesystem import build_file_tools
from .mcp_client import (
    MCPServerRegistration,
    build_mcp_status_tool,
    get_session_mcp_registration_from_config,
    register_mcp_bindings,
)
from .ptc import build_ptc_tool
from .research import build_research_tools
from .schedule import feishu_schedule, feishu_schedule_cancel, feishu_schedule_list
from .sessions import build_session_tools
from .shell import build_shell_tools
from .web import baidu_search, bing_search, web_fetch


async def register_tools(
    *,
    user_id: str,
    provider: Any,
    workspace: Path,
    config: ToolConfig,
    not_interactive: bool = False,
    channel: str | None = None,
    user_organization: str | None = None,
) -> None:
    """向模型提供方注册内置工具及配置中的 MCP 工具。"""
    workspace = workspace.resolve()
    skills_loader = SkillsLoader(
        workspace, channel=channel, user_organization=user_organization, not_interactive=not_interactive,
    )
    read_image, read_video, read_file, write_file, edit_file = build_file_tools(
        workspace=workspace,
        max_result_chars=config.max_result_chars,
    )
    if provider.support_image:
        provider.register_function_tool(read_image)
    if provider.support_video:
        provider.register_function_tool(read_video)
    for tool in (read_file, write_file, edit_file):
        provider.register_function_tool(tool)

    provider.register_function_tool(
        build_execution_request_tool(
            build_shell_tools(workspace=workspace, skills_loader=skills_loader),
            requires_confirmation=partial(shell_requires_confirmation, workspace=workspace),
            not_interactive=not_interactive,
        )
    )
    provider.register_function_tool(build_ptc_tool(workspace=workspace))
    provider.register_function_tool(web_fetch)
    for tool in (baidu_search, bing_search):
        provider.register_function_tool(tool)
    for tool in build_research_tools(workspace=workspace):
        provider.register_function_tool(tool)

    browser_preview = config.browser_preview
    provider.register_function_tool(
        build_browser_preview_tool(
            workspace=workspace,
            browser_preview_path=browser_preview.browser_preview_path,
            browser_preview_base_url=browser_preview.browser_preview_base_url,
        )
    )
    provider.register_function_tool(build_document_tool(workspace=workspace))
    provider.register_function_tool(build_file_message_tools(workspace=workspace))
    if not not_interactive:
        provider.register_function_tool(ask_user_question)
    for tool in build_session_tools(
        user_id=user_id,
        history_path=settings.path.session_history_path,
        max_files_per_user=settings.llm.max_history_files_per_user,
    ):
        provider.register_function_tool(tool)
    if not not_interactive:
        for tool in (feishu_schedule, feishu_schedule_list, feishu_schedule_cancel):
            provider.register_function_tool(tool)

    registrations: dict[str, MCPServerRegistration] = {}
    filter_context = FilterContext(channel=channel, org=user_organization)
    for server_name, server_config in config.mcp_servers.items():
        if not filter_allows_access(server_config.filter, filter_context):
            continue
        registration = await get_session_mcp_registration_from_config(
            user_id,
            server_name,
            server_config,
            workspace=workspace,
            skills_loader=skills_loader,
        )
        registrations[server_name] = registration

    if registrations:
        provider.register_function_tool(build_mcp_status_tool(registrations=registrations))
    for server_name, registration in registrations.items():
        server_config = config.mcp_servers[server_name]
        task = registration.start_tool_discovery()
        register_bindings = partial(
            register_mcp_bindings,
            registration=registration,
            provider=provider,
            execution_request_tools=server_config.execution_request_tools,
            not_interactive=not_interactive,
        )
        if task.done():
            register_bindings(task)
        else:
            task.add_done_callback(register_bindings)
