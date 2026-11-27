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
from .execution_request import ExecutionConfirmation, build_execution_request_tool
from .file_message import build_file_message_tools
from .filesystem import build_file_tools
from .fragment import fragment
from .mcp_client import (
    MCPServerRegistration,
    build_mcp_status_tool,
    get_session_mcp_registration_from_config,
    register_mcp_bindings,
)
from .research import build_research_tools
from .schedule import feishu_schedule_cancel, feishu_schedule_cron, feishu_schedule_list, feishu_schedule_once
from .sessions import build_session_tools
from .shell import build_shell_tools
from .skills import build_read_skill_tool
from .web import baidu_search, bing_search, web_fetch


async def register_tools(
    *,
    user_id: str,
    provider: Any,
    workspace: Path,
    config: ToolConfig,
    not_interactive: bool = False,
    channel: str,
    user_organization: str | None = None,
) -> None:
    """向模型提供方注册内置工具及配置中的 MCP 工具。"""
    workspace = workspace.resolve()
    skills_loader = SkillsLoader(
        workspace, user_id=user_id, channel=channel, user_organization=user_organization,
        not_interactive=not_interactive,
    )

    # 文本文件工具始终注册；图片和视频读取取决于模型能力。
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

    # skill 加载专属技能
    provider.register_function_tool(build_read_skill_tool(skills_loader=skills_loader))

    # Shell 命令在执行前经过安全检查，需要确认时走用户确认流程。
    provider.register_function_tool(
        build_execution_request_tool(
            build_shell_tools(workspace=workspace, skills_loader=skills_loader),
            requires_confirmation=ExecutionConfirmation(
                callback=partial(shell_requires_confirmation, workspace=workspace),
                prompt="危险操作、大批量不可恢复的操作或覆盖重要数据时需要用户确认；仅读取的命令无需确认。",
            ),
            not_interactive=not_interactive,
        )
    )

    # 网页抓取、搜索和论文检索工具。
    provider.register_function_tool(web_fetch)
    for tool in (baidu_search, bing_search):
        provider.register_function_tool(tool)
    for tool in build_research_tools(workspace=workspace):
        provider.register_function_tool(tool)

    # 工作区内容的预览、文档读取和文件发送。
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

    # fragment 是空操作；由工具描述要求模型只在用户明确请求时调用。
    provider.register_function_tool(fragment)

    # 后台执行无法等待用户回答，但仍可读取当前用户的会话历史。
    if not not_interactive:
        provider.register_function_tool(ask_user_question)
    for tool in build_session_tools(
        user_id=user_id,
        history_path=settings.path.session_history_path,
        max_history_days=settings.llm.max_history_days,
    ):
        provider.register_function_tool(tool)

    # 定时任务工具只开放给飞书交互会话。
    if channel == "feishu" and not not_interactive:
        for tool in (feishu_schedule_cron, feishu_schedule_once, feishu_schedule_list, feishu_schedule_cancel):
            provider.register_function_tool(tool)

    # 按当前用户、渠道和组织筛选允许接入的 MCP 服务。
    registrations: dict[str, MCPServerRegistration] = {}
    filter_context = FilterContext(channel=channel, org=user_organization, user_id=user_id)
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

    # 不等待 MCP 工具发现；发现结束后再将工具注册给提供方。
    for server_name, registration in registrations.items():
        server_config = config.mcp_servers[server_name]
        task = registration.start_tool_discovery()
        register_bindings = partial(
            register_mcp_bindings,
            registration=registration,
            provider=provider,
            execution_request_tools=server_config.execution_request_tools,
            execution_request_tool_prompts=server_config.execution_request_tool_prompts,
            not_interactive=not_interactive,
        )
        if task.done():
            register_bindings(task)
        else:
            task.add_done_callback(register_bindings)
