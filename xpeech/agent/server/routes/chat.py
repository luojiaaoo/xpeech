from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Header, UploadFile
from fastapi.sse import EventSourceResponse

from ....config.settings import settings
from ....provider.litellm_provider import LiteLLMProvider
from ....provider.schema import ProviderChatKwargs
from ....utils.helper import ensure_path_async, save_to_workspace
from ....utils.session import create_workspace_templates
from ...loop import AgentLoop, QuestionEvent
from ...tools.registry import register_default_tools
from ..dependencies import acquire_chat_session, content, sender_name_header, session_metadata
from ..schema import InboundMessage, InputContent

router = APIRouter()


@router.post("/answer_question")
async def answer_question(
    session_id: Annotated[str, Header(description="会话的ID", alias="x-session-id")],
    answer: Annotated[str, Form(description="回答内容")],
):
    """提交用户答案并唤醒正在等待的 Agent 循环。"""
    question_event: QuestionEvent | None = AgentLoop.SESSION_QUESTION_EVENT.get(session_id)
    if question_event is not None:
        question_event.answer = answer
        question_event.event.set()
        return {"message": "Answer received"}
    return {"message": "Question event not found"}


@router.post(
    "/chat",
    response_class=EventSourceResponse,
    dependencies=[Depends(acquire_chat_session, scope="request")],
)
async def chat(
    session_id: Annotated[str, Header(description="会话的ID", alias="x-session-id")],
    sender_name: Annotated[str, Depends(sender_name_header)],
    session_metadata: Annotated[dict[str, str], Depends(session_metadata)],
    content: Annotated[InputContent, Depends(content)],
    timestamp: Annotated[datetime, Form(default_factory=datetime.now, description="消息的时间戳")],
    files: Annotated[list[UploadFile], File(default_factory=list, description="消息附件列表")],
):
    """接收用户消息，并以 SSE 流持续返回 Agent 事件。"""
    workspace = (settings.path.workspace_base_path / session_id).resolve()
    await create_workspace_templates(await ensure_path_async(workspace))

    saved_files = []
    for file in files:
        saved_files.append(await save_to_workspace(file=file, workspace=workspace))

    message = InboundMessage(
        session_id=session_id,
        sender_name=sender_name,
        session_metadata=session_metadata,
        content=content,
        timestamp=timestamp,
        files=saved_files,
    )

    provider = LiteLLMProvider(
        api_key=settings.llm.api_key,
        api_base=settings.llm.api_base,
        default_model=settings.llm.default_model,
        default_context_token=settings.llm.default_context_token,
        default_top_p=settings.llm.default_top_p,
        default_reasoning_effort=settings.llm.default_reasoning_effort,
        support_image=settings.llm.support_image,
        support_video=settings.llm.support_video,
        support_json_output=settings.llm.support_json_output,
        extra_headers={"Authorization": "Bearer " + settings.llm.api_key},
    )
    agent_loop = AgentLoop(
        provider=provider,
        workspace=workspace,
        tools=settings.llm.default_tools,
        max_iterations=settings.llm.max_iterations,
        provider_chat_kwargs=ProviderChatKwargs(reasoning_effort=None),
    )
    await register_default_tools(
        provider=provider,
        workspace=workspace,
        config=settings.tool,
    )
    async for event in agent_loop.run(message=message):
        yield event
