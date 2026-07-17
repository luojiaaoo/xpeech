from .server import app
from fastapi import Depends
from typing import Annotated
from datetime import datetime
from fastapi import File, Form, UploadFile, HTTPException, status, Header, Query
from fastapi.responses import FileResponse
from .schema import InputContent, InboundMessage
import json
from pathlib import Path
from uuid import UUID
from ...config.settings import settings
from ...utils.helper import save_to_workspace, ensure_path
from ...utils.session import create_workspace_templates
from ...provider.litellm_provider import LiteLLMProvider
from ..loop import AgentLoop, QuestionEvent
from ...provider.schema import ProviderChatKwargs
from fastapi.sse import EventSourceResponse
from ..tools.helper import safe_resolve_workspace_path


def session_metadata(
    session_metadata: Annotated[
        str,
        Form(
            description='会话的元数据，JSON 格式字符串，如：{"sender_id": "xxxx","channel": "feishu"}',
        ),
    ] = "{}",
):
    """将会话元数据的 JSON 字符串解析为字典。"""
    try:
        parsed_dict = json.loads(session_metadata)
        return parsed_dict
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid JSON format")


def content(
    content: Annotated[
        str,
        Form(
            description='消息内容列表，JSON 格式字符串。[{"text": "你好"}, {"text": "world"}]',
        ),
    ],
):
    """将消息内容列表的 JSON 字符串解析为 InputContent 列表。"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid JSON format")


@app.get("/sessions/{session_id}/files")
async def download_session_file(
    session_id: str,
    path: Annotated[str, Query(description="File path returned by a send_file event.")],
):
    workspace = (settings.path.workspace_base_path / session_id).resolve()
    try:
        file_path = safe_resolve_workspace_path(path, workspace, True)
    except PermissionError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="File is outside the workspace or Built-in skills directory")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    return FileResponse(file_path, filename=file_path.name)


@app.get(
    f"{settings.tool.browser_preview.route_path}/{{preview_id}}/{{file_path:path}}",
    include_in_schema=False,
)
async def preview_file(preview_id: UUID, file_path: str):
    relative_path = Path(file_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid preview path")
    preview_file_path = settings.tool.browser_preview.browser_preview_path / str(preview_id) / relative_path
    if not preview_file_path.exists() or not preview_file_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview file not found")
    return FileResponse(preview_file_path, headers={"Cache-Control": "no-store"})


@app.post("/answer_question")
async def answer_question(
    session_id: Annotated[str, Header(description="会话的ID", alias="x-session-id")],
    answer: Annotated[str, Form(description="回答内容")],
):
    question_event: QuestionEvent = AgentLoop.SESSION_QUESTION_EVENT.get(session_id)
    if question_event is not None:
        question_event.answer = answer
        question_event.event.set()
        return {"message": "Answer received"}
    else:
        return {"message": "Question event not found"}


@app.post("/chat", response_class=EventSourceResponse)
async def chat(
    session_id: Annotated[str, Header(description="会话的ID", alias="x-session-id")],
    session_metadata: Annotated[dict[str, str], Depends(session_metadata)],
    content: Annotated[InputContent, Depends(content)],
    timestamp: Annotated[datetime, Form(default_factory=datetime.now, description="消息的时间戳")],
    files: Annotated[list[UploadFile], File(default_factory=list, description="消息附件列表")],
):
    """Receive a message and return a response."""

    workspace = (settings.path.workspace_base_path / session_id).resolve()

    # 创建工作目录
    if not workspace.exists():
        await create_workspace_templates(ensure_path(workspace))

    # 把files都保存到工作目录
    files_ = []
    for file in files:
        file_path = await save_to_workspace(file=file, workspace=workspace)
        files_.append(file_path)

    # 创建消息对象
    message = InboundMessage(
        session_id=session_id,
        session_metadata=session_metadata,
        content=content,
        timestamp=timestamp,
        files=files_,
    )

    # 开启Agent Loop
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
    al = AgentLoop(
        provider=provider,
        workspace=workspace,
        tools=settings.llm.default_tools,
        max_iterations=settings.llm.max_iterations,
        provider_chat_kwargs=ProviderChatKwargs(
            reasoning_effort=None,
        ),
    )
    # 注册默认工具
    await al.register_default_tools()
    # 运行Agent Loop
    async for i in al.run(message=message):
        yield i
