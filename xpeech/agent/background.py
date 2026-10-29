"""Execute scheduled Agent runs inside the backend."""

from datetime import datetime
from uuid import uuid4

from .server.schema import InboundMessage, InputText
from .tools.schedule.schema import ScheduledExecution, ScheduledExecutionResult

SCHEDULED_TASK_PROMPT_PREFIX = (
    "## 定时任务运行说明\n\n"
    "本次运行由后台定时任务触发，不是用户实时发起的新对话。\n"
    "请完整执行下方的原始定时任务。完成任务后、给出最终回复前，"
    "请将本次定时任务的执行记录追加到 `memory/HISTORY.md`，"
    "不要覆盖已有内容。记录必须以 `[YYYY-MM-DD HH:MM]` 开头，并包含任务内容和执行结果摘要。\n\n"
    "最终回复会直接发送给用户，必须呈现原始任务要求的实际结果。"
    "不要提及后台调度、`memory/HISTORY.md`、"
    "执行记录或追加动作，更不能仅回复“执行记录已追加”等操作确认。\n\n"
    "## 原始定时任务提示词\n"
)


async def execute_scheduled_task(user_id: str, request: ScheduledExecution) -> ScheduledExecutionResult:
    """Run a background Agent with isolated history and no interactive tools."""
    from .runner import AgentRunner

    message = InboundMessage(
        user_id=user_id,
        session_id=uuid4().hex,
        sender_name=request.sender_name,
        session_metadata={**request.session_metadata, "source": "scheduled_task"},
        content=[
            InputText(
                text=(
                    f"{SCHEDULED_TASK_PROMPT_PREFIX}"
                    f"调度类型：{request.schedule_type}\n"
                    f"调度表达式：{request.schedule_value}\n\n"
                    f"{request.prompt}"
                ),
            ),
        ],
        timestamp=datetime.now().astimezone(),
        files=[],
    )
    runner = await AgentRunner.create(user_id)
    content = await runner.run_once(message, background=True)
    return ScheduledExecutionResult(content=content)
