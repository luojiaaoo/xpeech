"""工具返回的交互请求：真实效果由 Agent 主循环在工具返回后处理。"""

from dataclasses import dataclass


class InteractionRequest:
    """需要 Agent 主循环介入的工具返回值基类。

    工具函数只负责构造并返回请求对象，真正的交互效果（询问用户、发送文件、
    确认后执行）由 Agent 主循环识别返回值后完成；无法交互的执行环境
    应直接拒绝这类返回值。
    """


@dataclass(frozen=True)
class ExecutionRequest(InteractionRequest):
    """请求用户确认后执行工具原函数（requires_confirmation 流程）。"""


@dataclass(frozen=True)
class SendFileRequest(InteractionRequest):
    """请求把文件发送给用户。"""

    source: str


@dataclass(frozen=True)
class QuestionRequest(InteractionRequest):
    """请求向用户展示表单并等待回答，question 为完整表单 JSON 字符串。"""

    question: str
