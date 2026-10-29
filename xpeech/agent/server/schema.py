from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator


# ------------------ 输入内容块 ------------------
class InputText(BaseModel):
    """文本输入内容块。"""

    text: Annotated[str, Field(description="发送给模型的文本内容")]


class InputContent(RootModel[list[InputText]]):
    """输入内容块。"""


class InboundMessage(BaseModel):
    """请求的消息的 schema。"""

    user_id: Annotated[str, Field(description="用户 ID")]
    session_id: Annotated[str, Field(description="会话 ID")]
    sender_name: Annotated[str, Field(description="发送者用户名")]
    session_metadata: Annotated[dict[str, str], Field(description="会话元数据")]
    content: Annotated[list[InputText], Field(description="消息内容")]
    timestamp: Annotated[datetime, Field(description="消息时间戳")]
    files: Annotated[list[Path], Field(description="消息附件")]
    json_output: bool = Field(
        default=False,
        description='是否为大模型请求添加 response_format={"type":"json_object"}，不修改提示词；模型不支持时返回 HTTP 400',
    )


# ------------------ 定时任务输出消息 ------------------
class BackgroundMessageChannel(StrEnum):
    """后台消息支持的对外渠道。"""

    FEISHU = "feishu"


class FeishuBackgroundMessage(BaseModel):
    """等待飞书渠道投递的 Agent 输出。"""

    model_config = ConfigDict(frozen=True)

    channel: Literal[BackgroundMessageChannel.FEISHU] = BackgroundMessageChannel.FEISHU
    open_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    schedule_type: Literal["run_at", "cron"]
    schedule_value: str

    @field_validator("open_id", "content")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value
