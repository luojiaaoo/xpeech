from typing import Annotated
from pydantic import BaseModel, Field, RootModel
from datetime import datetime
from pathlib import Path


# ------------------ 输入内容块 ------------------
class InputText(BaseModel):
    """文本输入内容块。"""

    text: Annotated[str, Field(description="发送给模型的文本内容")]


class InputContent(RootModel[list[InputText]]):
    """输入内容块。"""

    pass


class InboundMessage(BaseModel):
    """请求的消息的 schema。"""

    session_id: Annotated[str, Field(description="会话ID")]
    session_metadata: Annotated[dict[str, str], Field(description="会话元数据")]
    content: Annotated[list[InputText], Field(description="消息内容")]
    timestamp: Annotated[datetime, Field(description="消息时间戳")]
    files: Annotated[list[Path], Field(description="消息附件")]
