from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from ..tool_executor import ToolExecutor


class PtcArgs(BaseModel):
    """PTC 执行参数，源码模式与文件模式二选一。"""

    code: str | None = Field(
        default=None,
        description="源码模式：要直接执行的 PTC 代码。与 file_path 二选一，必须恰好提供一个。",
    )
    file_path: str | None = Field(
        default=None,
        description="文件模式：要执行的 PTC 代码文件路径，先读取文件内容，再按源码模式执行。与 code 二选一，必须恰好提供一个。",
    )

    @model_validator(mode="after")
    def check_exclusive_source(self) -> "PtcArgs":
        if (self.code is None) == (self.file_path is None):
            raise ValueError("必须且只能提供 code 或 file_path 之一")
        return self


def build_ptc_tool(workspace: Path):
    base = workspace.resolve()
    if not base.exists():
        raise ValueError(f"Invalid workspace: {workspace}")

    async def run_ptc(args: PtcArgs, tool_executor: ToolExecutor) -> str:
        """执行 PTC 代码，把工具 / MCP 调用通过 Python 串联起来执行。

        源码模式（code）直接执行 PTC 代码；文件模式（file_path）先读取对应文件内容，再按源码模式执行。
        执行时可通过注入的 tool_executor 串联调用工具 / MCP。
        """
        if args.file_path is not None:
            # TODO: 读取 file_path 对应文件内容，转入源码模式执行
            raise NotImplementedError("PTC 文件模式尚未实现")
        # TODO: 在受控环境中执行 PTC 代码，通过 tool_executor 串联调用工具 / MCP
        raise NotImplementedError("PTC 源码模式尚未实现")

    return run_ptc
