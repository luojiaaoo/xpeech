"""集中登记额外注入到 Starlark PTC 的纯数据处理方法。

所有注入函数及其登记名称统一使用 ptc_ 前缀，脚本可以直接按登记名称调用。
这里是受信任的宿主 Python 代码：只写同步的纯计算函数，不读写文件、访问网络、
执行命令或发起交互；有外部效果的操作必须注册为工具，通过 call_tool 执行。
运行时仅向这些方法传递 JSON 数据副本，不注入执行器、会话或其他宿主对象。

新增方法的步骤：
1. 定义以 ptc_ 开头的同步函数，为参数、返回值写清类型，并用中文 docstring 解释输入与输出。
2. 将同样以 ptc_ 开头的脚本调用名称和函数登记到下方 INJECTED_METHODS 字典中。
3. 重启服务并重新注册工具；模型会在工具描述里看到方法的名称、签名和说明。

只有显式登记的方法会注入。返回值应为字符串、数值、布尔值、None、列表或字典等
JSON 数据，不应返回宿主对象、协程或 InteractionRequest。不要依赖跨调用的可变
全局状态；方法在解释器工作线程中运行，也不能假设自己处于 asyncio 事件循环中。
"""

from collections.abc import Callable
from typing import Any


def ptc_normalize_lines(lines: list[str]) -> list[str]:
    """清洗文本行列表，返回去重后按字符串顺序排序的新列表。

    lines 中的每一项必须是字符串。处理时先去除首尾空白、丢弃空行，
    再去重排序；不会修改传入列表，也不保留原始行顺序。
    例如 [" banana ", "apple", "banana", " "] 返回 ["apple", "banana"]。
    """
    if not isinstance(lines, list) or any(not isinstance(line, str) for line in lines):
        raise TypeError("ptc_normalize_lines expects a list of strings")
    return sorted({line.strip() for line in lines if line.strip()})


# 键是 Starlark 中直接调用的名称，值是宿主 Python 函数；函数不会自动获得工具权限。
# 注入名称统一使用 ptc_ 前缀，与 Starlark 标准函数和工具桥接入口区分；无需维护冲突名单。
INJECTED_METHODS: dict[str, Callable[..., Any]] = {
    "ptc_normalize_lines": ptc_normalize_lines,
}
