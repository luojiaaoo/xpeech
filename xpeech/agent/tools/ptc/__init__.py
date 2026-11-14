"""Starlark PTC 的公共导出入口。

工具注册模块通过 build_ptc_tool 构造工具，参数模型由运行器内部维护。
执行与线程桥接位于 ptc.py，额外纯计算方法在 patch.py 中显式登记。
PTC 始终作为内置工具注册，不需要配置开关。
"""

from .ptc import build_ptc_tool

__all__ = ["build_ptc_tool"]
