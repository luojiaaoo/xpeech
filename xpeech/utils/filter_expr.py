"""受限的资源过滤表达式（filter）解析与求值。

语法是 Python 布尔表达式的极小子集::

    expr   := or_expr
    or     := and ("or" and)*
    and    := not_ ("and" not_)*
    not_   := "not" not_ | primary
    primary := "(" or ")" | call | "True" | "False"
    call   := ("channel" | "org") "(" STRING ("," STRING)* ")"

内置短名函数（参数为非空字符串字面量）：

- ``channel("crm", ...)``：当前 JWT 通道名（kid）等于任一参数时为真。
- ``org("公司/研发/", ...)``：用户组织字符串以任一参数为前缀时为真。

未配置过滤器的资源公开；已配置的资源仅在表达式求值为真时可见。

安全模型：表达式经 ``ast.parse`` 解析后按严格节点白名单校验，再由本模块
直接解释执行，全程不调用 eval/exec/compile，不接触 Python 对象模型；
表达式长度、嵌套深度和参数个数均有上限。
"""

import ast
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

FILTER_MAX_LENGTH = 256
FILTER_MAX_DEPTH = 8
FILTER_MAX_ARGS = 32


class FilterExpressionError(ValueError):
    """过滤表达式语法或结构不合法。"""


@dataclass(frozen=True)
class FilterContext:
    """过滤表达式可见的上下文；None 表示未知，任何匹配函数都视为不命中。"""

    channel: str | None = None
    org: str | None = None


def _match_channel(context: FilterContext, args: tuple[str, ...]) -> bool:
    return context.channel is not None and context.channel in args


def _match_org(context: FilterContext, args: tuple[str, ...]) -> bool:
    return context.org is not None and context.org.startswith(args)


FILTER_FUNCTIONS: dict[str, Callable[[FilterContext, tuple[str, ...]], bool]] = {
    "channel": _match_channel,
    "org": _match_org,
}


def _check_arg(node: ast.expr, call_name: str) -> str:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        raise FilterExpressionError(f"{call_name}() arguments must be string literals")
    value = node.value
    if not value or value != value.strip():
        raise FilterExpressionError(f"{call_name}() arguments must be non-empty strings without surrounding whitespace")
    return value


def _check(node: ast.AST, depth: int) -> None:
    if depth > FILTER_MAX_DEPTH:
        raise FilterExpressionError(f"filter nesting is limited to {FILTER_MAX_DEPTH} levels")
    if isinstance(node, ast.Expression):
        _check(node.body, depth)
    elif isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)) or not node.values:
            raise FilterExpressionError("filter only supports 'and' / 'or'")
        for value in node.values:
            _check(value, depth + 1)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.Not):
            raise FilterExpressionError("filter only supports 'not'")
        _check(node.operand, depth + 1)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, bool):
            raise FilterExpressionError("filter only accepts True / False constants")
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FILTER_FUNCTIONS:
            raise FilterExpressionError("filter only calls built-in functions: " + ", ".join(sorted(FILTER_FUNCTIONS)))
        if node.keywords:
            raise FilterExpressionError(f"{node.func.id}() does not accept keyword arguments")
        if not node.args or len(node.args) > FILTER_MAX_ARGS:
            raise FilterExpressionError(f"{node.func.id}() requires 1-{FILTER_MAX_ARGS} arguments")
        for arg in node.args:
            _check_arg(arg, node.func.id)
    else:
        raise FilterExpressionError(f"filter does not allow {type(node).__name__}")


def _evaluate(node: ast.AST, context: FilterContext) -> bool:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp):
        return not _evaluate(node.operand, context)
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_evaluate(value, context) for value in node.values)
        return any(_evaluate(value, context) for value in node.values)
    call = node
    args = tuple(arg.value for arg in call.args)  # type: ignore[union-attr]
    return FILTER_FUNCTIONS[call.func.id](context, args)  # type: ignore[union-attr]


@dataclass(frozen=True)
class ParsedFilter:
    """A validated filter expression ready for repeated evaluation."""

    source: str
    _tree: ast.Expression = field(repr=False, compare=False)

    def evaluate(self, context: FilterContext) -> bool:
        return _evaluate(self._tree.body, context)


@lru_cache(maxsize=256)
def parse_filter(source: str) -> ParsedFilter:
    """Validate a filter expression and cache its AST for evaluation."""

    if not isinstance(source, str) or not source.strip():
        raise FilterExpressionError("filter must be a non-empty expression string")
    if len(source) > FILTER_MAX_LENGTH:
        raise FilterExpressionError(f"filter is limited to {FILTER_MAX_LENGTH} characters")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise FilterExpressionError(f"filter syntax error: {exc.msg}") from exc
    _check(tree, 0)
    return ParsedFilter(source=source, _tree=tree)


def validate_filter_source(source: str | None) -> str | None:
    """Return a validated filter source unchanged; None means a public resource."""

    if source is None:
        return None
    return parse_filter(source).source


def filter_allows_access(source: str | None, context: FilterContext) -> bool:
    """Public without a filter; otherwise the expression must evaluate to True."""

    if source is None:
        return True
    return parse_filter(source).evaluate(context)
