# Starlark 语言速查与 PTC 环境差异

本指南按编写 PTC 的决策组织，标准语义来自同目录的 starlark-spec.md；实际可执行能力以 run_ptc 当前运行环境为准。

## 常用语法

| 需求 | 写法或约束 |
| --- | --- |
| 定义函数 | `def helper(value):`，使用缩进；入口为 `def main(inputs):` |
| 条件判断 | `if`、`elif`、`else`；表达式可写 `a if condition else b` |
| 有限循环 | `for row in rows:`、`for i in range(n):`；可用 `break`、`continue` |
| 列表处理 | `[row["name"] for row in rows if row.get("enabled", False)]` |
| 字典缺省值 | `row.get("score", 0)`；需要区分缺失时用 `"score" in row` |
| 字符串处理 | `text.strip()`、`text.split("\n")`、`"\n".join(lines)`、`"数量：{}".format(n)` |
| 常用函数 | `len`、`range`、`enumerate`、`sorted`、`str`、`int` 等；不假定 Python 标准库存在 |
| 空值和布尔值 | 源码中写 `None`、`True`、`False`；工具调用外层 JSON 使用 `null`、`true`、`false` |
| 导入和 I/O | 不支持 Python `import`；当前环境禁用 `load`，文件、网络和命令操作通过工具完成 |
| Python 扩展语法 | 不写类型标注、f-string、`class`、`while`、`try/except`、`with`、`async/await`、`yield` |
| 集合去重 | 使用字典记录已见过的键，不依赖 Python 的 `set` 或集合推导式 |

将处理逻辑、工具调用和要修改的列表、字典放在 `main` 或辅助函数内部；模块级常量仅用于读取。需要重复处理时使用有限循环，不依赖递归。

## 按编写任务查规范

精确行号见 [spec-index.md](spec-index.md)，完整原文见 [starlark-spec.md](starlark-spec.md)。遇到不确定的参数、返回类型或边界条件，读取对应章节，不要套用 Python 的同名接口。

| 当前任务 | 先查的原文章节 | 编码时关注 |
| --- | --- | --- |
| 写字符串与转义 | String literals、String escapes | 源码中的 `\n` 与外层 JSON 字符串转义是两层；序列化源码时不要重复手工转义 |
| 处理列表、元组和字典 | Lists、Tuples、Dictionaries | 可变性、索引越界、字典键要求、插入顺序和方法返回值 |
| 做条件与过滤 | Conditional expressions、Comparisons、Membership tests | 空值、真假值、`in` 检查及比较对象的类型 |
| 批量处理和推导式 | Comprehensions、For statements、Iteration | 输入范围有限；不要在遍历集合时增删同一个集合 |
| 写辅助函数 | Function definitions、Function and method calls | 默认参数、位置参数、关键字参数和局部变量 |
| 遇到全局状态或 Frozen 错误 | Name binding and variables、Freezing a value、Module execution | 模块级对象会冻结，修改对象应在函数中重新创建 |
| 拼接、拆分和提取文本 | string·join、string·split、string·partition、string·rpartition | 分隔符不存在时的返回值，及左侧拆分与右侧拆分的区别 |
| 补齐字典字段或取值 | dict·get、dict·items、dict·keys、dict·update | 缺失键与值为 None 的区别，不要把原地修改方法的返回值当作新字典 |
| 排序、编号和分批 | sorted、enumerate、range、Slice expressions | 参数形式、排序依据、区间终点和切片边界 |
| 明确中止处理 | fail | 条件失败就终止脚本；没有异常捕获来恢复流程 |

## 从 Python 改写时的决策

- 优先使用清晰的 `for` 循环、条件、字典和列表；Python 的内置函数或模块不自动存在。例如求和可以显式累加，不凭印象使用 `sum`、`collections` 或 `itertools`。
- `dict.get(key, default)` 的默认值只用于键不存在的情况；如果键存在但值是 `None`，需要另做判断。不要用 `or` 默认值覆盖合法的 `0` 或 `False`。
- 将 `set` 去重改为“已见键字典 + 结果列表”，按任务决定是否保留原顺序。排序改变顺序，不能在用户要求保序时随手加 `sorted`。
- 字典、列表中的对象可能仍共享引用；需要独立修改时重新构造目标数据，不假定赋值就完成深拷贝。
- 使用字符串的 `.format(...)` 或 `%` 插值，不写 Python f-string 或类型标注。
- 将 `print` 改成返回数据或构造摘要；本工具不提供 `print`。
- 将 Python 的网络、文件和子进程调用改为已有工具。工具本身有独立 schema，不把 Python 库的参数直接套给工具。

## 阅读标准规范时需要调整的写法

规范中的示例可能用于其他应用。编写本工具的脚本时，按下表调整调用方式。

| 能力 | 官方规范涉及的内容 | 当前 PTC 的处理 |
| --- | --- | --- |
| 模块导入 | `load` 语句与宿主模块加载 | 禁用 `load`；不导入 Python 模块 |
| 输出 | 标准文档包含 `print` | 未开放；由 `main(inputs)` 返回结果 |
| 额外函数 | 不同应用可以提供额外类型、函数 | 仅使用当前工具描述列出的方法，通过 `call_tool` 调用其余工具 |
| JSON | 不同应用可以提供数据转换方法 | 可用 `json.decode`、`json.encode`，解析前先实测数据正文格式 |
| 工具调用 | 某些示例中的函数可以直接执行外部操作 | 使用当前工具列表中的名称和参数；需确认的工具在 PTC 外直接调用 |
| 可变数据 | 规范定义冻结与模块执行 | 要修改的列表、字典在函数内部新建，不修改模块级数据 |
| 其他类型和方法 | 规范还描述 bytes、set 等能力 | 返回 JSON 兼容结果，不依赖当前工具未说明的扩展 |

规范中的示例可能依赖其他宿主的模块、全局对象、输出函数或类型；先提取适用的语言规则，再改写到 `main(inputs)`，不要整段照搬。

JSON 正文和状态码一起返回的完整示例见 [ptc-patterns.md](ptc-patterns.md)。
