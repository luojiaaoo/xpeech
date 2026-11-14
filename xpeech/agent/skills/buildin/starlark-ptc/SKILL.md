---
name: starlark-ptc
internal: true
description: 为 run_ptc 编写或修复 Starlark PTC 代码，完成多步工具串联、批量调用和文本/JSON 数据整理。在准备生成 PTC 脚本，或遇到 Python 与 Starlark 语法差异、工具返回值处理错误时使用。
---

# Starlark PTC（Programmatic Tool Calling）编写指南

使用 `run_ptc` 将工具调用和数据整理写成一段 Starlark 脚本。按本次工具列表中的名称、参数和实际返回值编排；最后返回任务需要的结果。

## 编排前先取样

**编写依赖工具输出的脚本前，先直接执行本次取数工具，检查真实返回值，再决定如何解析。** 当前任务已有相同输出形式的有效样本时可以复用。

重点检查：

- 数据是字符串、字典、列表，还是多个内容块；需要的数据位于哪个字段。
- 字符串整体是否为 JSON。看起来像 JSON 的正文，前后可能还有状态码、耗时、日志或 Markdown 围栏。
- 是否有行号、分页、截断提示；空结果或错误响应是否采用不同格式。

只有确认整段字符串为合法 JSON 时才直接 `json.decode`。有固定前后缀时，按实测格式分离正文并检查状态；不要靠第一个 `{` 和最后一个 `}` 猜边界。格式尚不清楚时先补充样本，再写依赖该格式的长流程。

对于写入、发送等有外部效果的工具，复用本任务已授权执行的结果或已有预览功能，不为取样重复触发真实操作。

## 决定哪些调用可以写进 PTC

| 工具或参数特征 | 调用方式 |
| --- | --- |
| `ask_user_question` | 在 PTC 外直接调用，取得回答后再继续 |
| `send_file` | PTC 完成文件生成后，在外部直接发送 |
| `shell` | 可能被 DCG 拦截或要求确认；先直接执行预期命令，验证无需确认即可运行后才能加入 PTC |
| 参数中包含 `execution_description` 的其他工具 | 先直接执行预期调用，验证无需确认即可运行后才能加入 PTC |
| `run_ptc` | 不放入脚本，禁止嵌套执行 PTC |

其余工具可通过 `call_tool("工具名", 参数字典)` 调用，参数遵循该工具定义。使用当前工具列表中的准确名称，不根据示例猜测一个工具已经可用。

验证针对本次实际命令和参数，不能用另一条简单命令的成功结果代替。被拒绝或需要用户确认的调用继续放在 PTC 外执行；结果带有 `【用户确认后执行】` 时，表示该次成功依赖用户确认，不能据此认定以后可以直接在 PTC 中运行。

验证本身就是真实执行。已经完成的写入或提交不要再放进脚本重复执行，可以将验证结果作为 `inputs`，从后续步骤继续。后续执行仍可能被拦截，遇到此情况先检查报告，再将受影响的调用移到 PTC 外处理。

## 选择源码或文件

`code` 和 `path` 必须且只能提供一个；`inputs` 默认 `{}`，`timeout_ms` 根据预计总耗时选择。

直接传入源码：

```json
{
  "code": "def main(inputs):\n    return inputs[\"text\"].strip()\n",
  "inputs": {"text": " hello "}
}
```

需要复用时，先用 `write_file` 保存验证后的完整源码，再运行工作区文件：

```json
{
  "path": "scripts/normalize.star",
  "inputs": {"lines": [" banana ", "apple", "banana"]},
  "timeout_ms": 120000
}
```

文件示例对应 [编写模式](references/ptc-patterns.md) 中的文本行清洗脚本。`path` 可以是工作区相对路径或工作区内的绝对路径，不能越界；每次执行读取最新内容。

`timeout_ms` 默认 120,000，最大 240,000，包含文件读取和等待工具返回的时间；工具本身的超时也可能提前结束调用。每次最多调用 32 次工具，较大的批次拆开执行。

## 编写脚本

- 定义 `main(inputs)`，用 `return` 返回字符串、数值、布尔值、`None`、列表或字符串键字典等 JSON 数据。
- 使用 `call_tool` 的返回值继续计算或构造下一次调用的参数；仅对已确认格式的 JSON 文本调用 `json.decode`。
- 不写 Python `import`、`async/await`、`try/except` 或 `print`，也不直接打开文件、发起网络请求或执行命令。
- 额外可调用的方法以当前 `run_ptc` 描述中的名称和签名为准，可直接调用，无需导入。
- 只返回需要的摘要或结构化结果，避免把所有中间正文重复输出。

按需要读取资料，路径均相对本技能目录：

- [语言速查](references/language-guide.md)：表达式、循环、函数、数据结构及 Python 写法迁移。
- [编写模式](references/ptc-patterns.md)：筛选聚合、批量读取、查询后写出，以及 JSON 尾部状态码处理。
- [规范索引](references/spec-index.md)：需要精确语义时定位章节，再用 `read_file` 的 `offset`、`limit` 读取 [规范原文](references/starlark-spec.md) 的对应范围。

## 根据返回结果继续任务

如果返回的是截断预览和完整报告路径，先用 `read_file` 读取完整报告，不直接把预览当作 JSON。参数或文件读取失败时也可能直接返回错误文本，按其中的错误修正调用。

报告中的 `ok=true` 表示脚本完整结束，读取 `result`；`ok=false` 时先检查 `error`，再查看 `tool_calls` 中每次调用的 `name` 和 `status`：

| `status` | 下一步 |
| --- | --- |
| `succeeded` | 此次调用已完成；已成功的写入或提交不要重复执行 |
| `failed` | 查看 `error` 并修复；如果该操作可能已产生部分效果，先查询实际状态 |
| `interaction` | 该调用需要确认或与用户交互，改为在 PTC 外直接调用 |
| `cancelled` | 调用被取消，外部操作是否完成需另行核对，不能据此认定已经撤销 |

脚本中途失败后，后续语句不再执行，已经完成的操作不会撤销。只读处理可在修复后重试；涉及写入、提交时先核对记录和实际结果，再从未完成的步骤继续。恢复示例见 [编写模式](references/ptc-patterns.md)。

## 规范来源

[starlark-spec.md](references/starlark-spec.md) 下载自 [bazelbuild/starlark 官方规范](https://github.com/bazelbuild/starlark/blob/c0af0ea03dc90d3bbc021394b693e6957cc15030/spec.md)，保留原文。中文速查、索引和示例针对当前工具整理；规范中的其他环境扩展不能直接当作本工具已提供的能力。
