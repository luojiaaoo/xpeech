# PTC 编写模式与可执行示例

先完成 SKILL.md 中的输出取样与评估，再选择所需示例。代码块均为完整的 code 源码；工具和输入格式须与当前会话的实际样本一致。

## 筛选、去重、排序与聚合

下面示例的 `inputs` 为 `{"rows":[{"name":" banana ","enabled":true,"quantity":2},{"name":"apple","enabled":true,"quantity":3}]}`。

```starlark
def main(inputs):
    seen = {}
    names = []
    total = 0
    for row in inputs["rows"]:
        if not row.get("enabled", False):
            continue
        total += row.get("quantity", 0)
        name = row["name"].strip()
        if name and name not in seen:
            seen[name] = True
            names.append(name)
    return {
        "names": sorted(names),
        "quantity": total,
        "summary": "共 {} 个名称，数量合计 {}".format(len(names), total),
    }
```

## 有限批量读取

`inputs` 中传入 `paths` 文件路径列表。此例使用已注册的 `read_file(path, limit)`，返回精简预览，不把整批正文重复输出。读取工具本身可能包含行号、提示或截断信息，不能假定结果就是文件的原始字节。

```starlark
def main(inputs):
    previews = []
    for path in inputs["paths"]:
        text = call_tool("read_file", {"path": path, "limit": 20})
        previews.append({"path": path, "preview": text[:200]})
    return {"count": len(previews), "files": previews}
```

循环中的调用按顺序执行，当前接口不提供并行或后台任务原语。批量输入过多时分批执行，避免耗尽总时间或工具调用预算。

## 解析查询结果后调用另一个工具

`call_tool` 原样返回文本或 JSON 数据，不能把所有字符串都当作 JSON。先按 SKILL.md 的要求实测输出，只有确认整段字符串是合法 JSON 时才直接使用 `json.decode`；已有列表、字典无需再次解码。纯文本直接用字符串方法处理。

下面示例适用于实测确认返回值整体为 `{"items":[{"name":"A","score":0.9}]}` 这类 JSON 文本的查询工具。`inputs` 提供其准确的 `query_tool`、符合该工具定义的 `query_args`，以及用户指定的 `output_path`；确认本次任务需要写入该报告后再执行。

```starlark
def main(inputs):
    raw = call_tool(inputs["query_tool"], inputs["query_args"])
    data = json.decode(raw)
    selected = []
    for item in data["items"]:
        if item.get("score", 0) >= inputs.get("min_score", 0.8):
            selected.append({"name": item["name"], "score": item["score"]})
    report = {"count": len(selected), "items": selected}
    call_tool("write_file", {
        "path": inputs["output_path"],
        "content": json.encode(report),
    })
    return report
```

## JSON 正文后附带状态信息

例如实测得到下面的文本，它整体不是 JSON：

```text
{"items":[{"name":"A","score":0.9}]}

HTTP_STATUS: 200
```

只有当前工具的实际输出确实采用这个固定尾部约定时，才使用下面的拆分方式。其他工具应根据各自实测格式处理，不要照搬分隔标记：

```starlark
def main(inputs):
    raw = call_tool(inputs["query_tool"], inputs["query_args"])
    body, marker, status = raw.rpartition("\n\nHTTP_STATUS: ")
    if not marker:
        fail("工具返回格式变化：没有找到预期状态标记")
    if status.strip() != "200":
        fail("查询未成功，状态为：" + status.strip())
    data = json.decode(body)
    return {"count": len(data["items"]), "items": data["items"]}
```

## 使用已注入的纯计算方法

以当前 run_ptc 工具描述为准。目前的文本行清洗方法可这样使用，`inputs` 传入 `lines` 字符串列表：

```starlark
def main(inputs):
    lines = ptc_normalize_lines(inputs["lines"])
    return {"lines": lines, "text": "\n".join(lines)}
```

## 中途失败后从哪里继续

假设报告包含下面的记录，表示文件已写入，后续网页读取失败：

```json
{
  "ok": false,
  "error": "PTC tool web_fetch failed: request timed out",
  "tool_calls": [
    {"name": "write_file", "status": "succeeded", "succeeded": true},
    {"name": "web_fetch", "status": "failed", "succeeded": false}
  ]
}
```

先确认已写出的文件符合预期，随后只重试网页读取及依赖它的后续处理，不重新执行写文件的前半段。
如果记录为 `interaction`，将该次调用移到 PTC 外直接完成；如果记录为 `cancelled`，先查询实际结果再决定是否重试。
报告已经被截断时，先读取返回路径中的完整报告，不能根据预览推测哪些操作成功。

