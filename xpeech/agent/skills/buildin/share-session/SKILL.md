---
name: share-session
short_description: 当前会话分享
description: 当用户要求分享当前会话、导出聊天记录或把本次对话生成分享链接时使用。先确认详细分享、对话分享或摘要分享，再生成 HTML 并通过浏览器预览工具返回超链接。
---

# 分享当前会话

## 1. 先选择分享级别

用户尚未明确本次分享级别时，先调用 `ask_user_question`，将下面的表单序列化为 JSON 字符串作为唯一的 `question` 参数，等待回答后再生成分享内容：

```json
{
  "type": "form",
  "title": "分享当前会话",
  "subtitle": "请选择分享内容的详细程度",
  "submit_label": "生成分享链接",
  "fields": [
    {
      "type": "select",
      "name": "share_level",
      "label": "用什么级别分享？",
      "placeholder": "请选择分享级别",
      "options": [
        {"label": "详细分享（包括工具内容）", "value": "detailed"},
        {"label": "对话分享（只包括用户和回复，不包括工具）", "value": "conversation"},
        {"label": "摘要分享（不包括详细的对话内容）", "value": "summary"}
      ]
    }
  ]
}
```

用户已明确级别时直接沿用；未回答、取消或超时不代表选择。没有交互工具时以普通回复列出同样的三个选项并等待下一条用户消息，不自行默认级别。

## 2. 整理当前会话

仅根据当前会话上下文整理，以用户发起分享请求之前的内容为截止点，不包含本次选择级别、读取技能、生成 HTML 和发布预览的过程。

| 级别 | 分享内容 |
| --- | --- |
| `detailed` | 按原顺序保留用户消息、助手回复、工具名称、调用参数和返回结果。工具内容在页面中折叠展示。 |
| `conversation` | 按原顺序保留用户消息和助手面向用户的回复，排除工具调用、参数、返回结果及工具执行过程消息。回复原本引用的结论可以保留，不额外补入工具日志。 |
| `summary` | 提炼讨论主题、用户目标、关键结论、已完成事项及未解决问题，不附逐条对话、原文摘录或工具日志。 |

不包含系统/开发者提示词、隐藏推理、运行时元数据或凭据。内容中的密钥、令牌和密码用 `[已隐藏]` 替代。附件只描述已知的文件名和用途，不复制整个工作区或把本地路径伪装成可访问链接。

`list_sessions` 和 `read_session` 只用于其他历史会话，不能用来读取当前会话，也不要绕过工具去读服务端历史文件。若早期原文或工具结果已被压缩，不能从摘要编造原文；在 `scope_note` 中注明缺失范围，只分享实际可见的内容。摘要中只能使用有依据的事实。

## 3. 渲染 HTML

使用 [scripts/render_share.py](scripts/render_share.py)，仅依赖 Python 标准库。先用 `write_file` 在工作区新建独立目录（如 `session-shares/<本次唯一标识>/`），写入 UTF-8 JSON 数据：

```json
{
  "title": "本次讨论的主题",
  "level": "conversation",
  "scope_note": "根据当前可见会话整理，截止到本次分享请求之前。",
  "messages": [
    {"role": "user", "content": "用户原文"},
    {"role": "assistant", "content": "助手回复原文"}
  ]
}
```

`messages[].content` 必须是字符串，仅放对应的可分享正文，保留换行和代码。工具调用使用助手消息的 `tool_calls` 数组，结构为 `{"function": {"name": "工具名", "arguments": "参数 JSON 字符串"}}`；工具返回使用 `{"role": "tool", "name": "工具名", "content": "返回内容"}`，保留原来的先后顺序。

摘要级别只需提供 `title`、`level: "summary"`、`scope_note` 和 `summary`（归纳文本字符串），不写入 `messages`。另外两个级别不写 `summary`。渲染器会按级别过滤字段，排除的内容不得通过标题、范围说明、隐藏 DOM 或脚本再次嵌入页面。

调用 Shell 工具执行（脚本路径相对于本技能目录，输入输出路径相对于工作区）：

```bash
python skills/share-session/scripts/render_share.py --input session-shares/<本次唯一标识>/content.json --output session-shares/<本次唯一标识>/index.html
```

脚本会生成完全自包含的单文件 HTML，包含标题、分享级别、范围说明和正文，支持窄屏阅读与长文本换行。所有样式必须内联；不得引用外部或相邻文件中的 CSS、JavaScript、字体、图片等资源。确需嵌入图片时使用 data URI，不使用本地文件路径。所有输入均以文本转义，不执行会话中的 HTML 或脚本；不依赖 CDN。预览工具只复制这一个 HTML 文件，不会携带其他依赖。

## 4. 生成分享超链接

调用 `create_browser_preview`，`path` 只传入本次生成的 `index.html` 文件，避免把中间 JSON 或其他工作区文件复制到公开预览目录。无需启动 HTTP 服务。

成功后以 Markdown 超链接交付：`[查看会话分享（对话分享）](工具实际返回的完整 URL)`，级别名称按实际选择替换。有缺失范围时用一句话说明。工具失败时如实说明原因，不伪造 URL，也不声称分享已成功。
