# Xpeech 后端接入说明

本文面向需要将自有业务后端、机器人通道或其他服务接入 Xpeech Agent 的开发者，说明鉴权、聊天请求、SSE 事件、追问回调和文件下载的完整流程。

## 1. 接入概览

Xpeech API 默认监听 `http://localhost:7878`。一次完整的聊天交互如下：

1. 接入方后端使用双方共享的 JWT 密钥签发短期 Bearer Token。
2. 接入方以表单请求调用 `POST /chat`；携带附件时使用 `multipart/form-data`。
3. Xpeech 通过 SSE 持续返回思考、正文、工具调用、文件和追问等事件。
4. 收到 `question` 事件时，接入方展示表单，并通过 `POST /answer_question` 提交答案；原 `/chat` SSE 连接继续保持。
5. 收到 `send_file` 事件时，接入方通过 `GET /sessions/{user_id}/files` 下载文件。
6. SSE 连接结束，代表本轮 Agent 运行完成。

> Web 客户端的 `/api/chat`、`/api/answer_question` 和 `/api/files` 是基于登录 Cookie 的代理接口。服务到服务接入应直接调用本文介绍的 Xpeech API。

## 2. 基础约定

| 项目 | 约定 |
| --- | --- |
| API Base URL | 默认 `http://localhost:7878` |
| 健康检查 | `GET /health`，正常返回 `200 ok` |
| 鉴权 | `Authorization: Bearer <JWT>` |
| 请求格式 | 表单编码；携带附件时为 `multipart/form-data` |
| 流式响应 | `text/event-stream` |
| 字符编码 | UTF-8 |
| OpenAPI | `/openapi.json`、`/docs`、`/redoc` |

生产环境必须使用 HTTPS，并限制 API 只能被可信服务访问。

Compose 部署默认由 Nginx 对外提供该 Base URL，并按 `x-user-id` 做一致性哈希，
将同一用户的请求稳定转发到同一个 backend 副本。这是因为 `/chat` 的运行状态和
`/answer_question` 的等待事件位于 backend 进程内存中。接入方必须在聊天、追问回答和文件请求中
使用完全相同的 `x-user-id`；扩缩容或替换 backend 容器前，应等待正在进行的 SSE 会话结束。

## 3. JWT 鉴权

除健康检查、API 文档和浏览器预览外，本文涉及的 API 都需要短期 JWT。JWT 使用 `conf.toml` 中的配置签名：

```toml
[jwt]
secret_key = "replace-with-a-random-secret-of-at-least-32-characters"
algorithm = "HS256"
access_token_expire_seconds = 60
```

Token 至少需要包含 `exp`，签名算法必须与配置一致。建议接入方在每次请求前签发新 Token，不要缓存到过期边界。

Python 签发示例：

```python
from datetime import UTC, datetime, timedelta

import jwt


def create_xpeech_token(secret: str) -> str:
    return jwt.encode(
        {"exp": datetime.now(UTC) + timedelta(seconds=60)},
        secret,
        algorithm="HS256",
    )
```

如果本次 Agent 请求需要调用飞书用户身份命令，可以在 JWT 中额外放入字符串类型的 `lark_user_access_token`：

```python
claims = {
    "exp": datetime.now(UTC) + timedelta(seconds=60),
    "lark_user_access_token": "<feishu-user-access-token>",
}
```

该字段只在当前请求上下文中使用，不会保存到会话历史。后续请求需要重新携带。
携带非空字段时，沙盒加载 `lark-skills-token`，使用 JWT 中的用户令牌，
且 `lark-cli` 不会回退请求 `lark-token-manager`；未携带时加载 `lark-skills` 和 `lark-cli-auth`，使用令牌容器。

`POST /token` 主要供 Swagger UI 调试：表单中的 `password` 为共享 JWT 密钥，`username` 可任意填写。业务系统应自行签发 JWT，不应在浏览器、移动端或其他不可信客户端中保存或传输共享密钥。

## 4. 会话和用户标识

聊天接口使用以下请求头：

| 请求头 | 必填 | 说明 |
| --- | --- | --- |
| `Authorization` | 是 | 短期 Bearer JWT |
| `x-user-id` | 是 | 业务侧稳定用户 ID，用于隔离工作区、技能和 MCP 连接 |
| `x-session-id` | 否 | 会话 ID，默认 `default`，用于隔离上下文和并发请求 |
| `x-sender-name` | `/chat` 必填 | 展示用用户名；非 ASCII 字符必须做 UTF-8 URL 编码 |

`user_id` 和 `session_id` 长度为 1～128，只允许 ASCII 字母、数字、空格、连字符 `-` 和下划线 `_`。

同一用户可同时使用多个不同的 `session_id`。会话历史按 `user_id + session_id` 隔离，但同一用户的文件工作区是共享的。只有 `default` 会话会加载后台任务工具。

同一个 `user_id + session_id` 同时只能存在一个 `/chat` 请求；重复请求会返回 `409 Conflict`。接入方应在本地按会话串行发送消息，或为相互独立的聊天分配不同 `session_id`。

## 5. 发送聊天消息

### 5.1 接口

```http
POST /chat
Content-Type: multipart/form-data
Accept: text/event-stream
```

表单字段：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `content` | JSON 字符串 | 是 | - | 文本块数组，例如 `[{"text":"你好"}]` |
| `session_metadata` | JSON 字符串 | 否 | `{}` | 会话元数据对象，值建议统一使用字符串 |
| `timestamp` | ISO 8601 日期时间 | 否 | 服务端当前时间 | 原始消息时间，例如 `2026-10-28T10:30:00+08:00` |
| `files` | 文件，可重复 | 否 | 空 | 消息附件列表 |
| `json_output` | 布尔值 | 否 | `false` | 请求模型输出 JSON 对象；服务端必须启用相应能力 |

`content` 与 `session_metadata` 虽然是 JSON 数据，但它们是表单中的字符串字段，不能把整个请求直接发送为 `application/json`。没有附件时可以使用 `application/x-www-form-urlencoded`；上传附件时必须使用 `multipart/form-data`。

最小请求：

```bash
curl -N -X POST "http://localhost:7878/chat" \
  -H "Authorization: Bearer <JWT>" \
  -H "x-user-id: demo-user" \
  -H "x-session-id: chat-001" \
  -H "x-sender-name: demo-user" \
  -H "Accept: text/event-stream" \
  -F 'session_metadata={"channel":"my_backend"}' \
  -F 'content=[{"text":"你好，请介绍一下自己"}]'
```

上传多个附件时重复传递 `files`：

```bash
curl -N -X POST "http://localhost:7878/chat" \
  -H "Authorization: Bearer <JWT>" \
  -H "x-user-id: demo-user" \
  -H "x-session-id: chat-001" \
  -H "x-sender-name: demo-user" \
  -F 'content=[{"text":"请分析这些附件"}]' \
  -F 'files=@report.pdf' \
  -F 'files=@chart.png'
```

用户名包含中文时，请先编码请求头。例如 `张三` 应发送为：

```http
x-sender-name: %E5%BC%A0%E4%B8%89
```

### 5.2 JSON 输出模式

传入 `json_output=true` 后，Xpeech 会为模型请求添加 `response_format={"type":"json_object"}`，但不会自动修改提示词。接入方仍应在 `content` 中明确说明目标 JSON 结构。

该模式要求服务端配置：

```toml
[llm]
support_json_output = true
```

如果服务端未启用而请求了该模式，接口会在 SSE 开始前返回 `400`。成功响应仍然是 SSE，需要拼接所有 `assistant` 分片，再对完整字符串执行一次 JSON 解析。

## 6. 解析 SSE 响应

每个 SSE 数据块的 `data` 都是一个 JSON 对象：

```text
data: {"event":"assistant","context":"你"}

data: {"event":"assistant","context":"好"}

data: {"event":"assistant_end","context":""}

data: {"event":"token_usage","context":"{\"上下文使用率\":\"1.23% (1k / 100k)\",\"会话时长\":\"2秒\",\"大模型请求次数\":\"1\"}"}

```

`context` 始终是字符串；部分事件的 `context` 本身又是 JSON 字符串，需要按事件类型进行第二次解析。

| `event` | `context` | 接入方建议 |
| --- | --- | --- |
| `assistant` | 助手正文分片 | 按顺序追加到当前正文段 |
| `assistant_end` | 空字符串 | 结束当前正文段；一次请求可能有多个正文段 |
| `thinking` | 思考内容分片 | 可增量展示，也可隐藏 |
| `thinking_end` | 空字符串 | 结束当前思考段 |
| `tool_call` | JSON 字符串 | 工具调用列表，可作为状态信息展示 |
| `tool_call_result` | JSON 字符串 | 工具执行结果列表，可作为状态信息展示 |
| `command` | 文本 | 展示 `/new`、`/clear` 等命令执行结果 |
| `question` | 问题表单 JSON 字符串 | 展示表单并调用 `/answer_question` |
| `send_file` | 工作区相对路径 | 调用文件下载接口 |
| `token_usage` | 指标 JSON 字符串 | 展示上下文使用率、耗时和模型请求次数 |

不能把一次 SSE data 块等同于一条完整回复。最终正文由所有 `assistant` 分片按到达顺序拼接；工具调用前后可能出现多个正文段。以 SSE 流 EOF 作为整轮完成信号，并在 EOF 时兜底结束尚未收到 `*_end` 的展示段。

Python 异步接入示例：

```python
import json
from collections.abc import AsyncIterator
from urllib.parse import quote

import httpx
from httpx_sse import aconnect_sse


async def chat_events(
    *,
    base_url: str,
    token: str,
    user_id: str,
    session_id: str,
    sender_name: str,
    text: str,
) -> AsyncIterator[dict]:
    headers = {
        "Authorization": f"Bearer {token}",
        "x-user-id": user_id,
        "x-session-id": session_id,
        "x-sender-name": quote(sender_name, safe=""),
    }
    data = {
        "content": json.dumps([{"text": text}], ensure_ascii=False),
        "session_metadata": json.dumps(
            {"channel": "my_backend"}, ensure_ascii=False
        ),
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with aconnect_sse(
            client,
            "POST",
            f"{base_url.rstrip('/')}/chat",
            headers=headers,
            data=data,
        ) as source:
            source.response.raise_for_status()
            async for event in source.aiter_sse():
                yield json.loads(event.data)


async def collect_answer(**kwargs) -> str:
    chunks: list[str] = []
    async for event in chat_events(**kwargs):
        if event["event"] == "assistant":
            chunks.append(event["context"])
        elif event["event"] == "question":
            # 将 json.loads(event["context"]) 交给业务 UI 展示。
            # 答案必须由另一个并发请求提交，不能等待当前 SSE 结束。
            pass
    return "".join(chunks)
```

服务端生成可能持续较长时间，HTTP 客户端的读取超时应关闭或设置为大于 Agent 最长运行时间。反向代理也需要关闭响应缓冲并适当提高空闲超时，例如 Nginx 使用 `proxy_buffering off`。

## 7. 处理 Agent 追问

收到 `question` 事件后，`context` 是一个表单定义。例如：

```json
{
  "type": "form",
  "title": "部署参数确认",
  "subtitle": "您有300秒时间填写表单",
  "submit_label": "确认并继续",
  "fields": [
    {
      "type": "input",
      "name": "service_name",
      "label": "服务名称",
      "placeholder": "例如 order-api",
      "default_value": "order-api"
    },
    {
      "type": "select",
      "name": "environment",
      "label": "请选择环境",
      "options": [
        {"label": "测试环境", "value": "test"},
        {"label": "生产环境", "value": "prod"}
      ]
    },
    {
      "type": "multi_select",
      "name": "regions",
      "label": "部署区域",
      "options": [
        {"label": "上海", "value": "cn-shanghai"},
        {"label": "北京", "value": "cn-beijing"}
      ]
    },
    {
      "type": "date",
      "name": "release_date",
      "label": "发布日期"
    },
    {
      "type": "datetime",
      "name": "change_window",
      "label": "变更时间"
    }
  ]
}
```

字段类型可能为 `input`、`select`、`multi_select`、`date` 或 `datetime`。推荐提交的答案是以字段 `name` 为键的 JSON 对象：

```json
{
  "service_name": "order-api",
  "environment": "prod",
  "regions": ["cn-shanghai", "cn-beijing"],
  "release_date": "2026-10-29",
  "change_window": "2026-10-29 22:00:00"
}
```

用户提交后，请在原 `/chat` SSE 仍保持连接时调用：

```http
POST /answer_question
Authorization: Bearer <new JWT>
x-user-id: <与 /chat 完全相同>
x-session-id: <与 /chat 完全相同>
Content-Type: application/x-www-form-urlencoded
```

表单只有一个 `answer` 字段。使用 `application/x-www-form-urlencoded` 或 `multipart/form-data` 均可；`answer` 的值是序列化后的 JSON 字符串：

```bash
curl -X POST "http://localhost:7878/answer_question" \
  -H "Authorization: Bearer <JWT>" \
  -H "x-user-id: demo-user" \
  -H "x-session-id: chat-001" \
  -F 'answer={"service_name":"order-api","environment":"prod","regions":["cn-shanghai","cn-beijing"],"release_date":"2026-10-29","change_window":"2026-10-29 22:00:00"}'
```

成功响应：

```json
{"message":"Answer received"}
```

如果没有找到对应的等待事件，会返回：

```json
{"message":"Question event not found"}
```

注意：

- 问题默认等待 300 秒，超时后 Agent 会自动继续处理。
- `/answer_question` 必须复用原请求的 `user_id` 和 `session_id`，否则无法唤醒对应会话。
- JWT 最长有效期为 60 秒，因此答题时通常需要签发一个新 Token。
- 不要关闭原 `/chat` SSE 后再答题；Agent 正在该连接对应的任务中等待答案。

### 7.1 可运行的 Python 完整示例

下面的命令行示例演示了完整闭环：发起聊天、持续消费 SSE、解析所有问题字段、收集用户输入、签发新 JWT、通过第二个 HTTP 请求提交答案，然后继续读取原 SSE，直到 Agent 完成本轮回复。

先安装依赖：

```bash
pip install httpx httpx-sse pyjwt
```

保存为 `xpeech_question_demo.py`：

```python
import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import quote

import httpx
import jwt
from httpx_sse import aconnect_sse


BASE_URL = os.getenv("XPEECH_BASE_URL", "http://localhost:7878").rstrip("/")
JWT_SECRET = os.environ["XPEECH_JWT_SECRET"]
USER_ID = "demo-user"
SESSION_ID = "chat-001"
SENDER_NAME = "张三"


def create_token() -> str:
    """每次 API 调用前签发新 Token，避免答题时复用过期 Token。"""
    return jwt.encode(
        {"exp": datetime.now(UTC) + timedelta(seconds=60)},
        JWT_SECRET,
        algorithm="HS256",
    )


async def async_input(prompt: str) -> str:
    """避免同步 input() 阻塞 asyncio 事件循环。"""
    return (await asyncio.to_thread(input, prompt)).strip()


async def collect_question_answer(question: dict[str, Any]) -> dict[str, Any]:
    """在终端展示问题，并按字段类型组装答案对象。"""
    print(f"\n[{question.get('title', '需要补充信息')}]")
    if subtitle := question.get("subtitle"):
        print(subtitle)

    answer: dict[str, Any] = {}
    for field in question.get("fields", []):
        field_type = field["type"]
        name = field["name"]
        label = field["label"]
        placeholder = field.get("placeholder")
        default_value = field.get("default_value", "")
        hint = f"（{placeholder}）" if placeholder else ""

        if field_type in {"select", "multi_select"}:
            options = field.get("options", [])
            print(f"\n{label}{hint}")
            for index, option in enumerate(options, start=1):
                print(f"  {index}. {option['label']} [{option['value']}]")

            while True:
                suffix = "，多个选项用逗号分隔" if field_type == "multi_select" else ""
                raw = await async_input(f"请选择序号{suffix}: ")
                try:
                    indexes = [int(item.strip()) for item in raw.split(",")]
                    if not indexes or any(index < 1 or index > len(options) for index in indexes):
                        raise ValueError
                except ValueError:
                    print("输入无效，请重新选择。")
                    continue

                values = [options[index - 1]["value"] for index in indexes]
                answer[name] = values if field_type == "multi_select" else values[0]
                break
            continue

        prompt = f"{label}{hint}"
        if default_value:
            prompt += f" [默认：{default_value}]"
        if field_type == "date":
            prompt += " [格式：YYYY-MM-DD]"
        elif field_type == "datetime":
            prompt += " [格式：YYYY-MM-DD HH:mm:ss]"

        value = await async_input(f"{prompt}: ")
        answer[name] = value or default_value

    return answer


async def submit_answer(answer: dict[str, Any]) -> None:
    """使用新 JWT 和相同的用户、会话标识回答当前追问。"""
    headers = {
        "Authorization": f"Bearer {create_token()}",
        "x-user-id": USER_ID,
        "x-session-id": SESSION_ID,
    }
    data = {"answer": json.dumps(answer, ensure_ascii=False)}

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{BASE_URL}/answer_question",
            headers=headers,
            data=data,
        )
        response.raise_for_status()
        result = response.json()

    if result.get("message") != "Answer received":
        raise RuntimeError(f"追问已超时或会话不匹配：{result}")
    print("答案已提交，等待 Agent 继续处理……\n")


async def run_chat(text: str) -> str:
    """发起聊天；遇到 question 时回答，然后继续消费原 SSE。"""
    headers = {
        "Authorization": f"Bearer {create_token()}",
        "x-user-id": USER_ID,
        "x-session-id": SESSION_ID,
        "x-sender-name": quote(SENDER_NAME, safe=""),
    }
    data = {
        "content": json.dumps([{"text": text}], ensure_ascii=False),
        "session_metadata": json.dumps(
            {"channel": "python_demo"}, ensure_ascii=False
        ),
    }
    assistant_chunks: list[str] = []

    async with httpx.AsyncClient(timeout=None) as client:
        async with aconnect_sse(
            client,
            "POST",
            f"{BASE_URL}/chat",
            headers=headers,
            data=data,
        ) as source:
            source.response.raise_for_status()

            async for sse in source.aiter_sse():
                event = json.loads(sse.data)
                event_type = event["event"]
                context = event["context"]

                if event_type == "assistant":
                    assistant_chunks.append(context)
                    print(context, end="", flush=True)
                elif event_type == "assistant_end":
                    print()
                elif event_type == "thinking":
                    print(f"[思考] {context}", end="", flush=True)
                elif event_type == "thinking_end":
                    print()
                elif event_type == "question":
                    question = json.loads(context)
                    answer = await collect_question_answer(question)
                    # submit_answer 会新建 HTTP 连接；此时 /chat SSE 仍保持打开。
                    await submit_answer(answer)
                elif event_type == "send_file":
                    print(f"\n[文件] {context}")
                elif event_type == "command":
                    print(f"\n[命令] {context}")
                elif event_type == "token_usage":
                    print(f"\n[统计] {json.loads(context)}")
                elif event_type in {"tool_call", "tool_call_result"}:
                    print(f"\n[{event_type}] {json.loads(context)}")

    return "".join(assistant_chunks)


async def main() -> None:
    final_text = await run_chat("帮我制定部署计划，缺少的信息请向我提问")
    print(f"\n完整助手输出：\n{final_text}")


if __name__ == "__main__":
    asyncio.run(main())
```

运行：

```bash
export XPEECH_BASE_URL="http://localhost:7878"
export XPEECH_JWT_SECRET="<conf.toml 中的 jwt.secret_key>"
python xpeech_question_demo.py
```

生产环境通常不会在 Xpeech 接入进程中直接等待终端输入。更常见的流程是：接入后端收到 `question` 事件后，将 `context` 和自己的会话标识通过 WebSocket/SSE 转发给前端；前端提交表单到接入后端；接入后端再用已保存的 `user_id + session_id` 调用 Xpeech `/answer_question`。整个过程中，负责消费 `/chat` 的任务必须继续存活，不能提前关闭 SSE。

## 8. 下载 Agent 生成的文件

收到事件：

```json
{"event":"send_file","context":"outputs/result.xlsx"}
```

使用事件中的 `context` 原值作为 `path` 查询参数：

```bash
curl --get "http://localhost:7878/sessions/demo-user/files" \
  -H "Authorization: Bearer <JWT>" \
  --data-urlencode "path=outputs/result.xlsx" \
  --output result.xlsx
```

路径参数会限制在该用户的工作区中。文件不存在返回 `404`，路径越界返回 `403`。响应中的 `Content-Disposition` 包含建议文件名。

## 9. 接收飞书后台任务消息（可选）

这一接口只用于自建飞书投递通道接收定时 Agent 任务的执行结果；普通聊天接入不需要调用。

```http
GET /background_message?channel=feishu
Authorization: Bearer <JWT>
```

接口采用最长约 50 秒的长轮询：有消息时返回 `200` 和 JSON，无消息时返回 `204 No Content`。

```json
{
  "channel": "feishu",
  "open_id": "ou_xxx",
  "content": "后台任务执行结果"
}
```

投递方收到消息后，应使用 `open_id` 将 `content` 发送给对应飞书用户，然后立即发起下一次长轮询。当前只支持 `channel=feishu`。

## 10. 内置会话命令

当 `content` 只包含一个文本块且正文以 `/` 开头时，Xpeech 会优先处理内置命令：

| 命令 | 说明 |
| --- | --- |
| `/help` | 查看命令列表 |
| `/new` | 总结并保存记忆，清空当前会话历史 |
| `/clear` | 直接清空当前会话历史 |

`/new` 和 `/clear` 后可继续携带新消息，例如 `/new 帮我规划下一季度工作`。命令结果通过 `command` 事件返回。

## 11. 状态码和异常处理

| 状态码 | 常见原因 | 建议处理 |
| --- | --- | --- |
| `400` | 开启了服务端不支持的 `json_output` | 关闭该参数或修改模型配置 |
| `401` | JWT 缺失、签名错误或已过期 | 重新签发 Token；不要盲目无限重试 |
| `403` | 文件路径越出用户工作区 | 检查是否原样使用 `send_file.context` |
| `404` | 下载文件不存在 | 提示文件已不存在或记录日志 |
| `409` | 同一用户、同一会话已有聊天请求 | 等待现有请求完成，按会话排队 |
| `422` | 请求头 ID 非法、用户名为空、JSON 字符串或字段格式不合法 | 修正参数，不重试原请求 |
| `500` | 模型服务或 Agent 执行异常 | 记录响应和请求追踪信息，有限次数退避重试 |

HTTP 错误可能发生在 SSE 建立前，此时响应体通常为：

```json
{"detail":"错误说明"}
```

SSE 建立后发生异常时，连接可能直接结束，不能假设一定会收到业务错误事件。接入方应区分“正常收到 EOF”和“网络/解析异常”，并根据业务幂等策略决定是否允许用户重新发送。聊天请求没有请求幂等键，自动重试可能造成重复模型调用或重复工具操作，因此不建议在连接中断后无条件自动重放。

## 12. 生产接入检查清单

- JWT 密钥只保存在可信后端和 Xpeech 配置中，未下发到浏览器或 App。
- Xpeech 对外访问使用 HTTPS，API 网络入口受到访问控制。
- 每次 API 调用都签发短期 JWT，答题回调不会复用已过期 Token。
- `user_id` 使用业务侧稳定且不可冲突的标识。
- `session_id` 能区分多个聊天窗口，并在聊天、答题流程中保持一致。
- 同一会话的消息在接入层串行化，能正确处理 `409`。
- SSE 按空行分隔事件并解析 `data`，不会按 TCP chunk 直接解析 JSON。
- `assistant` 分片按序拼接，以 EOF 判断整轮完成。
- `question` 事件通过并行 HTTP 请求回答，原 SSE 连接保持打开。
- `send_file.context` 经过 URL 编码后传给下载接口。
- HTTP 客户端、网关和反向代理已为长连接配置足够的超时并关闭缓冲。
- 不对中断的聊天请求进行无条件自动重放。
