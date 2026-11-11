# Xpeech API 对接文档

面向需要通过 HTTP 接口直接对接 Xpeech Agent 的外部系统。读完本文可以完成：JWT 认证、发起对话并消费 SSE 事件流、处理 `ask_user_question` 表单追问、处理 `send_file` 文件下载，以及使用同步接口和 JSON 输出模式。

- 默认地址：`http://<host>:7878`，下文以 `http://localhost:7878` 举例；实际地址以部署配置为准。
- 健康检查：`GET /health`，无需认证，返回纯文本 `ok`。
- OpenAPI 文档：`GET /docs`（Swagger UI），也可使用 `/redoc`、`/openapi.json`。
- 本文接口面向服务到服务调用。Web 客户端的 `/api/chat`、`/api/answer_question`、`/api/files` 是基于登录 Cookie 的代理接口，不能混用认证方式。

---

## 1. 整体交互模型

```text
你的系统                                      Xpeech API
  |                                               |
  |-- 本地签发 JWT，或 POST /token --------------->|  获得短期 Bearer Token
  |                                               |
  |-- POST /chat（SSE 长连接，一条消息）----------->|
  |<------------- text/event-stream ---------------|  持续推送 Agent 事件
  |      assistant / thinking / tool_call / ...     |
  |                                               |
  |   [若收到 question 事件]                       |
  |-- POST /answer_question（另一条请求）--------->|  原 /chat 连接保持打开
  |<------------- SSE 流继续 ----------------------|  Agent 恢复执行
  |                                               |
  |   [若收到 send_file 事件]                      |
  |-- GET /files?path=... ------------------------>|  下载用户工作区文件
  |                                               |
  |<------------- token_usage，随后 EOF -----------|  常规对话完成
```

核心约定：

1. **一次 `/chat` 请求 = 一轮对话**。请求体是一条用户消息，响应是一条 SSE 事件流；以流正常关闭（EOF）作为整轮完成信号。常规对话在结束前发送 `token_usage`，纯命令可能直接结束。
2. **JWT `sub` 标识用户，`x-session-id` 标识会话**。同一通道、同一用户、同一会话的流式多轮消息共享历史上下文；同一用户的不同会话共享文件工作区和记忆。
3. **同一用户、同一会话只允许一个进行中的 `/chat` 请求**，重复请求返回 `409`。`/chat_sync` 不占用会话并发锁，可与 `/chat` 或其他同步调用并发。收到 `question` 时原流仍在运行，应通过另一条请求提交答案。
4. **业务请求使用通道 JWT**。聊天、答题、文件和技能接口还必须包含用户 `sub`；不能使用内部服务 JWT 代替。

不需要流式返回时，可调用 `POST /chat_sync`，一次性获取事件列表，见第 3.4 节。

---

## 2. JWT 认证

Xpeech 与调用方共享通道密钥，API 通过标准请求头校验：

```http
Authorization: Bearer <access_token>
```

### 2.1 密钥与 Token 结构

服务端在 `conf.toml` 中为每个通道配置独立密钥。以下以 `web_client` 为例；自建系统也可以使用独立通道（如 `crm`）：

```toml
[jwt]
secret_key = "replace-with-an-internal-secret-of-at-least-32-characters"
algorithm = "HS256"
access_token_expire_seconds = 60

[jwt.channels.web_client]
secret_key = "replace-with-a-distinct-web-client-secret-at-least-32-characters"

[jwt.channels.feishu]
secret_key = "replace-with-a-distinct-feishu-secret-at-least-32-characters"

[jwt.channels.crm]
secret_key = "replace-with-a-distinct-crm-secret-at-least-32-characters"
```

| 项 | 约定 |
| --- | --- |
| 签名密钥 | 使用 `jwt.channels.<通道名>.secret_key`，长度至少 32；内部密钥和所有通道密钥必须互不相同 |
| 算法 | 默认 `HS256`，支持 `HS256` / `HS384` / `HS512`，签发方与服务端配置保持一致 |
| JWT header `kid` | 必填，值为已配置的通道名，例如 `web_client`、`feishu`、`crm`；不能使用保留名 `internal` |
| JWT payload `exp` | 必填，过期时间；服务端签发时有效期由 `jwt.access_token_expire_seconds` 决定，取值 1–60 秒，默认 60 秒；自行签发也应遵循此配置 |
| JWT payload `sub` | 用户接口必填，用作稳定的用户 ID，规则见第 3.1 节 |

`jwt.secret_key` 仅供 backend、scheduler 等内部服务互信。内部 JWT 不带 `kid` 或 `sub`，不能调用本文的聊天、答题、文件、技能接口；`/statistics` 及其子接口、内部调度接口也不属于外部通道的权限范围。

通道名就是 JWT header 的 `kid`，由服务端按验签后的 `kid` 读取，接入方不能通过 payload、请求头或元数据覆盖。内建技能、用户工作区技能与 MCP 使用 `filter` 过滤表达式：省略 `filter` 的资源为公共资源，一律展示或注册；配置 `filter` 的资源仅在表达式求值为真时可见。表达式为 Python 布尔表达式极小子集，内置 `channel("...")`（匹配通道名）与 `org("...")`（匹配组织前缀）两个短名函数，服务端按白名单解析求值，不使用 eval。

`org()` 依赖服务端通过 `[organization].command_template` 查询当前已鉴权 `user_id` 的组织字符串，使用 stdout 作为组织值。命令必须包含 `${user_id}` 或 `$user_id`，默认超时 10 秒；JWT 或请求元数据不能覆盖组织。无查询命令、查询失败或空结果时组织视为未知，`org()` 不匹配任何前缀。过滤表达式的完整语法与配置示例见 README。

### 2.2 方式 A：本地签发（业务系统推荐）

持有通道密钥的可信后端可以直接签发 JWT，省去一次网络往返：

```python
from datetime import UTC, datetime, timedelta

import jwt  # pip install pyjwt


def create_xpeech_token(secret: str, user_id: str, kid: str = "web_client") -> str:
    return jwt.encode(
        {"exp": datetime.now(UTC) + timedelta(seconds=60), "sub": user_id},
        secret,
        algorithm="HS256",
        headers={"kid": kid},
    )
```

示例使用默认算法和有效期；服务端调整配置后，签发方需同步调整。密钥只保存在可信后端，不下发到浏览器或移动端。

### 2.3 方式 B：调用 `POST /token` 换取

该接口主要用于 Swagger UI 调试，也可由可信后端调用。**通道凭据通过 HTTP Basic 传入，用户 ID 通过表单 `username` 传入**：

```http
POST /token
Authorization: Basic <base64(channel:channel_secret)>
Content-Type: application/x-www-form-urlencoded

username=demo-user
```

```bash
curl -sS --fail-with-body -X POST "http://localhost:7878/token" \
  --user "web_client:<jwt.channels.web_client.secret_key>" \
  --data-urlencode 'username=demo-user'
```

成功响应：

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

- 返回的用户 Token 同时包含通道 `kid` 和用户 `sub`。表单 `username` 留空时只签发通道 Token，不能调用用户接口。
- 凭据错误返回 `401`，`detail` 为 `Incorrect JWT secret`，响应头带 `WWW-Authenticate: Basic`。
- 成功响应带 `Cache-Control: no-store` 和 `Pragma: no-cache`。
- Swagger UI 的 `client_id` 填通道名、`client_secret` 填对应密钥，凭据位置选择 Authorization header；`username` 填用户 ID，`password` 留空。

### 2.4 刷新与认证失败

未携带、无效或过期的 Bearer Token，以及缺少或非法用户 `sub`，均返回 `401`。建议**每次 API 调用前重新签发或换取 Token**，尤其是等待用户答题后或下载文件前。

SSE 一旦建立，不会因 Token 在流中途过期而中断；`/answer_question`、`/files` 是独立请求，仍需携带有效 Token。答题、下载和技能请求应使用与聊天相同的用户 `sub`。

### 2.5 飞书用户令牌（可选）

飞书通道需要以用户身份调用命令时，可在用户 JWT payload 中额外携带字符串 `lark_user_access_token`：

```python
claims = {
    "sub": "demo-user",
    "exp": datetime.now(UTC) + timedelta(seconds=60),
    "lark_user_access_token": "<feishu-user-access-token>",
}
```

使用 `jwt.channels.feishu.secret_key` 签名，header 设置 `kid="feishu"`。该声明仅用于当前请求，不保存到会话历史，后续请求需重新携带；内部 JWT 不接受此声明。

携带非空令牌时，沙盒加载 `lark-skills-token`，`lark-cli` 使用该令牌且不回退请求 `lark-token-manager`；未携带时加载 `lark-skills` 和 `lark-cli-auth`，使用令牌容器。

---

## 3. 发起对话 `POST /chat`

通过表单发送一条用户消息，以 SSE 返回 Agent 的执行过程。

### 3.1 用户、会话与请求头

| 头或声明 | 必填 | 说明 |
| --- | --- | --- |
| `Authorization` | 是 | `Bearer <token>`，见第 2 节 |
| JWT `sub` | 是 | 用户 ID，长度 1–128，只允许 ASCII 字母、数字、空格、连字符 `-` 和下划线 `_` |
| `x-session-id` | `/chat`、`/chat/stop`、`/answer_question` 必填；`/chat_sync` 不需要 | 规范的小写连字符 UUIDv7，例如 `018f0000-0000-7000-8000-000000000001`；所有通道均遵循此规则 |
| `x-sender-name` | 是 | 展示用用户名，不能为空或纯空白；非 ASCII 字符使用 UTF-8 URL 编码 |
| `Accept` | 建议 | `text/event-stream` |

例如 `quote("张三", safe="")` 对应：

```http
x-sender-name: %E5%BC%A0%E4%B8%89
```

用户身份只取已验签 JWT 的 `sub`，对外接口忽略 `x-user-id` 请求头。响应在有对应值时回传 `x-user-id` 和 `x-channel`。

流式会话历史按 `channel + user_id + session_id` 隔离；并发限制按 `user_id + session_id` 判断。换一个会话 ID 可以开启独立历史，**不会隔离同一用户的文件工作区与记忆**。流式模式的所有会话都会注册后台任务工具；同步模式会禁用这些工具。

### 3.2 请求体（form 表单）

无附件时可以使用 `application/x-www-form-urlencoded`；带附件时必须使用 `multipart/form-data`。

| 字段 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `content` | 是 | — | **JSON 字符串**，文本块列表，例如 `[{"text":"你好"},{"text":"第二段"}]`；目前仅支持文本块；文本中加入完整标记 `::请严格以JSON格式回复::` 可启用 JSON 输出，见第 3.5 节 |
| `session_metadata` | 否 | `"{}"` | **JSON 字符串**，元数据对象，例如 `{"sender_id":"u-123"}`；值请使用字符串 |
| `files` | 否 | 空 | multipart 文件字段，可重复多个；附件保存到用户工作区的 `attachments/YYYY-MM-DD/` 下，重名自动加序号 |

`content`、`session_metadata` 是表单里的字符串，不能把整个请求发送为 `application/json`。非法 JSON 返回 `422`；`session_metadata` 必须为对象。渠道由 JWT `kid` 决定，元数据中的 `channel` 键会被丢弃。

消息时间由服务端记录，当前接口不提供 `timestamp` 表单参数。

最小请求：

```bash
curl -N --fail-with-body -X POST "http://localhost:7878/chat" \
  -H "Authorization: Bearer <JWT>" \
  -H "x-session-id: 018f0000-0000-7000-8000-000000000001" \
  -H "x-sender-name: demo-user" \
  -H "Accept: text/event-stream" \
  -F 'content=[{"text":"你好，请介绍一下自己"}]'
```

### 3.3 响应与内置命令

响应 `Content-Type: text/event-stream`，事件格式见第 4 节。HTTP `200` 表示请求通过校验、流已建立；Agent 的后续执行通过事件流返回。

在流式接口中，`content` 只有一个文本块且去除首尾空白后以 `/` 开头时，优先处理内置命令：

| 指令 | 行为 |
| --- | --- |
| `/help` | 返回指令说明 |
| `/new` | 总结并保存记忆，然后清空当前会话历史；可拼接正文，如 `/new 帮我规划下一季度工作` |
| `/clear` | 直接清空当前会话历史，不做记忆总结；同样支持拼接正文 |
| 其他 `/xxx` | 返回未识别提示 |

命令通过 `command` 事件回显。没有后续正文的纯命令会直接结束流，**不保证发送 `token_usage`**。系统提示（如压缩上下文）也可能通过 `command` 返回。

### 3.4 同步接口 `POST /chat_sync`

请求头、表单字段与 `/chat` 相同，但不需要 `x-session-id`。接口阻塞到 Agent 运行结束，一次性返回 JSON 事件列表：

```bash
curl --fail-with-body -X POST "http://localhost:7878/chat_sync" \
  -H "Authorization: Bearer <JWT>" \
  -H "x-sender-name: demo-user" \
  -F 'content=[{"text":"你好，请介绍一下自己"}]'
```

```json
[
  {"event":"assistant","context":"你好！我是 Xpeech Agent。"},
  {"event":"token_usage","context":"{\"大模型请求次数\":\"1\"}"}
]
```

- 保留事件顺序，合并连续的 `assistant`、`thinking` 分片，过滤 `assistant_end`、`thinking_end`；文本不会跨结束标记合并，其他事件保留。
- 与 `/chat` 共享用户工作区，但不占用会话并发锁，可与 `/chat` 或其他同步调用并发。每次同步调用在服务端生成独立会话 ID，从空的历史开始，不读取已有聊天上下文；本轮历史另存为非交互类型，不影响流式会话历史。需要连续多轮上下文时请使用 `/chat`。
- 在工具注册阶段跳过追问及创建、查看、取消调度工具，适合不需要交互的任务。
- 同步模式不拦截 `/help`、`/new`、`/clear` 等内置命令，这些内容按普通消息处理。
- 客户端和网关需为整个任务设置足够的等待时间。

### 3.5 JSON 输出模式

`/chat`、`/chat_sync` 根据本次用户消息中的完整标记 `::请严格以JSON格式回复::` 判断是否启用 JSON 输出，不再通过 `json_output` 表单参数控制。服务端需启用：

```toml
[llm]
support_json_output = true
```

在 `content` 的文本块中加入该标记，并说明目标 JSON 结构，例如：

```json
[{"text":"列出三个新闻源，返回包含 name 和 url 字段的对象。::请严格以JSON格式回复::"}]
```

标记必须完整、精确匹配，前后的 `::` 均不可省略。不带符号的“请严格以JSON格式回复”、缺少任一侧符号或普通的 `JSON` 字样都不会触发。
标记保留在发送给模型的原始文本中，仅本次请求生效，不从历史消息继承。
触发后为模型请求添加 `response_format={"type":"json_object"}`，不会额外修改提示词。

未启用能力时返回 `400`，`detail` 为 `The configured model does not support JSON output`，发生在 SSE 建立前。启用后响应格式保持不变：流式接口仍返回 SSE，同步接口仍返回事件列表。模型输出位于 `assistant.context` 中；流式分片需先拼接，再解析完整的 JSON 文本。

### 3.6 停止生成 `POST /chat/stop`

通过独立请求停止当前用户指定会话的 SSE 生成任务：

```bash
curl --fail-with-body -X POST "http://localhost:7878/chat/stop" \
  -H "Authorization: Bearer <JWT>" \
  -H "x-session-id: 018f0000-0000-7000-8000-000000000001"
```

无需请求体或 `x-sender-name`。用户取已验签 JWT 的 `sub`，会话取 `x-session-id`；不会停止其他用户
或其他会话的任务。返回 HTTP `200`，响应为 `{"stopped": true}` 或 `{"stopped": false}`：

- `true`：生成任务已退出，占用的大模型信号量名额已归还。排队等待名额、模型响应及用户回答时均可停止。
- `false`：当前用户的该会话没有活跃生成任务；重复调用安全。

原 SSE 流随后正常结束，已进入网络缓冲区的事件仍可能到达，不保证发送 `assistant_end`、
`thinking_end` 或 `token_usage`；客户端应在 EOF 时结束展示。会话并发锁在原响应结束后释放。
Agent 通过 `finally` 保存已经加入上下文的消息，停止请求会等待这次保存收尾。
本接口仅停止 `/chat`，不作用于 `/chat_sync` 或已提交的后台定时任务。
停止请求需要有效 JWT；原聊天的 JWT 已过期时应重新签发。
多实例部署须让 `/chat/stop` 和 `/chat` 按 JWT `sub` 落到同一 backend 进程。

Web 客户端提供 `POST /api/chat/stop` 代理，复用现有登录 Cookie 或嵌入 JWT 认证及会话校验。

---

## 4. SSE 事件流协议

### 4.1 线上格式

每条事件通过标准 SSE `data:` 字段发送，**事件类型放在 JSON 的 `event` 中，SSE 层不设置 `event:` 名称**：

```text
data: {"event":"assistant","context":"你好"}

data: {"event":"assistant","context":"，有什么可以帮你？"}

data: {"event":"assistant_end","context":""}

data: {"event":"token_usage","context":"{\"上下文使用率\":\"1.56% (2k / 128k)\",\"输入 Token\":\"1.23k\",\"输出 Token\":\"567\",\"总 Token\":\"1.8k\",\"会话时长\":\"5秒\",\"大模型请求次数\":\"1\"}"}

```

统一结构：

```json
{
  "event": "<事件类型>",
  "context": "<字符串型载荷>"
}
```

解析约定：

- **`context` 始终是字符串**。纯文本直接使用；结构化事件需要对 `context` 再执行一次 `JSON.parse` / `json.loads`。
- SSE 按空行分隔事件。TCP chunk 可能只包含半个事件或多个事件，不能直接逐 chunk 解析 JSON。
- 忽略以 `:` 开头的 SSE 注释和保活行；不要依赖注释内容或 SSE 的 `event:` 字段。
- 流是一次性的，没有 `id:` / `Last-Event-ID` 续传机制。
- `/chat` 使用 POST，浏览器原生 `EventSource` 仅支持 GET；请使用 `fetch` + `ReadableStream` 或支持 POST 的 SSE 客户端。Python 可使用 `httpx` + `httpx-sse`。

### 4.2 事件类型一览

| `event` | `context` 格式 | 含义与处理方式 |
| --- | --- | --- |
| `thinking` | 纯文本增量片段 | 思考过程，按顺序拼接，可按业务需要展示或隐藏 |
| `thinking_end` | `""` | 当前思考段结束 |
| `assistant` | 纯文本增量片段 | 回复正文，按顺序拼接到当前正文段 |
| `assistant_end` | `""` | 当前正文段结束；工具调用前后可能有多个正文段 |
| `tool_call` | JSON 字符串：`[["<id>","<tool_name>",{参数对象}], ...]` | 一批工具调用 |
| `tool_call_result` | JSON 字符串：`{"id":"<id>","name":"<tool_name>","result":结果,"duration_seconds":耗时秒数}` | 单次工具调用的结果；`result` 可能为文本或结构化内容 |
| `question` | 问题表单的 JSON 字符串 | Agent 等待用户回答，原流保持打开，见第 5 节 |
| `send_file` | 用户工作区相对路径 | 例如 `outputs/report.pdf`，通过 `/files` 下载，见第 6 节 |
| `command` | 纯文本 | 内置命令回显或系统提示 |
| `token_usage` | 指标的 JSON 字符串 | 含 `上下文使用率`、`输入 Token`、`输出 Token`、`总 Token`、`会话时长`、`大模型请求次数`；常规对话的末尾统计 |

Token 消耗按本次请求内所有模型调用汇总，包括工具调用前后的多轮请求，以及上下文压缩、记忆整理等内部模型调用。
`输入 Token`、`输出 Token` 使用模型提供方返回的用量；`总 Token` 为累计输入与输出用量之和。
某次调用未返回某项用量时，该次用量按 `0` 累计。计数先精确汇总，再按十进制单位展示：
1000 为 `1k`，100 万为 `1m`，最多保留两位小数并去除末尾的 0，例如 `1.23k`、`1.5m`。
`上下文使用率` 是当前上下文大小的估算，Token 消耗是本轮累计用量，两者统计口径不同。

`tool_call` 一次包含一批调用，`tool_call_result` 每个事件包含一个结果对象。例如，解析结果事件的 `context` 后得到：

```json
{"id": "call-1", "name": "shell", "result": "hello", "duration_seconds": 0.12}
```

`id` 为工具调用 ID，`name` 为工具名，`result` 为执行结果，`duration_seconds` 为耗时秒数。并行调用的结果按完成顺序逐条发送，可能与调用顺序不同；请使用 `id` 匹配对应调用并更新状态。

**`send_file`、`ask_user_question` 的成功调用也会出现在 `tool_call_result` 中**：文件结果为 `已发送：<路径>`，追问结果为用户回答或超时默认答案。展示文件和问题表单时请使用专属事件，避免重复处理。

### 4.3 典型事件时序与结束判断

```text
thinking → ... → thinking_end
assistant → ... → assistant_end       （可选的中间说明）
tool_call
  ├─ question → 等待回答              （可选）
  └─ send_file                        （可选）
tool_call_result                     （每条结果一个事件，可能连续出现多次）
assistant → ... → assistant_end       （后续回复）
token_usage
EOF                                  （本轮连接关闭）
```

一轮可能多次循环「模型 → 工具 → 模型」。`assistant_end`、`question`、`send_file` 都不代表整轮结束。

**以 SSE 正常 EOF 作为本轮完成信号**，并在 EOF 时兜底结束尚未收到 `*_end` 的展示段。常规对话末尾有 `token_usage`，但纯命令可能没有；网络或解析异常造成的中断应单独处理，见第 7.2 节。

---

## 5. `ask_user_question`：表单追问交互

当 Agent 缺少关键信息时，可能调用 `ask_user_question`。API 表现为 `/chat` 流上出现 `question` 事件，随后等待回答；**不要关闭原 SSE 或重新发送聊天请求**。

### 5.1 `question.context` 结构

`context` 是 JSON 字符串，解析后为通用表单对象：

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
      "label": "部署环境",
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
    {"type": "date", "name": "release_date", "label": "发布日期"},
    {"type": "datetime", "name": "change_window", "label": "变更时间"}
  ]
}
```

| 键 | 说明 |
| --- | --- |
| `type` | 固定为 `"form"` |
| `title` | 表单标题，默认 `"需要补充信息"` |
| `subtitle` | 副标题，当前服务端写入 `"您有300秒时间填写表单"`，不是实时倒计时 |
| `submit_label` | 提交按钮文案，默认 `"提交"` |
| `fields[].type` | `input` 文本 / `select` 单选 / `multi_select` 多选 / `date` 日期 / `datetime` 日期时间 |
| `fields[].name` | 字段名，匹配 `^[a-z][a-z0-9_]*$`，表单内唯一；提交时作为 key |
| `fields[].label` | 展示名称 |
| `fields[].placeholder` | 占位提示，可为 `null` |
| `fields[].default_value` | 仅文本字段具有，默认 `""` |
| `fields[].options` | 单选或多选字段的 `{label, value}` 列表；提交时使用 `value` |

`fields` 可能为空数组，此时可收集自由文本。

### 5.2 提交回答 `POST /answer_question`

回答与聊天使用两条独立 HTTP 请求：

```http
POST /answer_question
Authorization: Bearer <new-token>
x-session-id: <与 /chat 相同的会话 ID>
Content-Type: application/x-www-form-urlencoded

answer=<回答内容>
```

| 项 | 说明 |
| --- | --- |
| JWT `sub` | 必须与原 `/chat` 相同，答题时使用新 Token |
| `x-session-id` | 必须与正在等待的 `/chat` 相同 |
| `answer` | 必填表单字符串；推荐为以字段 `name` 为键的 JSON 文本，也接受自由文本 |

推荐答案格式：

```json
{
  "service_name": "order-api",
  "environment": "prod",
  "regions": ["cn-shanghai", "cn-beijing"],
  "release_date": "2026-11-05",
  "change_window": "2026-11-05 22:00:00",
  "user_customization": "其他补充说明"
}
```

单选提交一个 `value` 字符串，多选提交 `value` 字符串数组；日期、日期时间建议使用 `YYYY-MM-DD`、`YYYY-MM-DD HH:mm:ss`。可附加 `user_customization` 等自定义 key。

成功响应（HTTP `200`）：

```json
{"message":"Answer received"}
```

Agent 被唤醒，原 `/chat` SSE 继续推送。找不到等待事件时同样返回 HTTP `200`：

```json
{"message":"Question event not found"}
```

此时通常是问题已结束或已超时，也应检查用户、会话是否一致，无需盲目重试。表单也可以使用 `multipart/form-data` 提交。

### 5.3 超时行为与前后端协作

等待时间为 **300 秒**。超时后以 `User timeout answering the question` 作为默认答案继续执行，流不会因此中断。接入方应在 5 分钟内提交，并及时关闭过期表单。

生产接入中，后端收到 `question` 后，可将表单和业务会话标识通过 WebSocket/SSE 转发给前端；前端向接入后端提交答案，再由后端使用相同的 `sub + session_id` 调用 `/answer_question`。负责消费原 `/chat` 的任务必须保持存活。

---

## 6. `send_file`：文件下发与下载

### 6.1 事件示例

```text
data: {"event":"send_file","context":"outputs/report.pdf"}

```

`context` 是**用户工作区内的相对路径**，不是 URL，不包含用户工作区的绝对目录。

### 6.2 下载接口 `GET /files`

```http
GET /files?path=outputs/report.pdf
Authorization: Bearer <new-token>
```

| 项 | 说明 |
| --- | --- |
| JWT `sub` | 与生成文件的聊天使用相同用户 |
| 查询参数 `path` | `send_file.context` 原值，通过 HTTP 客户端进行 URL 编码 |
| `x-session-id` | 下载接口不需要此请求头，文件按用户工作区定位 |

```bash
curl --fail-with-body --get "http://localhost:7878/files" \
  -H "Authorization: Bearer <JWT>" \
  --data-urlencode 'path=outputs/report.pdf' \
  --output report.pdf
```

- 成功返回二进制文件，`Content-Disposition` 带建议文件名。
- 路径越出用户工作区返回 `403`，`detail` 为 `File is outside the workspace`。
- 文件不存在或目标不是文件时返回 `404`，`detail` 为 `File not found`。

同一用户的多个会话共享工作区；切换 `session_id` 不会改变文件可见范围。

### 6.3 给 Agent 上传文件

直接在 `/chat` 或 `/chat_sync` 中重复传入 `files` multipart 字段：

```bash
curl -N --fail-with-body -X POST "http://localhost:7878/chat" \
  -H "Authorization: Bearer <JWT>" \
  -H "x-session-id: 018f0000-0000-7000-8000-000000000001" \
  -H "x-sender-name: demo-user" \
  -F 'content=[{"text":"请分析这些附件"}]' \
  -F 'files=@report.pdf' \
  -F 'files=@chart.png'
```

文件保存在 `attachments/YYYY-MM-DD/<原文件名>`，重名自动加序号，Agent 可以直接在工作区读取。

---

## 7. 约束与错误码

### 7.1 并发与会话

| 场景 | 行为 |
| --- | --- |
| 同一用户、同一会话已有进行中的 `/chat` | 新的 `/chat` 请求返回 `409`，`detail` 为 `Session '<id>' already has an active chat request`；`/chat_sync` 不受此限制 |
| 同一通道的上一轮 `/chat` 结束后发下一条消息 | 正常，沿用该用户、会话的流式历史 |
| 同一用户换新会话 ID 或换通道 | 隔离流式历史，文件工作区与记忆仍共享 |
| 不同用户使用相同会话 ID | 按用户隔离 |
| `x-session-id` 缺失、空值或非规范 UUIDv7 | 返回 `422`（仅 `/chat`、`/chat/stop`、`/answer_question`；`/chat_sync` 不需要该头） |

请以响应结束作为释放本地会话队列的时机；收到 `token_usage` 后继续读取至 EOF，再发送下一轮。

### 7.2 错误码与中断处理

| 状态码 | 场景 | `detail` 示例或处理建议 |
| --- | --- | --- |
| `400` | 服务端不支持请求的 JSON 输出模式 | `The configured model does not support JSON output` |
| `401` | JWT 缺失、无效、过期，`kid` 不合法，用户 `sub` 缺失/非法，或 `/token` 凭据错误 | `Invalid or expired bearer token` / `Incorrect JWT secret`；修正身份或重新签发 |
| `403` | 文件路径越界，或非 Feishu 通道访问后台消息 | `File is outside the workspace` 等 |
| `404` | 下载文件不存在 | `File not found` |
| `409` | 同一用户、同一会话并发 `/chat` | 等待现有请求结束，按会话排队 |
| `422` | 必填头缺失、会话 ID 非法、用户名空白或非法 UTF-8、JSON 字符串非法等 | `Invalid session_id` / `Invalid JSON format` 等；修正参数 |
| `500` | 服务端或 Agent 执行异常，且尚未建立流式响应 | 记录响应和请求信息，按业务策略处理 |

SSE 建立前的 HTTP 错误通常返回：

```json
{"detail":"错误说明"}
```

SSE 建立后发生异常时，连接可能直接结束，不保证收到业务错误事件。接入方应单独处理网络、读取和解析异常；即使收到 EOF，也不能仅据此认定业务任务已成功。

聊天接口没有请求幂等键。连接中断后自动重放可能重复模型调用或工具操作，应按业务幂等策略决定是否重试。

### 7.3 长连接与部署

- HTTP 客户端的读取超时应关闭或设置为足够覆盖整个 Agent 运行；答题等待本身可长达 300 秒。
- 网关和反向代理需关闭响应缓冲，并设置足够的空闲超时，例如 Nginx 使用 `proxy_buffering off`。
- Compose 默认通过 Nginx 暴露 API。Nginx 读取 JWT `sub` 做一致性哈希，将同一用户路由到同一 backend；验签由 backend 完成。内部调用以 `x-user-id` 分流，无标识时轮询。
- 聊天运行状态和追问等待事件位于 backend 进程内存。自行部署多副本时需保持同一用户的请求路由一致；扩缩容、替换容器前应等待进行中的 SSE 结束。
- backend 默认放开 CORS 来源、方法、请求头及响应头读取。浏览器可携带短期 Bearer Token 调用；共享密钥仍只保存在可信后端。
- 对外部署使用 HTTPS，并按业务需要控制 API 网络入口。

---

## 8. 参考调用示例

### 8.1 curl：聊天、答题与下载

在可信后端环境中设置地址和通道凭据；每次业务请求前重新换取 Token：

```bash
BASE='http://localhost:7878'
CHANNEL='web_client'
SECRET='<jwt.channels.web_client.secret_key>'
USER_ID='demo-user'
SESSION_ID='018f0000-0000-7000-8000-000000000001'

get_token() {
  curl -sS --fail-with-body -X POST "$BASE/token" \
    --user "$CHANNEL:$SECRET" \
    --data-urlencode "username=$USER_ID" \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["access_token"])'
}

# 1) 发起聊天；中文用户名已做 UTF-8 URL 编码。
TOKEN=$(get_token)
curl -N --fail-with-body -X POST "$BASE/chat" \
  -H "Authorization: Bearer $TOKEN" \
  -H "x-session-id: $SESSION_ID" \
  -H 'x-sender-name: %E5%BC%A0%E4%B8%89' \
  -H 'Accept: text/event-stream' \
  -F 'content=[{"text":"帮我制定部署计划，缺少的信息请向我提问"}]' \
  -F 'session_metadata={"sender_id":"u-123"}'
```

如果出现 `question`，保持上面的连接打开，在另一个终端执行同样的变量与 `get_token` 定义后提交答案（字段名应以实际问题为准）：

```bash
TOKEN=$(get_token)
curl -sS --fail-with-body -X POST "$BASE/answer_question" \
  -H "Authorization: Bearer $TOKEN" \
  -H "x-session-id: $SESSION_ID" \
  --data-urlencode 'answer={"service_name":"order-api","environment":"prod","regions":["cn-shanghai"]}'
```

如果出现 `send_file`，使用事件中的实际路径下载：

```bash
TOKEN=$(get_token)
curl --fail-with-body --get "$BASE/files" \
  -H "Authorization: Bearer $TOKEN" \
  --data-urlencode 'path=outputs/report.pdf' \
  --output report.pdf
```

### 8.2 Python：完整追问闭环

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
JWT_KID = os.getenv("XPEECH_JWT_KID", "web_client")
USER_ID = "demo-user"
SESSION_ID = "018f0000-0000-7000-8000-000000000001"
SENDER_NAME = "张三"


def create_token() -> str:
    """每次 API 调用前签发新 Token，避免答题时复用过期 Token。"""
    return jwt.encode(
        {"exp": datetime.now(UTC) + timedelta(seconds=60), "sub": USER_ID},
        JWT_SECRET,
        algorithm="HS256",
        headers={"kid": JWT_KID},
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
    fields = question.get("fields", [])
    if not fields:
        return {"user_customization": await async_input("请输入回答: ")}

    for field in fields:
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
                    if field_type == "select" and len(indexes) != 1:
                        raise ValueError
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
        "x-session-id": SESSION_ID,
        "x-sender-name": quote(SENDER_NAME, safe=""),
    }
    data = {
        "content": json.dumps([{"text": text}], ensure_ascii=False),
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
                elif event_type == "tool_call":
                    for call_id, tool_name, arguments in json.loads(context):
                        print(f"\n[工具调用] {call_id} {tool_name}: {arguments}")
                elif event_type == "tool_call_result":
                    tool_result = json.loads(context)
                    print(
                        f"\n[工具结果] {tool_result['id']} {tool_result['name']} "
                        f"({tool_result['duration_seconds']:.2f}s): {tool_result['result']}"
                    )

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
export XPEECH_JWT_SECRET="<conf.toml 中的 jwt.channels.web_client.secret_key>"
export XPEECH_JWT_KID="web_client"
python xpeech_question_demo.py
```

生产环境通常不会在 Xpeech 接入进程中直接等待终端输入。更常见的流程是：接入后端收到 `question` 事件后，将 `context` 和自己的会话标识通过 WebSocket/SSE 转发给前端；前端提交表单到接入后端；接入后端再用已保存的 `user_id + session_id` 调用 Xpeech `/answer_question`。整个过程中，负责消费 `/chat` 的任务必须继续存活，不能提前关闭 SSE。

---

## 9. 接收飞书后台任务消息（可选）

该接口供自建飞书投递通道领取定时 Agent 任务结果，普通聊天接入无需调用。使用 `kid="feishu"`、含 `exp` 的通道 JWT，无需用户 `sub`：

```http
GET /background_message
Authorization: Bearer <feishu-channel-token>
```

接口最长约 50 秒长轮询。有消息返回 HTTP `200` 和 JSON，无消息返回 `204 No Content`：

```json
{
  "channel": "feishu",
  "open_id": "ou_xxx",
  "content": "后台任务执行结果",
  "schedule_type": "cron",
  "schedule_value": "0 9 * * *"
}
```

投递方根据 `open_id` 将 `content` 发送给飞书用户，然后重新获取 Token 并发起下一次长轮询。通道由 JWT 决定，无需 `channel` 查询参数；其他通道调用返回 `403`。

---

## 10. 接口清单速查

| 方法 | 路径 | 认证 | 说明 |
| --- | --- | --- | --- |
| GET | `/health` | 无 | 健康检查，纯文本 `ok` |
| GET | `/docs`、`/redoc`、`/openapi.json` | 无 | API 文档 |
| POST | `/token` | HTTP Basic 通道名与通道密钥 | 表单 `username` 指定用户，换取短期 JWT |
| POST | `/chat` | 通道 JWT，含用户 `sub` | 发送消息并返回 SSE 事件流 |
| POST | `/chat_sync` | 通道 JWT，含用户 `sub` | 非交互执行，一次性返回合并后的事件列表；服务端生成独立会话，不需要 `x-session-id`，不占会话并发锁 |
| POST | `/answer_question` | 通道 JWT，含同一用户 `sub` | 提交当前会话的追问答案 |
| POST | `/chat/stop` | 通道 JWT，含同一用户 `sub` | 停止指定会话的 SSE 生成，回收大模型并发名额 |
| GET | `/files?path=...` | 通道 JWT，含同一用户 `sub` | 下载用户工作区文件 |
| GET | `/background_message` | Feishu 通道 JWT，无需 `sub` | 领取后台任务消息，长轮询 |

接入前确认：JWT `kid/sub/exp` 与配置匹配，用户和会话在聊天、答题过程中保持一致；正确拼接 SSE 分片并读取至 EOF；答题使用独立请求，原流保持打开；文件路径经 URL 编码传递；代理关闭缓冲且超时足够；同一会话串行发送，不无条件重放中断的聊天请求。
