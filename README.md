<p align="center">
  <img src="xpeech-logo.png" alt="Xpeech Logo" width="220">
</p>

<h1 align="center">Xpeech</h1>

![Xpeech screenshot](screenshot/image.png)

Xpeech 是一个基于 FastAPI 的 Agent 服务。它提供一个 `/chat` 接口，可以接收文本、图片和文件，调用大模型生成流式回复，并在需要时调用工具完成任务。

适合用来快速启动一个可扩展的 AI Agent API 服务。

## 功能

- 提供 HTTP API 和 SSE 流式响应
- 支持文本、图片和文件输入
- 支持多轮会话和独立工作区
- 支持 LiteLLM 兼容的大模型服务
- 支持内置工具和自定义 Python 工具
- 支持通过 MCP Server 扩展 Agent 工具
- 支持飞书消息桥接
- 使用 `conf.toml` 统一管理应用配置和密钥
- YAML 格式存储会话历史，可读性更好
- 自动历史消息压缩（三级压缩策略），避免超出上下文限制
- 内置记忆系统，自动总结和保存关键信息
- 支持视频输入
- Token 使用率实时监控
- 丰富的内置工具集：文件读写、Shell 执行、Web 搜索与网页抓取、`agent-browser` 浏览器自动化、Office 文档读取、文件发送、向用户提问

## 安装

需要 Python 3.12+ 和 uv。项目依赖通过 `uv sync` 安装：

```bash
uv sync
```

Shell 工具只支持 Linux，并依赖 bubblewrap 沙盒。Debian/Ubuntu 可这样安装：

```bash
sudo apt-get install bubblewrap
```

提前安装内置技能依赖：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

npm config set registry https://registry.npmmirror.com/
```

Docker 镜像已安装 `agent-browser`。Compose 通过 Browserless Chromium 容器提供 CDP 服务，Xpeech 不会在 backend 容器内安装、启动或管理本地浏览器。

## 配置

普通配置写在 `conf.toml`：

```toml
[path]
session_path = "data/session"
session_history_path = "data/session/history"
workspace_base_path = "data/workspace_base"
sandbox_home_path = "data/sandbox-home"
cache_path = "data/cache"

[llm]
api_key = "your_api_key_here"
api_base = "https://api.siliconflow.cn/v1"
default_model = "openai/Pro/moonshotai/Kimi-K2.6"
default_context_token = 256000
default_top_p = 0.7
tools_python_package = "custom_tools"
default_tools = ["echo", "hello"]
system_name = ""
custom_system_prompt = ""
# default_reasoning_effort = "normal"
support_image = true
support_video = true
support_json_output = true
parallel = 4
max_iterations = 40

[tool]
[tool.browser_preview]
browser_preview_base_url = "http://backend:7878/browser_preview"
browser_preview_path = "data/browser_preview"

[tool.mcpServers.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
# env = { DATABASE_URL = "postgres://user:pass@localhost:5432/db" }
enabled_tools = ["*"]
tool_timeout = 30

# [tool.mcpServers.my-api]
# url = "https://mcp.example.com/sse"
# headers = { Authorization = "Bearer xxx" }
# enabled_tools = ["*"]
# tool_timeout = 120

[feishu]
app_id = "cli_xxx"
app_secret = "your_feishu_app_secret_here"
idle_timeout = 3
```

可以从模板创建本地配置：

```bash
cp conf.toml.exmple conf.toml
```

进程环境变量写在 `.env`，例如 PPT 导出脚本使用的远程 CDP 地址：

```env
CDP_URL=ws://browserless:3000
```

## 启动

启动 API 服务：

```bash
uv run -m xpeech api
```

如果不指定服务，默认也是启动 API：

```bash
uv run -m xpeech
```

服务默认运行在：

```text
http://localhost:7878
```

启动后可以打开：

- Swagger UI: `http://localhost:7878/docs`
- ReDoc: `http://localhost:7878/redoc`

启动飞书桥接：

```bash
uv run -m xpeech feishu
```

飞书桥接会从配置中读取：

- `feishu.app_id`：飞书应用 ID
- `feishu.idle_timeout`：同一会话消息合并等待时间，单位秒
- `feishu.app_secret`：飞书应用密钥

如需连接非默认 API 地址，可以传入 API 基地址：

```bash
uv run -m xpeech feishu --chat-url http://127.0.0.1:7878
```

## Docker Compose 部署

Compose 会启动三个容器：

- `browserless`：Browserless Chromium CDP 服务，宿主机通过 `http://localhost:3000/docs` 查看文档
- `backend`：Xpeech API、Agent 和工具执行服务
- `feishu`：飞书长连接桥接服务，通过 Docker 内网访问后端

先准备配置和环境变量：

```bash
cp conf.toml.exmple conf.toml
cp .env.example .env
```

填写 `conf.toml` 中的 `llm.api_key` 和 `feishu.app_secret`，并确认
`.env` 中的 `CDP_URL` 与容器网络一致，再确认
`conf.toml` 中的 `llm`、`feishu.app_id` 等普通配置正确，然后构建并启动：

```bash
docker compose up -d --build
```

查看运行状态和日志：

```bash
docker compose ps
docker compose logs -f browserless backend feishu
```

后端默认暴露在 `http://localhost:7878`。如需修改宿主机端口，请修改
`compose.yaml` 中 `backend.ports` 的宿主机端口。持久化数据统一映射到宿主机
的 `./docker_data/` 目录，其中包含 `session`、`workspace_base`、`sandbox-home` 和
`browser_preview`；缓存目录不做宿主机磁盘映射。`conf.toml` 以只读方式挂载，`.env`
通过 `env_file` 注入进程，修改后重建容器即可生效：

```bash
docker compose up -d --force-recreate browserless backend feishu
```

## 发送消息

`/chat` 需要通过请求头传入会话 ID：

```bash
curl -N -X POST "http://localhost:7878/chat" \
  -H "x-session-id: demo-session" \
  -F 'session_metadata={"channel":"curl"}' \
  -F 'content=[{"text":"你好，介绍一下你自己"}]'
```

上传文件：

```bash
curl -N -X POST "http://localhost:7878/chat" \
  -H "x-session-id: demo-session" \
  -F 'session_metadata={"channel":"curl"}' \
  -F 'content=[{"text":"帮我看看这个文件"}]' \
  -F "files=@example.txt"
```

响应是 SSE 流，可以边生成边读取。

### 内置命令

在聊天中输入以下命令可以使用快捷功能：

- `/help` - 显示帮助信息
- `/new` - 开始一个新会话，自动总结并保存当前会话记忆

## 自定义工具

在 `conf.toml` 里指定工具包：

```toml
[llm]
tools_python_package = "custom_tools"
default_tools = ["echo", "hello"]
```

工具包示例：

```text
custom_tools/
  __init__.py
  test_tools.py
```

`custom_tools/__init__.py`：

```python
from .test_tools import echo, hello

__all__ = ["echo", "hello"]
```

`custom_tools/test_tools.py`：

```python
from typing import Annotated

from pydantic import BaseModel, Field

def hello():
    """Return a hello message."""
    return "hello"


class Message(BaseModel):
    content: Annotated[str, Field(description="The content to echo")]


def echo(message: Message):
    """Echo the message content."""
    return message.content

```

工具函数需要有 docstring。函数可以不接收参数，也可以接收一个 Pydantic `BaseModel` 参数。

## 浏览器自动化

浏览器自动化通过内置 `agent-browser` 技能完成。Compose 内的 backend 通过
`ws://browserless:3000` 连接 Browserless；宿主机上对应的地址是 `ws://localhost:3000`，
文档页为 `http://localhost:3000/docs`。Agent 使用 Shell 执行
`agent-browser` 命令时，执行层会自动追加当前请求的 `--session` 和配置的
`--cdp` 参数。

模型在首次进行浏览器操作前会加载
`xpeech/agent/skills/buildin/agent-browser/SKILL.md`，并按其约束复用注入的
CDP 连接和 session。Xpeech 不提供本地浏览器回退；CDP 连接失败时会直接报错。

## MCP 工具

可以在 `conf.toml` 的 `[tool.mcpServers.<name>]` 下配置 MCP Server。启动会话时，Xpeech 会连接这些 Server、发现可用工具，并把它们注册为 Agent 默认工具。

stdio Server 示例：

```toml
[tool.mcpServers.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
enabled_tools = ["*"]
tool_timeout = 30
```

远程 MCP Server 示例：

```toml
[tool.mcpServers.my-api]
url = "https://mcp.example.com/sse"
headers = { Authorization = "Bearer xxx" }
enabled_tools = ["search", "read_record"]
tool_timeout = 120
```

每个会话都会把当前用户 workspace 作为 MCP workspace root。stdio MCP
进程同时以该目录作为 `cwd`；HTTP/SSE MCP 通过标准 `roots/list` 获取同一目录。
因此 MCP、内置文件工具和 Shell 的相对路径基准保持一致。远程 MCP 服务若需
直接读写文件，必须能以相同绝对路径访问该目录。

普通搜索和网页文本抓取使用 `web_search` 和 `web_fetch`；需要浏览器渲染或
交互操作时使用 `agent-browser`。`browser_preview_base_url` 只负责生成
`agent-browser` 能访问的 URL 前缀，FastAPI 路由由该 URL 的 path 部分自动注册；
`browser_preview_path` 只负责存放预览文件，两者没有
路径推导关系。`create_browser_preview` 会把目录复制到 `<browser_preview_path>/<uuid>/`；
传入目录时返回该 UUID 目录的 URL 前缀；传入单个 HTML 时保留源文件名，
并返回完整的文件 URL。

字段说明：

- `command` / `args`：启动 stdio MCP Server 的命令和参数。
- `url`：连接远程 MCP Server。`/sse` 结尾的地址使用 SSE transport，其他地址默认使用 streamable HTTP。
- `env`：stdio Server 的环境变量。
- `headers`：远程 MCP Server 的请求头。
- `enabled_tools`：允许注册的 MCP 工具名，`["*"]` 表示全部注册。
- `tool_timeout`：单次 MCP 工具调用超时时间，单位秒。

`command` 和 `url` 只能二选一。MCP Server 配置会按原样传入，不会做运行时字符串替换。

注册后的工具名会加上 `mcp_<server>_` 前缀，例如 `filesystem` Server 暴露的 `read_file` 会注册为 `mcp_filesystem_read_file`。`enabled_tools` 可以填写 MCP 原始工具名，也可以填写加前缀后的工具名。

## 内置工具

Xpeech 提供丰富的内置工具，Agent 可以在对话中自动调用：

| 工具 | 说明 |
|------|------|
| `read_file` | 读取工作区或内置技能目录中的文件内容 |
| `write_file` | 向工作区写入文件 |
| `edit_file` | 编辑工作区内的文件 |
| `list_dir` | 列出工作区或内置技能目录内容 |
| `shell` | 执行 Bash 命令（带工作区路径限制和 bwrap 沙盒） |
| `web_fetch` | 抓取网页内容并转为 Markdown |
| `web_search` | 搜索网页并返回结果 |
| `create_browser_preview` | 将工作区内目录或单个 HTML 复制到 UUID 预览目录，返回目录 URL 前缀或完整文件 URL |
| `shell` + `agent-browser` | 通过注入的 CDP 连接搜索、打开、读取、检查和操作网页 |
| `office_read` | 读取 Office 文档（docx/xlsx/pdf/pptx 等） |
| `send_file` | 向用户发送工作区或内置技能目录中的文件 |
| `ask_user_question` | 向用户发送表单提问 |

### 工具安全

- **统一路径解析**：文件工具、文件发送工具和 Shell 绝对路径检查统一通过 `xpeech/agent/tools/helper.py` 解析路径；默认限制在工作区内，读取类操作可额外访问内置技能目录，写入和编辑仍只能落在工作区内
- **预览目录隔离**：`create_browser_preview` 只能复制当前会话工作区内的目录或 HTML，每次写入 `<browser_preview_path>/<uuid>/`，请求子路径不能越出对应 UUID 目录
- **Shell 沙盒**：Shell 命令通过 bubblewrap 运行，工作区可读写，内置技能目录只读挂载，系统运行时路径只读挂载，临时目录和工作区父目录使用 tmpfs 隔离
- **依赖安装隔离**：Shell 工具首次使用时自动创建 `<workspace>/.venv`；项目 Python 依赖进入当前工作区虚拟环境，`uv tool install` 和 `npm install -g` 安装到共享沙盒 HOME，便于不同会话复用 CLI 工具
- **Shell 黑名单**：禁止执行 `rm -rf`、`format`、`dd` 等危险命令
- **路径遍历检测**：拦截包含 `..` 的路径操作
- **内网 URL 拦截**：防止访问内部网络接口

### 沙盒

Shell 命令会在 bwrap 进程沙盒中执行：

- 每个会话工作区会绑定为可读写目录，并作为命令工作目录。
- 内置技能目录以只读方式挂载，便于读取技能脚本和资源。
- `/tmp` 和工作区父目录会使用临时文件系统隔离，避免命令看到其他工作区。
- `/usr`、`/bin`、`/lib`、证书和 DNS 配置等系统路径以只读方式挂载，提供基础运行环境。
- `HOME` 会指向工作区父目录下的共享沙盒目录 `.xpeech-sandbox-home`，并挂载为可读写目录。
- `PATH` 会优先包含共享沙盒 HOME 下的 `.local/bin` 和 `.npm-global/bin`，因此 `uv tool install`、`npm install -g` 安装的命令可被后续会话直接找到。

Shell 工具首次运行时会在工作区内执行 `uv venv .venv`，为该工作区创建独立 Python 环境。
Python 命令必须通过 `uv run python ...` 启动，直接执行 `python` / `pip` 会被安全检查拦截。
沙盒内设置了 `PIP_REQUIRE_VIRTUALENV=true` 和 `UV_PROJECT_ENVIRONMENT=<workspace>/.venv`，确保项目 Python 依赖安装进入当前工作区的 `.venv`。
同时设置 `UV_CACHE_DIR=<sandbox-home>/.cache/uv` 和 `NPM_CONFIG_PREFIX=<sandbox-home>/.npm-global`，让 uv 缓存、uv tool 工具和 npm 全局工具在沙盒 HOME 内共享。

如果 `npx` 来自 nvm 或真实用户 HOME 中的 Node 安装，沙盒默认看不到它；需要把 Node/npm/npx 安装到 `/usr`、`/bin`、`/opt` 等沙盒可见路径，或在沙盒内通过可见的 npm 安装全局工具。

## 开发

```bash
uv sync
uv run pytest
uv run -m xpeech api
```

如果需要检查配置是否能读取：

```bash
uv run python -c "from xpeech.config.settings import settings; print(settings.model_dump())"
```

## TODO

- [ ] 添加 cron
- [ ] 添加心跳
- [ ] 添加飞书 CLI
