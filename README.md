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
- 支持飞书一次性和周期后台定时 Agent 任务
- 使用 `conf.toml` 统一管理应用配置和密钥
- YAML 格式存储会话历史，可读性更好
- 自动历史消息压缩（三级压缩策略），避免超出上下文限制
- 内置记忆系统，自动总结和保存关键信息
- 支持视频输入
- Token 上下文使用率与本轮输入、输出、总消耗统计，支持 `k` / `m` 单位，未返回的用量按 `0` 累计
- 丰富的内置工具集：文件读写、Shell 执行、Web 搜索与网页抓取、HTML 预览、Office 文档读取、文件发送、向用户提问

## 安装

需要 Python 3.12+ 和 uv。项目依赖通过 `uv sync` 安装：

```bash
uv sync
```

Shell 工具和 stdio MCP 只支持 Linux，并依赖 bubblewrap 沙盒。Debian/Ubuntu 可这样安装：

```bash
sudo apt-get install bubblewrap slirp4netns
```

Shell 沙盒按用户工作区复用：首次调用时创建，同一用户的不同会话共享沙盒中的 `/tmp`、网络和后台进程。
同一用户的多个 shell 命令可在同一沙盒内并发执行，输出、超时和取消按请求独立处理。
每次命令使用独立 bash，并从工作区目录开始执行。正在运行或等待沙盒启动的请求会保持沙盒存活；最后一个请求结束后，
连续空闲达到 `tool.shell.sandbox_idle_timeout_seconds`（默认 600 秒，即 10 分钟）便自动销毁沙盒及其后台进程。
之后再次调用会创建新沙盒，工作区磁盘文件仍然保留。
服务关闭时也会清理所有 shell 沙盒。沙盒异常退出或切换飞书令牌认证模式时会重新创建。

可在 `conf.toml` 中调整空闲回收时间，修改后重启服务生效：

```toml
[tool.shell]
sandbox_idle_timeout_seconds = 600
```

Shell 危险命令检查还需要安装 [dcg](https://gitee.com/luojiaaoo/destructive_command_guard/releases/tag/v0.16.0)
并加入 backend 的 `PATH`。Docker 镜像已包含 dcg，无需额外安装。

提前安装内置技能依赖：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

npm config set registry https://registry.npmmirror.com/
```

## 配置

从模板创建本地配置：

```bash
cp conf.toml.example conf.toml
```

进程环境变量写在 `.env`。模板中包含 Compose 项目名和可选的服务端口配置。

内部服务互信与各通道认证使用彼此隔离的短期 JWT。先分别生成三个不同的密钥：

```bash
# 连续运行三次，分别保存输出
uv run python -c "import secrets; print(secrets.token_hex(32))"
```

```toml
[jwt]
# 仅供 backend、scheduler、Web 统计代理等内部服务互信
secret_key = "replace-with-an-internal-random-value"
algorithm = "HS256"
access_token_expire_seconds = 60

[jwt.channels.web_client]
secret_key = "replace-with-a-different-web-random-value"

[jwt.channels.feishu]
secret_key = "replace-with-a-different-feishu-random-value"
```

三组密钥至少 32 个字符、必须互不相同；算法和有效期在 `[jwt]` 下统一配置，
默认 HS256、60 秒，有效期最长 60 秒。Compose 中的服务读取同一份 `conf.toml`，使用各自的密钥。
内部 token 不携带 `kid`，直接使用 `secret_key` 验签。channel token 的 header
使用 `kid` 从 `jwt.channels` 选择同名通道的 `secret_key`，包括 `web_client` 和 `feishu`。
聊天、追问、文件、技能接口必须使用同时带 channel `kid` 和用户 `sub` 的 JWT；
后台消息接口只需带 `kid` 的 channel JWT。对外接口忽略 `x-user-id` 请求头；
`/statistics` 及其子接口、`/internal/scheduled-run` 和 scheduler
服务的接口使用内部 JWT，通过 `jwt.secret_key` 验签。

自定义系统接入可在 `jwt.channels` 下配置独立通道，通道名就是 JWT header 的 `kid`：

```toml
[jwt.channels.crm]
secret_key = "replace-with-a-distinct-crm-secret-of-at-least-32-characters"
channel_tag = "crm"
```

所有通道密钥至少 32 个字符，必须与内部密钥及其他通道密钥互不相同；通道名不能使用
`internal`。所有通道使用统一的 JWT 算法和有效期，支持业务接口，
也可通过 `/token` 使用通道名和对应密钥获取令牌；统计接口和内部接口使用内部 JWT。
每个通道只绑定一个 `channel_tag`，由服务端配置确定，JWT payload 或会话元数据中的标签不授予任何权限。
省略通道 `channel_tag` 时，渠道筛选允许公共内建技能、公共 MCP，以及用户自己的技能；这些资源还需通过下面的组织筛选。

Skill 的 YAML frontmatter 和 MCP 配置都支持 `channel_tags`。标签筛选仅作用于内建技能和 MCP，
用户工作区里的技能不受渠道标签匹配限制。内建技能和 MCP 省略 `channel_tags` 或配置 `[]` 表示公共资源；
非空表示专属资源，仅向至少命中一个标签的 JWT 通道展示或注册。标签区分大小写，
长度为 1–64 个字符，以英文字母或数字开头，可包含字母、数字、`.`、`_`、`:`、`-`。
无需给 Skill 名称增加特殊前缀。例如：

```yaml
---
name: crm-report
description: CRM 报表助手
channel_tags: [crm]
---
```

```toml
[tool.mcpServers.crm-api]
url = "https://crm.example.com/mcp"
channel_tags = ["crm"]
```

标签筛选作用于内建技能的列表、系统提示词（含 `always` 技能）和 MCP 工具注册；
它不是工作区文件的访问控制。同一 `user_id` 的工作区和历史仍然共享；需要系统间数据隔离时应使用独立的用户 ID。
鉴权后的 `channel_tag` 以单个字符串或 `None` 显式传递到 Runner、技能加载和工具注册。MCP 在注册阶段按标签筛选，不匹配时不会建立连接；
同步聊天和定时任务也在注册阶段跳过追问、创建/查看/取消调度等交互式工具。

Skill 的 YAML frontmatter 和 MCP 配置还支持 `organization_prefixes`，按用户的组织字符串筛选：

```toml
[organization]
command_template = "curl --fail --silent 'https://directory.example/organization?user_id=${user_id}'"
timeout = 10.0

[tool.mcpServers.engineering]
url = "https://engineering.example.com/mcp"
organization_prefixes = ["公司/研发/"]
```

```yaml
---
name: engineering-report
description: 研发报表助手
organization_prefixes: ["公司/研发/", "公司/平台/"]
---
```

查询命令在服务端执行，必须包含 `${user_id}` 或 `$user_id`，使用已鉴权的用户 ID，
不经过 shell；命令成功退出后的 UTF-8 stdout 去除首尾空白后作为组织字符串。
每次创建 Runner 时先查询一次，同一结果用于技能提示词和 MCP 注册；`GET /skills` 也先查询当前用户的组织。
组织不从 JWT 额外字段、渠道配置或会话元数据读取，也不跨用户或跨 Runner 缓存。
`organization_prefixes` 省略或为 `[]` 表示不限制组织；非空时，组织字符串必须以列表中任一前缀开头，
区分大小写，按字面字符串匹配，不解析通配符或自动补充分隔符。前缀必须是无首尾空白的非空字符串，支持中文。
例如 `公司/研发/平台` 匹配 `公司/研发/`，不匹配 `公司/销售/`。
未配置查询命令、命令失败、超时（默认 10 秒）或输出为空时，只加载未配置组织前缀的资源。
组织筛选适用于内建技能、工作区技能（含 `always` 技能）、技能列表 API，以及 MCP 状态、用途和工具注册；
同时配置 `channel_tags` 时，两种条件都必须满足。未匹配的 MCP 不会发起连接。
与渠道筛选一样，组织筛选控制资源加载，不是工作区文件的访问控制。

`not_interactive=True` 的同步聊天和定时任务也会从技能列表及系统提示词中跳过
`ask-user-question`，即使它标记为 `always: true`。交互式技能名称在代码中维护，
无需在 Skill 中添加标记；普通交互会话仍正常加载。

## 启动

按需在独立终端启动各服务：

```bash
# API 服务
uv run -m xpeech api

# API 服务（省略服务名时的等价写法）
uv run -m xpeech

# 调度服务（单实例；到期后通过 HTTP 请求 API 执行）
uv run -m xpeech scheduler

# Web 客户端（依赖 API 服务）
uv run -m xpeech web_client

# 飞书桥接（依赖 API 服务）
uv run -m xpeech feishu

# Lark 令牌管理
uv run -m xpeech token_manager
```

默认访问地址：

- API：`http://localhost:7878`
- Web 客户端：`http://localhost:7939`
- Swagger UI：`http://localhost:7878/docs`
- ReDoc：`http://localhost:7878/redoc`

Swagger UI、ReDoc 及 `/openapi.json` 可直接访问。在
Swagger UI 右上角点击 `Authorize`：`client_id` 填写 channel（如 `web_client` 或 `feishu`），
`client_secret` 填写对应的 `jwt.channels.<channel>.secret_key`，
`Client credentials location` 保持 `Authorization header`；`username` 和 `password` 留空。
Swagger 会调用 `/token` 校验密钥并签发带 channel `kid`、有效期
由 `jwt.access_token_expire_seconds` 决定的 JWT，后续调试请求会自动携带该令牌。部署时必须使用 HTTPS，避免共享密钥在
传输过程中泄露。

飞书桥接会从配置中读取：

- `feishu.app_id`：飞书应用 ID
- `feishu.app_secret`：飞书应用密钥

Docker 镜像会从 `https://gitee.com/luojiaaoo/lark-cli` 的 `v1.0.89` 标签编译
`lark-cli` 和非阻塞的 `/usr/local/bin/lark-cli-auth`。独立的 `lark-token-manager` 服务负责设备授权、
令牌加密存储和周期刷新；`lark-cli` 通过内部接口取得当前 Xpeech session 的短期令牌。
`app_id`、`app_secret` 只由 lark-token-manager 在运行时读取，修改后重启服务即可。
backend 通过环境变量 `XPEECH_TOKEN_MANAGER_URL` 获取 lark-token-manager 的内部地址。

已有用户访问令牌时，通过环境变量 `LARKSUITE_CLI_USER_ACCESS_TOKEN` 注入即可，
用户身份命令会直接使用该令牌，跳过 lark-token-manager 的令牌获取、迁移和失效通知。
此时无需 `SHELL_USER_ID`，令牌过期后由调用方替换：

```bash
LARKSUITE_CLI_USER_ACCESS_TOKEN="<lark_user_access_token>" lark-cli docs +fetch --doc "<文档 URL 或 token>" --as user
```

未提供令牌时使用下述托管认证流程。修改 CLI 后需重新构建 Docker 镜像才能生效。

首次执行需要用户身份的命令时，Credential Provider 会立即返回 `authorization_required` 和授权 URL，
不会阻塞命令。也可以先执行 `lark-cli-auth login --scope "<本次任务所需 scope>"` 或
`lark-cli-auth login --domain calendar,task` 获取链接；`login` 至少需要一个非空的
`--scope` 或 `--domain`，两者可以组合。用户在浏览器完成设备授权后重跑原命令，
lark-token-manager 会按 session 绑定飞书用户。令牌数据库持久化在 `data/lark-token-manager`，
后台任务每分钟扫描并在过期前刷新，refresh token 按单次轮换原子保存。
飞书授权使用设备码流程，无需配置 `redirect_uri` 或暴露 callback 端口。lark-token-manager 使用飞书
返回的 device code 有效期在后台轮询；lark-token-manager 只在 Xpeech 内部网络提供 session 级接口。
访问令牌固定在过期前 5 分钟刷新，manager 每 60 秒扫描一次，这两项无需配置。

业务命令首次触发透明授权时只建立 `offline_access`；业务权限必须通过
`lark-cli-auth login` 显式传入本次任务所需的 `--scope` 或 `--domain`。
需要增加权限时可单独执行 `lark-cli-auth`，`--scope` 可以重复，
也可以用逗号或空格一次传入多个 scope；`--domain` 使用原生 lark-cli 的业务域目录，
可传逗号/空格分隔的多个域或 `all`；已有授权范围由飞书授权服务维护：

```bash
lark-cli-auth login --scope docs:doc:readonly --scope drive:drive:readonly
lark-cli-auth login --scope "docs:doc:readonly,drive:drive:readonly"
lark-cli-auth login --domain calendar,task
```

需要修改监听地址、端口或后端地址时，可通过对应命令的 `--help` 查看参数。

## 快速部署平台

如果需要通过可视化界面快速部署和集中管理一个或多个 Xpeech 实例，可以使用
[Xpeech Deck](https://github.com/luojiaaoo/xpeech-deck)。它会为每个实例维护独立的 Xpeech Git
工作树和 Docker Compose 环境，并提供：

- 创建 Xpeech 实例，自动更新并在远程分支、标签和近期提交之间切换版本
- 在线配置 Backend / Web Client 端口、`conf.toml` 和自定义内置技能
- 执行 Up、Start、Stop、Restart、Down 和 PS 等常用 Docker Compose 操作
- 检查和拉取 Xpeech 所需镜像，通过 System Console 查看命令及执行结果

![Xpeech Deck 多实例管理平台](screenshot/PixPin_2026-08-25_10-40-18.jpg)

Xpeech Deck 是自托管平台，需要与 Docker 和所有受管 Xpeech 实例运行在同一台机器上。
安装、配置和访问方式请参阅 [Xpeech Deck 使用说明](https://github.com/luojiaaoo/xpeech-deck#readme)。

## 项目结构

配套服务和接入渠道统一放在 `xpeech/subapp/`，Agent 核心仍位于 `xpeech/agent/`：

```text
xpeech/
├── agent/                   # Agent、工具、后端 API 和共享 HTTP 协议/客户端
├── subapp/
│   ├── channel/             # 飞书、Web 客户端和共享通道逻辑
│   ├── lark_token_manager/  # Lark 设备授权与令牌管理
│   └── scheduler/           # 定时任务 API、后台调度和结果队列
├── config/
├── provider/
└── utils/
```

依赖方向为 `subapp → 核心`；`agent`、`config`、`provider`、`utils` 等核心代码不导入
`subapp`。调度工具、共享协议和 HTTP 客户端统一位于 `agent/tools/schedule/`，
后台 Agent 执行位于 `agent/background.py`。
所有服务统一使用 `uv run -m xpeech <服务名>` 启动；不指定服务名时默认启动 API。
Compose 已配置相同入口。

## Docker Compose 部署

Compose 会启动六类容器：

- `lark-token-manager`：飞书用户令牌管理服务
- `backend`：Xpeech API、Agent 和工具执行服务，默认启动 16 个副本
- `scheduler`：单实例调度服务，独占 `schedule.db`，通过 Nginx 调用 backend 执行到期任务
- `nginx`：后端 API 入口，按 JWT `sub` 将同一用户稳定分配到同一 backend
- `feishu`：飞书长连接桥接服务，通过 Docker 内网访问后端
- `web_client`：Web 客户端及认证代理，默认暴露在 `http://localhost:7939`

Nginx 使用普通 `hash $user_id` 直接选择 DNS 解析出的副本。这里不能使用
`consistent`：所有副本共用 `backend:8080` 名称时，它仍会在这些 IP 间轮询，
导致 `/chat` 和 `/answer_question` 落到不同进程，无法唤醒等待中的问答。
等待答案的状态保存在进程内存中；扩缩容或重建 backend 可能改变用户分配，
应在进行中的对话结束后操作。

先准备配置和环境变量：

```bash
cp conf.toml.example conf.toml
cp .env.example .env
```

填写 `conf.toml` 中的 `llm.api_key`、三组 JWT 密钥和 `feishu.app_secret`，
并确认 `feishu.app_id` 等普通配置正确，然后构建并启动：

```bash
docker compose up -d --build
```

`backend` 的沙盒需要 `SYS_ADMIN`（命名空间和挂载）、`NET_ADMIN`（TAP 网卡和
loopback 配置）以及 `/dev/net/tun` 设备映射，Compose 已配置。如果旧容器运行 Shell 时出现
`ioctl(TUNSETIFF): Operation not permitted` 或 `bwrap: loopback: Failed RTM_NEWADDR`，
更新 Compose 配置后执行以下命令，使权限变更生效（仅 restart 不会更新容器权限，无需重建镜像）：

```bash
docker compose up -d --no-deps --force-recreate backend
```

后端默认暴露在 `http://localhost:7878`，Web 客户端默认暴露在
`http://localhost:7939`。可通过环境变量 `BACKEND_REPLICAS` 修改后端副本数，
通过 `BACKEND_PORT` 和 `WEB_CLIENT_PORT` 修改对外端口；lark-cli 使用设备授权流程，
不依赖这两个端口：

```bash
BACKEND_REPLICAS=16 BACKEND_PORT=8080 WEB_CLIENT_PORT=8081 docker compose up -d --build
```

Nginx 按 JWT `sub`（对外请求）或 `x-user-id`（内部请求）做一致性哈希分流；两者都没有时正常轮询。
JWT 验签由 backend 负责。
`/chat`、`/answer_question`、文件和技能请求必须使用用户 JWT，且 `sub` 保持一致；对外接口忽略 `x-user-id` 请求头。
内部 JWT 仅用于服务鉴权，用户参数通过 `x-user-id` 直接传递。调整副本数或替换 backend 容器可能改变用户的映射，操作前应先等待
正在进行的 SSE 会话结束。

查看运行状态和日志：

```bash
docker compose ps
docker compose logs -f lark-token-manager backend scheduler nginx feishu web_client
```

持久化数据统一映射到宿主机
的 `./data/` 目录，其中包含 `session`、`workspace_base` 和
`browser_preview`，Web 用户数据库保存在 `web_client/users.db`；缓存目录不做宿主机磁盘
映射。`conf.toml` 以只读方式挂载，`.env` 通过 `env_file` 注入进程。普通运行时配置修改后重建容器
即可生效；`feishu.app_id` 或 `feishu.app_secret` 修改后需要增加 `--build` 重新编译 lark-cli 和
lark-cli-auth：

```bash
docker compose up -d --force-recreate lark-token-manager backend scheduler nginx feishu web_client

# lark-cli 构建配置发生变化时
docker compose up -d --build --force-recreate backend
```

后台定时任务由单实例 `scheduler` 管理，backend 使用内部 JWT 通过 HTTP 创建、查询和取消任务。
到期后 scheduler 携带内部 JWT 和 `x-user-id` 请求 `http://nginx:7878/internal/scheduled-run`，
由用户对应的 backend 使用 `background=True` 执行，结果返回 scheduler 供飞书轮询。
`/background_message` 经 Nginx 到达任意 backend，backend 校验通道 JWT 并限制仅 Feishu 通道可领取，
再使用内部 JWT 向 scheduler 领取消息，不需要绑定特定实例。
Nginx 对所有 API 使用同一条哈希转发规则。

数据库使用 `data/session/schedule.db`。scheduler 以单实例运行，独占该数据库。
调度期间需保持 Nginx/backend 可用，执行 HTTP 请求失败时会发送失败提示，不自动重试；
停机期间错过的任务会被跳过。待发送结果保存在 scheduler 内存中，重启会丢失未发送结果。

本地运行 API 时，也需要在另一个终端执行 `uv run -m xpeech scheduler`。
API 使用 `--scheduler-url` 指定 scheduler 地址，scheduler 使用 `--backend-url` 指定 backend
网关地址；两者默认分别为 `http://127.0.0.1:8090` 和 `http://127.0.0.1:7878`。

## 发送消息

需要从自有业务后端、机器人通道或其他服务完整接入聊天、SSE 事件、追问回调和文件下载时，
请参阅 [后端接入说明](docs/backend-integration.md)。

需要将现成聊天界面嵌入业务网站时，请参阅 [iframe 嵌入接入文档](docs/iframe-integration.md)。

`/chat` 需要带用户 `sub` 的 Bearer JWT，并通过 `x-sender-name` 请求头传入发送者用户名。
用户 ID 只取已验签 JWT 的 `sub`；用户名包含非 ASCII 字符时，可以使用 UTF-8 URL 编码。

`user_id` 是用户的稳定标识（例如飞书工号），用于隔离工作区、技能和 MCP 连接；`session_id`
标识该用户的一次会话，用于隔离上下文、并发聊天和提问回答。同一个用户可以同时拥有多个会话。
所有通道均必须传入非空请求头 `x-session-id`，只允许 `default`（主会话）或规范的小写连字符 UUIDv7；
省略、空值或其他格式均返回 HTTP 422。
仅 `session_id` 为 `default` 时，向模型提供后台任务工具
`feishu_schedule`、`feishu_schedule_list` 和 `feishu_schedule_cancel`。
对外用户身份只使用 JWT `sub`；仅内部请求通过 `x-user-id` 传递用户。代码字段使用 `user_id`，Shell 环境变量使用 `SHELL_USER_ID`，
OAuth2 用户标识配置使用 `user_id_claim`。

会话历史保存在 `session_history_path` 目录中，文件名固定为
`user_id=[user_id]&channel=[channel]&type=session&session_id=[session_id].yaml`（各值使用 URL 编码）。
后台任务每次运行的上下文也保存在该目录，文件名为
`user_id=[user_id]&channel=[channel]&type=not_interactive&session_id=[session_id].yaml`。后台任务每次执行时生成
UUIDv7 格式的 `session_id`，保存历史时直接复用此 ID；执行时间以文件修改时间为准。
`llm.max_history_files_per_user` 控制每个用户最多保留的历史文件数。加载历史时会按修改时间清理超额文件，
并删除历史目录中不符合上述命名格式的文件。
模型可感知当前 `session_id`，并通过 `list_sessions` 和 `read_session` 分页查看当前用户已保存的
普通会话及后台执行历史。历史查询只读，不会切换当前会话或自动合并其他会话的上下文。

同一会话同时只允许一个 `/chat` 请求，重复请求返回 HTTP 409；同一用户的不同会话可以并发运行，
并共享 workspace。文件读写和锁统一由 `xpeech/utils/file_io.py` 管理，同一路径的单次读写互斥，
覆盖写入通过同文件系统中的临时文件原子替换，读取会等待正在进行的受保护写入完成。
普通工作区文件在读取、处理和写回之间不持续持锁，并发编辑仍可能互相覆盖。
同名附件会在上传完成后原子分配名称，避免相互覆盖；
工作区模板只在文件不存在时创建。长期记忆整理按 `memory/MEMORY.md` 路径串行执行，
锁覆盖读取旧记忆、模型整理和保存新记忆，历史摘要追加也按文件串行执行。

上述锁适用于当前单进程 backend（`workers=1`）。Shell、MCP 和自定义工具直接操作文件时，
需要自行协调并发；同一文件的完整覆盖写入仍以后写入的内容为准。多会话同时处理文件时，
建议使用不同的输出路径。

代码中单独调用 `LockedFile(path).read_text()`、`write_text()`、`append_text()` 等方法时会自动
加锁。同一路径的不同对象共享同一把锁。需要保护多个步骤时，使用
`async with file.transaction()`，块内通过同一个 `file` 操作，已有锁会复用。
也可写成 `async with file.transaction() as locked`，其中 `locked` 就是传入的 `file` 对象。
其他任务的独立调用或新建的 `LockedFile(path)` 仍需获取锁。事务内创建的工具子任务可以
继承当前对象的事务，但应在退出上下文前等待其完成；此处的事务只提供互斥，不提供回滚。

```bash
curl -N -X POST "http://localhost:7878/chat" \
  -H "Authorization: Bearer <JWT>" \
  -H "x-session-id: 018f0000-0000-7000-8000-000000000001" \
  -H "x-sender-name: demo-user" \
  -F 'content=[{"text":"你好，介绍一下你自己"}]'
```

上传文件：

```bash
curl -N -X POST "http://localhost:7878/chat" \
  -H "Authorization: Bearer <JWT>" \
  -H "x-session-id: 018f0000-0000-7000-8000-000000000001" \
  -H "x-sender-name: demo-user" \
  -F 'content=[{"text":"帮我看看这个文件"}]' \
  -F "files=@example.txt"
```

响应是 SSE 流，可以边生成边读取。

需要模型输出 JSON 对象时，添加表单参数 `-F 'json_output=true'`，默认关闭。
此参数仅在 `llm.support_json_output = true` 时，为大模型请求添加
`response_format={"type":"json_object"}`；不添加或修改任何系统提示词、用户提示词。
如果请求开启 JSON 输出，但 `llm.support_json_output = false`，接口会在 SSE 流开始前返回 HTTP 400，
不会调用模型。
`/api/chat` 代理接口也支持此参数。响应仍为 SSE，JSON 内容通过 `assistant` 事件分片返回，
需要拼接完整的最终回复后再解析 JSON。

### 内置命令

普通实时消息如果只包含一个文本块，且去除首尾空白后以 `/` 开头，会先由后端按内置命令处理，
不会把命令文本直接交给大模型。支持以下命令：

- `/help`：显示可用命令列表并结束本次请求。
- `/new`：先总结当前上下文并保存需要保留的记忆，再清空会话历史，开始新会话。
- `/clear`：直接清空会话历史，不进行记忆总结。

`/new` 和 `/clear` 后面可以继续填写消息正文；后端完成上下文处理后，会把命令后面的内容作为
新上下文中的第一条用户消息继续交给 Agent。例如：

```text
/new 帮我规划下一季度的工作
/clear 忽略之前的讨论，重新分析这个问题
```

未知命令会返回错误并提示使用 `/help`。后台定时任务不会触发这些命令。

每次成功完成的普通对话都会通过 SQLModel 追加到统一的
`session_record_path` 数据库文件。SQLite 表名为 `conversation_records`，
包含 `user_id`、`sender_name`、`user_question`、`model_response`、`input_tokens`、
`output_tokens`、`model_call_count`、`created_at` 和 `duration_s` 九个业务字段，以及 ORM 使用的自增
`id`。`created_at` 是应用写入记录时的 UTC 时间，`duration_s` 是本轮 Agent 处理耗时（秒）。
`session_record_path` 默认是
`data/session/record.db`，可在 `conf.toml` 的 `[path]` 中覆盖。

对话记录、Web 用户与会话、飞书令牌的建表和业务读写统一使用 SQLModel；
定时任务保留 APScheduler 自带的 SQLAlchemyJobStore，并接入公共 SQLite 连接配置。
SQLite 连接统一启用 WAL 模式，
支持读写并发；写入之间仍然串行，遇到锁竞争时最多等待 30 秒。已有数据库会在服务重启后首次连接时自动切换。

## 飞书定时任务

在飞书会话中可以直接用自然语言要求 Agent 创建、查看或取消定时任务，例如“明天上午 9 点提醒我
提交周报”或“每天上午 9 点发送一份行业简报”。Agent 会分别调用 `feishu_schedule`、
`feishu_schedule_list` 和 `feishu_schedule_cancel`。创建任务仅支持飞书会话；查看和取消操作只会
作用于当前用户拥有的任务。每个用户最多同时保留 4 个定时任务。

创建任务时，`run_at` 和 `cron` 只能有一个有效值：

- `run_at`：单次任务的 ISO 8601 时间，例如 `2026-09-08T09:00:00+08:00`。时区可以省略；
  省略时按服务端系统本地时区解释。建议显式携带时区偏移，避免部署环境变化产生歧义。
- `cron`：周期任务的标准 5 段 cron 表达式，例如 `0 9 * * *` 表示每天 9:00。分钟字段不能
  包含 `*`，因此 `* * * * *`、`*/5 * * * *` 等表达式会被拒绝。
- 空字符串和纯空白字段按未提供处理；如果两个字段都有有效值，或两个字段都没有有效值，任务创建失败。

独立的 scheduler 服务启动时会读取 `[path].session_schedule_path` 指定的 SQLite 数据库。
backend 通过 `api --scheduler-url` 管理任务，scheduler 通过 `scheduler --backend-url` 触发执行；
Compose 将后者设置为 Nginx 地址。配置模板使用：

```toml
[path]
session_schedule_path = "data/session/schedule.db"
```

任务会跨服务重启持久化，但不会补跑停机期间错过的任务：过期的单次任务会被删除，周期任务会推进到
下一次未来执行时间。任务触发后会随机等待 0～3 分钟以分散并发，然后在独立的后台 Agent 运行中执行；
执行记录会追加到会话工作区的 `memory/HISTORY.md`，实际任务结果通过飞书卡片发送给用户。

## 统计接口

统计接口使用以 `jwt.secret_key` 签发的内部 Bearer JWT，统一位于 `/statistics`：

- `GET /statistics`：问答量、活跃用户、会话数、模型调用次数、Token 和平均耗时总览。
- `GET /statistics/timeseries`：按 `hour`、`day`、`week` 或 `month` 返回使用趋势，默认使用
  `Asia/Shanghai` 时区分桶。
- `GET /statistics/users`：按问答次数倒序返回用户统计。
- `GET /statistics/sessions`：按最近活跃顺序返回会话统计。
- `GET /statistics/records/latest?limit=20`：按 ID 倒序返回最新完整问答，供滚动大屏使用。
- `GET /statistics/records?after_id=<id>`：按 ID 正序返回指定 ID 之后的完整问答，供大屏增量刷新使用。

统计时间筛选使用左闭右开区间：`start_at <= created_at < end_at`。列表接口支持 `limit` 和
`offset`，其中 `limit` 最大为 100。服务端会按完整查询参数缓存结果 5 秒，并合并相同参数的并发查询。
完整问答接口仍返回 `Cache-Control: no-store`，避免浏览器或代理持久缓存问答内容。

```bash
curl "http://localhost:7878/statistics/records/latest?limit=20" \
  -H "Authorization: Bearer <INTERNAL_JWT>"
```

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


## MCP 工具

可以在 `conf.toml` 的 `[tool.mcpServers.<name>]` 下配置 MCP Server。启动会话时，Xpeech 会在后台并发连接这些 Server、发现可用工具，首次对话无需等待 MCP 连接。每个 Server 就绪后，其工具会自动注册，并从后续模型调用开始可用。同一用户的并发会话共享连接和工具发现任务。

Agent 可调用无参数的 `query_mcp_status` 工具，立即查询当前通道允许访问的 MCP 状态。返回的 `servers` 按配置名称列出 `description`、`status`（`connecting`、`ready`、`unavailable`、`disconnected`）、`transport`、`tool_count`、`tools` 和 `error`。`error` 仅包含异常类型；查询不会等待连接或发起网络请求。连接失败后，下次对话会立即在后台重新连接，无需等待固定冷却时间；正在进行的连接和工具发现任务仍由并发会话共享。

每个 MCP 可配置 `description` 说明用途。当前通道可访问的服务名称和用途会直接写入 `query_mcp_status` 的工具描述，模型从首次请求起就能判断是否需要等待某个 MCP，无需先完成连接或调用状态查询。状态查询结果也会返回用途说明。服务名称、用途、状态和工具均先按 `channel_tag` 筛选；未匹配的 MCP 不会连接或展示。未配置 MCP 或筛选后为空时，不注册 `query_mcp_status`。

stdio Server 示例：

```toml
[tool.mcpServers.filesystem]
description = "读写工作区文件；任务需要本地文件操作时可等待此服务就绪"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
enabled_tools = ["*"]
execution_request_tools = ["write_*", "edit_*"]
tool_timeout = 30
```

远程 MCP Server 示例：

```toml
[tool.mcpServers.my-api]
description = "查询 CRM 客户资料和订单；客户、订单相关任务可等待此服务就绪"
url = "https://mcp.example.com/sse"
transport = "sse"
command_authorization_token = "curl --fail --silent 'https://auth.example.com/token?user_id=${user_id}'"
headers = { Authorization = "Bearer ${authorization_token}", X-User-ID = "$user_id" }
verify_tls = true
enabled_tools = ["search", "read_record"]
tool_timeout = 120
```

每个会话都会把当前用户 workspace 作为 MCP workspace root。stdio MCP
进程通过与 Shell 共用的 bubblewrap 沙盒启动，以该目录作为 `cwd`，并将 `HOME` 设置为
`<workspace>/home`，覆盖继承的 HOME；MCP 的 `env` 禁止配置 `PATH` 和 `HOME`。工作区可读写，系统运行时、公共 HOME 配置和内置技能只读挂载；
进程使用独立 PID 命名空间并移除所有 Linux capabilities，连接关闭时清理子进程。网络保持可用。
MCP SDK 直接启动沙盒进程，`command` 和 `args` 经 `shlex.join()` 合并后统一通过沙盒内的 `bash -c` 执行，参数中的特殊字符保持字面值；可执行文件及其依赖必须位于沙盒可见的系统运行时目录
（例如 `/usr`、`/opt`）或当前工作区内，backend 私有目录中的虚拟环境和文件不会自动暴露。
缺少 bubblewrap 或沙盒启动失败时连接报错，不回退到宿主机执行。
模型传入的 MCP 参数保持透传；HTTP/SSE MCP 通过标准 `roots/list` 获取工作区目录，不修改远端 HOME。
stdio MCP、内置文件工具和 Shell 的相对路径基准保持一致。远程 MCP 服务若需
直接读写文件，必须能以相同绝对路径访问该目录。

普通搜索使用 `baidu_search` 和 `bing_search`，网页文本抓取使用 `web_fetch`。`browser_preview_base_url` 只负责生成
用户可访问的 HTML 预览 URL 前缀，FastAPI 路由由该 URL 的 path 部分自动注册；
`browser_preview_path` 只负责存放预览文件，两者没有
路径推导关系。`create_browser_preview` 会把目录复制到 `<browser_preview_path>/<uuid>/`；
传入目录时返回该 UUID 目录的 URL 前缀；传入单个 HTML 时保留源文件名，
并返回完整的文件 URL。

字段说明：

- `description`：可选的服务用途说明，默认空字符串。供模型在连接完成前判断当前任务是否需要等待此服务。
- `command` / `args`：启动 stdio MCP Server 的命令和参数。
- `url`：连接远程 MCP Server。`/sse` 结尾的地址使用 SSE transport，其他地址默认使用 streamable HTTP。
- `transport`：可选，显式指定 `stdio`、`sse` 或 `streamable-http`。省略时按 `command` 或 URL 自动判断。
- `env`：仅用于 stdio Server 的环境变量，例如 `env = { API_KEY = "your-key" }`；HTTP/SSE 不允许配置。禁止设置 `PATH` 和 `HOME`，两者统一由沙盒管理。
- `command_authorization_token`：连接远程 MCP Server 前执行的取 token 命令，stdout 作为 `authorization_token`。
- `headers`：远程 MCP Server 的请求头，支持按用户替换占位符。
- `verify_tls`：是否校验远程 HTTPS 证书，默认 `true`。仅在自签名证书且无法安装内部 CA 时设为 `false`。
- `enabled_tools`：允许注册的 MCP 原始工具名或通配模式，默认 `["*"]` 表示全部注册；`[]` 表示全部禁用。
- `execution_request_tools`：需要用户确认的 MCP 原始工具名或通配模式，默认 `[]`；`["*"]` 表示所有已启用工具都需要确认。例如 `["write_*", "delete_*"]`。
- `channel_tags`：专属系统标签列表，省略或 `[]` 表示公共 MCP；当前 JWT 通道的 `channel_tag` 命中列表中的任一标签才连接并注册。
- `organization_prefixes`：用户组织前缀列表，省略或 `[]` 表示不限制组织；组织字符串以任一前缀开头才连接并注册，查询方式见上文 `[organization]`。
- `tool_timeout`：单次 MCP 工具调用超时时间，单位秒。

`command` 和 `url` 只能二选一。`command_authorization_token` 仅适用于远程 MCP，每次
建立或重建用户连接时执行；命令中的 `$user_id` 和 `${user_id}` 会替换为当前用户 ID，
但命令不经过 shell。命令必须在 10 秒内成功退出，并在 stdout 输出非空 UTF-8 token。

`headers` 的值支持 `$user_id` / `${user_id}` 和 `$authorization_token` /
`${authorization_token}`。后两者由 `command_authorization_token` 的 stdout 提供。其他 MCP 配置不做
运行时字符串替换。

MCP 连接按 `user_id` 隔离。同一用户在最后一次成功连接、工具发现或 MCP 工具调用后的 20 分钟内复用连接；
空闲超时后自动关闭，不同用户不会共享连接。

注册后的工具名会加上 `mcp_<server>_` 前缀，例如 `filesystem` Server 暴露的 `read_file` 会注册为 `mcp_filesystem_read_file`。
`enabled_tools` 和 `execution_request_tools` 均只匹配 MCP 原始工具名，支持 `*`、`?`、`[abc]`，区分大小写。

`execution_request_tools` 在 `enabled_tools` 筛选后生效，不会启用原本被禁用的工具。
命中的工具会增加必填的 `execution_description` 参数，并通过 `question` 表单等待用户确认；
说明字段不会传给 MCP 服务器。非交互模式下，未提供 `requires_confirmation` 回调时，这些工具会在注册阶段被过滤；提供回调时，按每次调用的判断决定直接执行或拒绝执行。

## 内置工具

Xpeech 提供丰富的内置工具，Agent 可以在对话中自动调用：

| 工具 | 说明 |
|------|------|
| `read_file` | 读取工作区或内置技能目录中的文件内容 |
| `write_file` | 向工作区写入文件 |
| `edit_file` | 编辑工作区内的文件 |
| `list_dir` | 列出工作区或内置技能目录内容 |
| `shell` | 执行 Bash 命令（带显式拦截规则和 bwrap 沙盒） |
| `web_fetch` | 使用 requests 抓取网页、Trafilatura 提取正文和评论并输出 Markdown，保留链接和表格；模型可通过 include_images 选择保留图片信息；支持 HTTP 重定向，不启动浏览器、不执行 JavaScript |
| `baidu_search` | 使用百度搜索中文网页，返回标题、链接、摘要；支持条数限制，不支持时间过滤 |
| `bing_search` | 使用必应 RSS 搜索网页，返回标题、链接、摘要；支持条数限制，不支持时间过滤 |
| `create_browser_preview` | 将工作区内目录或单个 HTML 复制到 UUID 预览目录，返回目录 URL 前缀或完整文件 URL |
| `read_doc_file` | 使用 firecrawl-anydoc 读取 Word、PowerPoint、Excel（含旧版和宏格式）、OpenDocument、RTF、EPUB、CSV、PDF，输出 Markdown |
| `send_file` | 向用户发送工作区或内置技能目录中的文件 |
| `ask_user_question` | 向用户发送表单提问 |
| `list_sessions` | 分页查看当前用户的普通会话和后台执行历史，包含各自的渠道名 |
| `read_session` | 按会话 ID 分页读取历史消息 |
| `feishu_schedule` | 为当前飞书会话创建单次或周期后台定时任务 |
| `feishu_schedule_list` | 查看当前用户的定时任务 |
| `feishu_schedule_cancel` | 取消当前用户拥有的定时任务 |

`shell` 已接入本地 dcg 危险命令检查：普通命令直接执行，破坏性文件/Git 操作需要用户确认，系统高危命令直接拒绝；检查失败不执行，也不弹确认。
模型调用时必须填写
`execution_description`，说明本次具体要做什么及目的；界面会显示说明和实际执行参数，
执行确认复用 `question` 事件，使用 `type: "confirm"` 展示说明、参数和“确认” / “取消”按钮。
该类型不接受 `fields` 或表单提交配置，也不追加补充说明；按钮提交 `{"decision":"confirm"}` 或 `{"decision":"cancel"}`。
`ask_user_question` 仍固定使用 `type: "form"` 收集表单信息。
回传字符串包含 `confirm` 且不包含 `cancel` 时才会运行原工具；包含 `cancel`、不含 `confirm` 或等待超过 300 秒均不会执行。
等待确认不占用工具执行超时；非交互模式下，需要确认的调用会直接拒绝执行，不发起询问。

`xpeech/agent/tools/registry.py` 通过 `build_execution_request_tool(tool)` 注册同名包装函数，
默认返回 `EXECUTION_REQUEST` 标记。`loop.py` 检查 `execution.value` 后发出 `question` 事件，
用户确认后才调用保存的真实函数；执行器无需处理特殊异常或额外结果字段。
Web 和飞书沿用现有 `/answer_question` 接口；后端直接检查整个回传字符串，不解析 JSON 或限定字段。
MCP 通过各 Server 的 `execution_request_tools` 配置启用同样的确认流程；默认 `[]`。

Shell 检查依赖 [Destructive Command Guard（dcg）](https://github.com/Dicklesworthstone/destructive_command_guard)。
本地运行需在 backend 所在环境安装 `dcg` 二进制并放入 `PATH`。
Dockerfile 从 [Gitee 镜像](https://gitee.com/luojiaaoo/destructive_command_guard/releases/tag/v0.16.0)
安装 dcg v0.16.0，支持 x86_64 和 aarch64，并在安装前校验发布文件的 SHA-256；
可通过构建参数 `DCG_VERSION` 指定版本。
Xpeech 仅调用 `dcg test` 检查命令，不需要安装其他 Agent 的 hook，也不会通过 dcg 执行待检命令。
`shell` 包通过 `__init__.py` 仅导出 `build_shell_tools` 和 `shell_requires_confirmation`，
`tool.py` 负责命令执行，`schema.py` 定义参数和结果。
检查函数 `shell_requires_confirmation` 位于 `shell/dcg/__init__.py`，由 `registry.py` 注册为确认判断回调。
注册时通过 `partial` 绑定当前工作区，检查进程以该目录为 `cwd`，`HOME` 使用同一工作区的沙盒 home，
`PWD` 同步为工作区路径；不修改后端进程的全局目录或环境。工作区不会作为模型可填写的参数暴露。
检查仍在宿主环境运行，不等同于完整复现 bwrap 的挂载视图（例如独立的 `/tmp` 和只读配置挂载）。
它通过随代码发布的 `xpeech/agent/tools/shell/dcg/config.toml` 启用 `core.filesystem`、`core.git` 和 `system`，
并加载 `dcg/agent.yaml` 补充规则。一次 `dcg test` 检查转换为字符串枚举 `ExecutionDecision`：
`allow` 对应 `ALLOW = "allow"`（直接执行），`ask` 对应 `CONFIRM = "confirm"`（用户确认），
`deny` 对应 `DENY = "deny"`（返回“发现高危命令，直接拒绝执行。”）。枚举定义在 `execution_request.py`。
普通递归删除（包括临时目录里的 `rm -rf` 及分离参数、长参数写法）和破坏性 Git 操作需要确认；
关机重启、`init 0/6`、`telinit` 改变运行级别、格式化、`dd` 写设备及 dcg 命中的其他系统破坏操作直接拒绝。
规则按实际可执行命令匹配，普通 `git init`、`echo shutdown` 不会仅因包含关键词而被拒绝。
混合命令采用最严格结果，确认级别的删除不能掩盖后续关机命令。静态检查不推断操作是否符合用户需求，
也不保证识别任意动态脚本内部的行为；文件系统隔离仍由 bwrap 负责。
未安装、检查超过 5 秒、返回格式异常或无法得出明确结论均返回工具错误，不执行、不询问。
非交互模式下，需要确认的调用也会直接拒绝。检查进程忽略继承的 `DCG_*` 环境覆盖，固定使用项目策略。
`SHELL_BLOCK_RULES` 保留 Python/pip 调用规范检查，key 为 `Callable[[str], bool]`，value 为提示词。
判断函数接收命令字符串，返回 `True` 时拒绝执行，并将对应提示作为工具错误返回给模型。
危险命令的确认/拒绝规则统一由 dcg 策略维护。文件系统访问边界由 bwrap 的挂载隔离和只读挂载保证，
Shell 不在宿主机上解析命令路径或拦截 `../`、`..\` 等路径文本。

需要按本次调用决定是否询问时，可在注册代码中传入可选的 `requires_confirmation`：

```python
import json

from xpeech.agent.tools.execution_request import ExecutionDecision, build_execution_request_tool
from xpeech.agent.tools.shell import build_shell_tools


def requires_confirmation(payload: str) -> ExecutionDecision:
    request = json.loads(payload)
    # 仅 pwd 直接执行，其他命令均询问用户。
    return ExecutionDecision.ALLOW if request["command"] == "pwd" else ExecutionDecision.CONFIRM


provider.register_function_tool(
    build_execution_request_tool(
        build_shell_tools(workspace=workspace),
        requires_confirmation=requires_confirmation,
        not_interactive=not_interactive,
    )
)
```

回调接收 JSON 字符串，例如 `{"command":"pwd","execution_description":"查看当前目录"}`。
`execution_description` 与原工具参数位于同一层，仅包含调用时提供并通过参数模型校验的字段，
不补入未提供的默认参数。回调支持同步函数或 `async def`：返回 `ExecutionDecision.CONFIRM` 时询问用户，
返回 `ExecutionDecision.ALLOW` 时直接执行原工具；返回 `ExecutionDecision.DENY` 时直接拒绝，提示“发现高危命令，直接拒绝执行。”，
不发出 `question` 事件；省略回调时每次都询问。回调抛出异常、超时或返回其他类型时，
按工具错误返回，原工具不会执行。

回调必须返回枚举成员；旧的布尔值、`None` 和普通字符串均作为类型错误处理，不会执行工具。

`not_interactive` 默认为 `False`。设置为 `True` 时，回调返回 `ExecutionDecision.CONFIRM` 会直接拒绝执行，
返回 `ExecutionDecision.ALLOW` 仍直接执行，不发出 `question` 事件。此时必须提供 `requires_confirmation`，
否则在构造包装器时抛出 `ValueError`，提示“请在注册的时候就屏蔽掉，而不是在运行时屏蔽”。
`registry.py` 会把当前会话的 `not_interactive` 传给 Shell 和需要确认的 MCP 包装器。

直接执行时会移除 `execution_description`，仅把原工具参数传给原函数。
MCP 包装也支持此回调，需在 `registry.py` 的 `register_mcp_bindings` 注册参数中传入；
它是 Python 注册参数，不是 `conf.toml` 配置项。非交互模式下未提供回调时，需要确认的 MCP 工具不会注册。

工具文本结果超过 `[tool].max_result_chars`（默认 10,000）时不会丢弃尾部内容：完整结果会以
文本文件保存到当前用户 workspace 的 `tool-results/`，工具消息返回原始内容前缀、总字符数和
相对文件路径。该阈值统一用于 shell、MCP、内置工具和自定义工具。`read_file` 是例外：执行器不会
把它的结果再次保存到 `tool-results/`；它会在返回内容超过该阈值时直接拒绝读取并提示缩小范围，
避免读取超长结果文件时再次生成结果文件。

`[tool].max_tool_timeout` 默认 300 秒，是每次工具调用的最终兜底上限。Shell、MCP 和网页抓取等
工具会先应用各自的执行或网络超时；触及兜底上限时，执行器取消该调用并把失败结果返回 Agent。
Shell 命令默认 120 秒超时，可在单次调用中通过 `timeout_ms` 指定，最大 240,000 毫秒；
超时后会终止命令及本次沙盒内的所有子进程，并把已产生的输出和超时状态返回 Agent。

### 工具安全

- **统一路径解析**：配置文件中的路径在加载时按 backend HOME 展开并转为绝对 `Path`；工作区文件 API/内置模型工具的路径参数保持字符串，在入口按当前工作区和沙盒 HOME 解析成绝对 `Path`，内部函数只接收 `Path`，不再兼容 `str | Path`。用户的 `~` 指向 `<workspace>/home`，`~用户名` 不会按 backend 账户展开；内部工作区 `Path` 通过 `resolve()` 规范化，不展开 backend HOME。MCP 参数按服务端协议透传，不根据字段名猜测本地路径。文件工具默认限制在工作区内，读取类操作可额外访问内置技能目录；Shell 文件系统访问由 bwrap 限制
- **预览目录隔离**：`create_browser_preview` 只能复制当前会话工作区内的目录或 HTML，每次写入 `<browser_preview_path>/<uuid>/`，请求子路径不能越出对应 UUID 目录
- **进程沙盒**：Shell 命令和 stdio MCP 通过 bubblewrap 运行，工作区可读写，内置技能目录只读挂载，系统运行时路径只读挂载，临时目录和工作区父目录使用 tmpfs 隔离；每个进程沙盒使用独立 PID 命名空间，并移除所有 Linux capabilities
- **依赖安装隔离**：Shell 工具首次使用时自动创建 `<workspace>/.venv`；项目 Python 依赖进入当前工作区虚拟环境，`uv tool install` 和 `npm install -g` 安装到 `<workspace>/home`，不会在会话间共享可写状态
- **系统高危命令拒绝**：dcg 拒绝关机重启、`init`/`telinit` 切换运行级别、格式化及写设备等高危操作，不弹确认
- **危险命令检查**：递归删除及破坏性 Git 操作由 dcg 返回确认；非交互模式直接拒绝，检查失败也不执行
- **Python/pip 调用检查**：继续拦截直接执行 `python` / `pip`，要求通过 uv 运行

### 沙盒

Shell 命令会在 bwrap 进程沙盒中执行：

- 每个用户工作区会绑定为可读写目录，并作为命令工作目录；该用户的多个会话共享此工作区。
- 每次调用启用 `--unshare-pid` 和 `--cap-drop ALL`，命令无法通过 `/proc` 查看其他调用的进程，也不保留挂载、切换用户等操作所需的 Linux capabilities。
- 每个 Shell 沙盒由 bwrap 的 `--unshare-net` 创建独立网络命名空间，再由 slirp4netns 配置出站 TCP/UDP 网络；网络就绪后才启动命令。不转发入站端口，并禁止通过 slirp 网关访问宿主机的 loopback。不同工作区沙盒的 `localhost`、监听端口和抽象 Unix socket 相互隔离，同一工作区的命令共享网络。
- bwrap 同时启用 `--unshare-ipc`，隔离 System V IPC 和 POSIX 消息队列；`/dev/shm` 随私有 `/dev` 隔离。相同用户的调用仍共享工作区，因此工作区内文件和文件型 Unix socket 可以共享。
- 出站 NAT 不限制公网或局域网目标；如果要阻止访问宿主机非 loopback 地址或其他内网服务，需要额外配置出口防火墙。允许外网也意味着可以通过外部服务间接交换数据。
- 命令正常结束、超时或取消后，本次沙盒内所有进程都会被清理，包括通过 `nohup`、`&` 或 `setsid` 启动的后台进程；工作区文件仍保留。需要临时服务时，应在同一次调用中启动服务、完成访问并停止服务，不能依赖服务跨调用存活。
- 内置技能目录以只读方式挂载，便于读取技能脚本和资源。
- `/tmp` 和工作区父目录会使用临时文件系统隔离，避免命令看到其他工作区。
- `/usr`、`/bin`、`/lib`、证书和 DNS 配置等系统路径以只读方式挂载，提供基础运行环境。
- `HOME` 会指向当前会话的 `<workspace>/home`；路径参数中的 `~` 也会解析到这个目录。
- `PATH` 会优先包含当前 HOME 下的 `.local/bin` 和 `.npm-global/bin`，`uv tool install`、`npm install -g` 的命令只在当前会话持久化。
- `sandbox_home_path` 指定公共 HOME 配置源目录（默认为 `assets/sandbox-home`）；其中的所有文件都会保留相对路径，通过 bwrap 只读映射到每个 HOME。

`wrap_sandbox_command(..., network_isolation=True)` 默认开启网络隔离，通过 Python 标准库启动器协调 bwrap 和 slirp4netns；设为 `False` 时直接运行 bwrap，共享宿主机网络，同时保留文件系统、进程、IPC 和 HOME 隔离。Shell 使用默认值；stdio MCP 显式关闭网络隔离，可访问宿主机的 `localhost` 服务，仅依赖 `bwrap` 和 `bash`。

Shell 的运行环境还需要 `slirp4netns`，Docker 镜像已安装；缺少依赖或网络初始化失败时拒绝启动，不会自动退回共享网络。Linux 必须允许创建所需命名空间，并提供 `/dev/net/tun`；Docker 中以 root 运行时还需要 `SYS_ADMIN` 和 `NET_ADMIN`（Compose 已为 backend 配置权限和设备映射）。不再依赖 RootlessKit、`uidmap`、`/etc/subuid` 或 `/etc/subgid`，也不再通过外层用户命名空间重映射 backend 的 UID/GID；root 按原身份进行 bwrap 挂载，普通用户由 bwrap 自行创建用户命名空间。Shell 的 `/etc/resolv.conf` 使用只读的私有配置，指向 slirp DNS `10.0.2.3`，以兼容宿主机 loopback DNS；stdio MCP 仍只读挂载宿主机的 DNS 配置。沙盒退出、启动失败或启动器终止时，网络辅助进程也会回收。

Shell 工具首次运行时会在工作区内执行 `uv venv .venv`，为该工作区创建独立 Python 环境。
Python 命令必须通过 `uv run python ...` 启动，直接执行 `python` / `pip` 会被安全检查拦截。
沙盒内设置了 `PIP_REQUIRE_VIRTUALENV=true` 和 `UV_PROJECT_ENVIRONMENT=<workspace>/.venv`，确保项目 Python 依赖安装进入当前工作区的 `.venv`。
同时设置 `UV_CACHE_DIR=<workspace>/home/.cache/uv` 和 `NPM_CONFIG_PREFIX=<workspace>/home/.npm-global`，让 uv 缓存、uv tool 工具和 npm 全局工具都保留在当前会话 HOME 内。

如果 `npx` 来自 nvm 或真实用户 HOME 中的 Node 安装，沙盒默认看不到它；需要把 Node/npm/npx 安装到 `/usr`、`/bin`、`/opt` 等沙盒可见路径，或在沙盒内通过可见的 npm 安装全局工具。

## 开发

```bash
uv sync
uv run pytest
```

如果需要检查配置是否能读取：

```bash
uv run python -c "from xpeech.config.settings import settings; print(settings.model_dump())"
```

## Web 客户端

Web 客户端基于 React 和 Ant Design X，使用 FastAPI 认证代理访问 Agent API，
并通过 SQLite 保存用户与登录会话。

![Xpeech Web 客户端](screenshot/PixPin_2026-08-25_10-44-39.jpg)

登录后的管理员可从页面右上角的“设置”下拉框进入“数据大屏”，也可通过 `/api/statistics`、`/api/statistics/timeseries`、
`/api/statistics/users`、`/api/statistics/sessions` 和 `/api/statistics/records*` 查询统计数据。
Web 服务会签发短期内部 JWT，将请求及全部查询参数转发到后端对应的 `/statistics*` 接口。
普通用户不会显示大屏入口，直接访问统计代理也会返回 403。

首次运行前构建前端：

```bash
cd xpeech/subapp/channel/web_client/frontend
npm install
npm run build
```

用户与登录会话数据库路径由 `conf.toml` 的
`web_client.database_path` 配置，默认为 `data/web_client/users.db`。登录 Cookie 名称由
`web_client.cookie_name` 配置，默认为 `xpeech_session`；同一 IP 上运行多个独立 Web
实例时，应为每个实例配置不同的名称（如 `xpeech_session_7939`），避免不同端口之间覆盖 Cookie。

### OAuth2 登录

Web 客户端支持同时配置多个通用 OAuth2 Authorization Code 登录服务商。启用后，登录页会显示
“账号密码”以及每个“`provider_name` 登录”标签页。`display_type = "qrcode"` 会显示一次性二维码；
`display_type = "link"` 会显示授权按钮，并在当前页面打开 OAuth2 授权页。两种方式授权
成功后都会自动登录原页面。授权请求默认使用 PKCE 和 `state`，授权链接有效期为 5 分钟。

当某个服务商的 `display_type = "link"` 时，未登录用户可以通过
`/?oauth2provider=<provider_name>` 直接进入授权流程，跳过登录界面。`oauth2provider` 参数会
忽略首尾空格和大小写，但必须与配置的 `provider_name` 匹配；参数缺失或不匹配时仍显示普通
登录页。例如 `provider_name = "飞书"` 时，可以使用：

```text
https://assistant.example.com/?oauth2provider=飞书
```

页面会先创建一次性 OAuth2 授权链接，再将当前页面重定向到服务商授权页；创建失败时会显示
错误信息和重试按钮。

### 登录提示词注入

`web_client.inject_prompt` 可以根据入口地址中的随机 `state`，调用任意命令取得下一条用户
消息的前缀，因此不依赖特定的 Redis、数据库或 HTTP 服务。账号密码、二维码和 OAuth2 链接
三种登录方式均支持：

```toml
[web_client.inject_prompt]
enabled = true
command_template = "curl --fail --silent 'https://prompt.example.com/get?state=${state}'"
```

`command_template` 必须包含 `${state}` 或 `$state`，占位符可以出现在任意参数中，也可以重复
使用。服务端会将其替换为入口随机码，但不会启动 shell，而是把替换后的内容作为可执行文件
及参数运行。例如 `state = "fR7p2mN9kL4qT8vX"` 时，上述配置实际执行：

```text
curl --fail --silent https://prompt.example.com/get?state=fR7p2mN9kL4qT8vX
```

命令须在 10 秒内成功退出，并通过 stdout 输出 UTF-8 提示词；输出内容不限制长度。服务端不
解释输出格式，只去除首尾空白后将文本返回给 Web 客户端。`state` 长度为 16～128 个字符，
并须匹配 `^[A-Za-z0-9._~-]+$`；调用方应生成不可预测且只可消费一次的随机值。

使用账号密码登录或在登录页手动选择登录方式时，可以直接访问：

```text
https://assistant.example.com/?state=fR7p2mN9kL4qT8vX
```

需要直接进入指定 OAuth2 服务商时，同时传入 `oauth2provider`：

```text
https://assistant.example.com/?oauth2provider=飞书&state=fR7p2mN9kL4qT8vX
```

账号密码登录成功后，Web 客户端会直接使用当前地址中的 `state` 请求提示词。回调定位登录记录后，链接登录只把原始入口 `state`
带回 Web 首页；二维码登录则继续使用原页面地址中的原始 `state`。命令占位符注入的也始终是
原始入口 `state`。

三种方式都会在登录成功后通过受认证的 Web 接口执行命令，并将命令输出保存在当前标签页的
`sessionStorage` 中，同时从地址栏移除 `state`。发送前，输入框上方会显示提示词的前 100
个字符，并可展开查看全文。用户下一次发送消息时，实际请求内容会按以下格式拼接；消息气泡
仍只显示用户输入的原始消息。发送后该前缀立即从 `sessionStorage` 清除，提示栏随之消失，
且仅生效一次。

```text
<命令输出的提示词>

===

<用户消息>
```

### OAuth2 配置

```toml
[[web_client.oauth2]]
enabled = true
provider_name = "XX"
display_type = "qrcode" # qrcode 或 link
client_id = "your-client-id"
client_secret = "your-client-secret"
authorization_url = "https://login.example.com/oauth2/authorize"
token_url = "https://login.example.com/oauth2/token"
userinfo_url = "https://login.example.com/oauth2/userinfo"
redirect_uri = "https://assistant.example.com/api/auth/oauth2/callback"
scopes = ["openid", "profile"]
user_id_claim = ".sub"
username_claim = ".name"
auto_create_users = false
use_pkce = true
token_auth_method = "client_secret_post"
```

`user_id_claim` / `username_claim` 是对 userinfo JSON求值的
[jq](https://jqlang.github.io/jq/manual/) 查询，例如 `".data.employee_no"` 或
`".data.items[0].id"`。

可选的 `userinfo_filter` 提供二次访问校验：登录前对 userinfo JSON 求值该 jq
表达式，**有输出即放行，无输出则拒绝**，可作为认证之外的补充限制，例如只允许
某个部门的人登录：

```toml
userinfo_filter = 'select(.data.department == "R&D")'
```

注意应使用 `select()` 这类"条件不满足时无输出"的写法来表达拒绝条件；单纯的
布尔比较（如 `.data.department == "R&D"`）无论是否命中都会产生输出，起不到
限制作用。表达式在启动时校验合法性，配置无效会直接启动失败；求值过程中出现
运行时错误（如对 null 迭代）也会拒绝登录。

需要配置多个服务商时，重复整个 `[[web_client.oauth2]]` 数组表。各服务商的
`provider_name` 必须唯一（忽略首尾空格和大小写），前端和创建授权请求时用它选择对应配置。

需要在 OAuth2 服务商后台将 `redirect_uri` 注册为回调地址。服务端会用
`user_id_claim` 指定的用户信息字段查找本地用户的 `user_id`。默认
`auto_create_users = false`，管理员需要预先创建匹配的本地账号；设为 `true` 时会自动创建
普通用户，并使用 `username_claim` 作为显示名称。若服务商要求 HTTP Basic 方式提交客户端
密钥，可将 `token_auth_method` 改为 `client_secret_basic`。额外的授权参数可通过
`extra_authorization_params = { prompt = "login" }` 配置。

启动命令统一见[启动](#启动)章节。打开 `http://127.0.0.1:7939` 即可访问。
新数据库的初始管理员为 `admin` / `admin123456`；
