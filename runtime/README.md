# Xpeech Runtime

`xpeech-runtime` 是独立构建、独立启动的容器运行时子项目。它运行完整的 `AgentLoop`：模型流、反复工具调用、MCP、自定义工具、文件和 Shell、提问等待、压缩、记忆和历史持久化都在同一个 OpenSandbox 容器内完成。

根项目 `xpeech` 提供 Channel、API、后台调度、统计记录和 OpenSandbox 委托。两个 wheel 通过 Python namespace package 共享 `xpeech` 协议/配置代码；运行时模块实际位于 `runtime/src/xpeech/`，入口和容器服务位于 `runtime/src/xpeech_runtime/`。只安装根项目不会安装 loop 或工具执行代码。

```text
Channel ── /chat SSE ── Gateway AgentRunner
                         │
                         ├─ OpenSandbox.create(runtime image + scoped volumes)
                         ├─ POST runtime /run ── SSE events ── Channel
                         ├─ POST runtime /answer ← Channel answer
                         └─ finally: destroy sandbox / revoke capability

OpenSandbox runtime: complete AgentLoop
  model → tools → model → tools → final
  filesystem / shell / MCP / custom tools / memory / history
        │
        └─ scoped callback → gateway scheduler / conversation records
```

每次 `/chat` 或定时任务委托创建一个容器；该请求内所有模型迭代和工具共用同一容器。容器退出后，用户 workspace、HOME、依赖、历史和预览通过持久目录保留。首次访问时会把旧 `history/<user>.yaml` 复制到 `history/<user>/<user>.yaml`，迁移标记防止 `/clear` 后重新导入旧历史。

## 启动

安装开发环境（包含两个 workspace 项目）：

```bash
uv sync --group dev
uv run pytest -q
```

配置 `conf.toml` 中的 LLM、JWT 和 Channel。将 `.env.example` 复制为 `.env` 后，设置：

```dotenv
XPEECH_HOST_DATA_PATH=/absolute/path/to/xpeech/docker_data
OPENSANDBOX_API_KEY=an-independent-random-key
```

`XPEECH_HOST_DATA_PATH` 是 Docker daemon 宿主机目录，必须为绝对路径。把已有 `session/`、`workspace_base/` 等数据保留在该目录下。将仓库 `data/sandbox-home/` 中的镜像源配置复制到它的 `sandbox-home/`，并创建 `cache/` 和 `browser_preview/`。Compose 将它们挂载到后端 `/app/data/`，OpenSandbox 将相同物理目录中的当前用户子目录挂入运行时。

```bash
# 首次和 runtime/、内置技能、自定义工具变化后都需要重新构建。
docker compose --profile build build runtime-image backend
docker compose up -d opensandbox browserless lark-token-manager backend feishu web_client
docker compose logs -f backend opensandbox
```

`runtime-image` 只用于构建镜像，不需要启动。后端默认配置为 `use_server_proxy=true`。OpenSandbox 和动态创建的容器连接固定 Docker network `xpeech-net`，从而能访问 `backend`、`browserless` 和 `lark-token-manager`。

API、Channel、附件、下载、预览和统计 URL 保持原有协议。镜像有两个 target：`backend` 只安装根项目和 API/Channel 依赖；`runtime` 安装完整运行时以及 uv、Node、agent-browser、lark-cli、媒体工具。后端无需 SYS_ADMIN、unconfined AppArmor/seccomp。

基础镜像使用官方 Python、Node、Go 和 Astral uv 镜像，可通过 Docker build args `PYTHON_IMAGE`、`NODE_IMAGE`、`GO_IMAGE`、`UV_IMAGE` 指向自己的镜像仓库。只构建后端不会编译 Go CLI 或安装 Runtime 的模型、MCP、媒体依赖。

## 配置与权限

后端 `[sandbox]` / `XPEECH_SANDBOX__*` 支持 `domain`、`protocol`、`api_key`、`image`、`host_data_path`、`callback_url`、`ttl_seconds`、`run_timeout_seconds`、`ready_timeout_seconds`、`resource`、`runtime_env`。

每个运行时只获得当前用户 workspace、历史、cache、独立预览目录和只读 HOME 配置目录。服务端独立签发每次运行的随机 capability；它只允许当前用户的记录、创建/列举/取消定时任务回调。用户 ID、发送者和 Channel 元数据由后端从原始消息绑定，容器不能通过 payload 改写所有权。

运行时不会收到后端 JWT secret、飞书 app secret、Web 用户数据库或全局记录/调度数据库。LLM Key 和所配置 MCP 凭据属于执行依赖，会被注入运行时。`runtime_env` 用于显式注入 CDP 等运行环境，不继承整个后端进程环境。

运行时在 `/run` 内执行完整 loop，`/answer` 唤醒同一 loop 的提问。二者以及 `/health` 都验证 capability。等待模型、工具或回答时每 10 秒发送 SSE 心跳；`runtime_done` 是完成且已清理 MCP 连接的标记，提前断流视为失败。正常结束、错误或取消均请求删除容器；后端退出会并行清理活动沙盒并撤销 capability，后端进程崩溃时由有限 TTL 兜底。

容器提供 Shell 隔离边界，生产运行时不需要再次启动 Bubblewrap。已有命令过滤、workspace 路径校验、每用户 HOME、uv/npm 缓存、命令进程组取消和部分输出保留继续生效。

## 验证范围

`tests/test_runtime_delegation.py` 验证后端导入隔离、委托完整 runtime 入口、SSE 转发、取消时删除、capability 撤销、记录幂等与用户绑定，以及 Runtime HTTP 服务中真实执行的五轮“写文件 → Shell 修改 → 读文件 → 问答回传 → 最终回答”。模型调用用确定性响应代替，不请求外部模型。

取消测试同时覆盖 `asyncio.Task.cancel()` 和 FastAPI/Starlette 的 AnyIO 重复取消语义。后端在受保护的清理作用域中等待删除沙盒，避免 SSE 断开后取消传播再次打断删除请求。

真实容器验收脚本是 `tests/integration/test_opensandbox_runtime.py`，默认跳过。准备一个专用于验收的 OpenSandbox Server（不要指向生产工作区），并构建 Runtime 镜像后运行：

```bash
XPEECH_TEST_SANDBOX_DOMAIN=127.0.0.1:8080 \
XPEECH_TEST_SANDBOX_API_KEY=your-test-control-plane-key \
XPEECH_TEST_RUNTIME_IMAGE=xpeech-runtime:local \
XPEECH_TEST_DATA_ROOT=/absolute/path/to/test-data \
XPEECH_TEST_CALLBACK_HOST=host.docker.internal \
uv run pytest -q -s tests/integration/test_opensandbox_runtime.py
```

`XPEECH_TEST_DATA_ROOT` 必须是 Server 允许挂载的宿主机目录，并且对测试进程以相同绝对路径可见。`XPEECH_TEST_CALLBACK_HOST` 是容器可访问的测试进程 IP/主机名；Linux bridge 网络需要自行配置 host gateway。单独使用 host 网络时可设为 `127.0.0.1`，这仅用于验收，不是 Compose 的生产网络配置。

测试使用真实 SDK、OpenSandbox Server、容器和 HTTP/SSE；模型端点是测试进程提供的确定性 OpenAI 兼容服务，不消耗外部模型额度。四次独立容器运行覆盖：附件 → 文件/Shell → 自定义工具 → HTTP MCP → 表单回答 → HTML 预览 → 文件下载 → 创建并执行后台任务 → 工作区/历史跨容器持久化 → 中断长任务 → 沙盒删除与中央统计记录。它不替代真实飞书账号、Browserless 或外部模型服务的业务联调。

2026-10-21 验收结果：全量默认测试 `509 passed, 1 skipped`，另有 `2868` 项子测试通过；上述真实容器测试单独运行通过，创建的四个沙盒均已删除，中央数据库写入三条已完成会话记录。验收环境使用 rootless Docker 28.3.3、OpenSandbox Server `release-1.1.0`、execd `v1.1.0` 和 Python SDK `1.1.1`。后端与 Runtime 镜像分别构建，根项目 wheel 在不安装 Runtime/LiteLLM 的独立环境中能够启动健康接口。

## 迁移与回滚

部署前停止旧后端和定时调度，备份整个数据目录，记录旧镜像版本；保持 `workspace_base/`、`session/`、`sandbox-home/`、`cache/` 和 `browser_preview/` 的相对布局。新后端与 OpenSandbox 必须使用同一个 Docker daemon 对应的物理数据目录。先用独立验收目录测试，再切换真实 Channel。

更新内置技能、自定义工具或 Runtime 源码后，需要重建 Runtime 镜像；后端挂载的本地技能不会自动更新已经构建的容器。当前不提供失败后在后端本机执行工具的自动降级。

需要回滚时先停止新服务，恢复旧版本镜像及其匹配的备份数据，再启动旧服务。新版本的历史保存在 `session/history/<user>/<user>.yaml`，旧版读取 `session/history/<user>.yaml`；直接切回旧镜像不会自动得到迁移后的新增历史。不要同时运行新旧调度器操作同一数据库。

当前实现使用同机 Docker Host Volume；多节点 Kubernetes 需要共享存储映射或进一步抽象工作区，不应把本机 Host Volume 配置直接搬到多节点环境。

这里的容器隔离不是完整的零信任多租户方案：Runtime 能访问所配置的 LLM/MCP 凭据和内部依赖服务。外部不可信租户上线前，还需独立评估网络出口和 token manager 等内部服务的鉴权；不能只依赖内部 Docker 网络。
