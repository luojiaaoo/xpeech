# OpenSandbox 执行环境

Shell 统一通过 OpenSandbox 执行。Agent、模型调用、文件工具、历史记录和文件下载继续在 backend；每用户的本地 workspace 以相同绝对路径挂入沙箱。没有本地 shell 回退。stdio MCP 仍由其独立的 bwrap 实现启动。

## 首次部署

使用同一台 Linux Docker 主机，并安装 Docker Compose v2（支持 configs.content）和 Buildx。

```bash
cp conf.toml.example conf.toml
cp .env.example .env
```

完成原有模型、JWT 和飞书配置。在 `.env` 中设置：

```dotenv
XPEECH_HOST_ROOT=/srv/xpeech
OPENSANDBOX_API_KEY=填入随机生成的独立密钥
# 如果 Docker 默认 bridge 网关不是 172.17.0.1，按实际地址修改：
# SANDBOX_HOST_GATEWAY=172.17.0.1
```

`XPEECH_HOST_ROOT` 必须是 **Docker 宿主机上本项目目录的绝对路径**，不要填写 backend 容器的 `/app`。可用 `pwd -P` 查看项目实际路径，用 `docker network inspect bridge` 查看网关。

```bash
mkdir -p docker_data/workspace_base docker_data/session docker_data/opensandbox docker_data/lark-cli
bash scripts/build-lark-cli.sh
docker compose build sandbox-image
docker compose up -d --build
```

服务端固定为 `opensandbox/server:release-1.2.0`，SDK 固定为 `opensandbox==1.2.0`。OpenSandbox 管理端只通过 Compose 内网提供服务，API key 由环境变量注入。执行容器由 Docker daemon 创建，不是 Compose 的 backend 副本。

## Python、Node 与 CLI

Dockerfile 提供三个可构建目标：

| 目标 | 内容 |
| --- | --- |
| `backend`（默认） | xpeech 服务；运行时也供媒体工具、stdio MCP 使用 |
| `sandbox` | Miniforge 26.9.4-1 管理的 Python 3.12、Node 24/npm、uv、conda/mamba，以及 bash、git、curl、jq、zip、ffmpeg 和命令监督程序；不包含 xpeech 服务或配置 |
| `lark-cli-export` | 从定制源代码单独编译的 `lark-cli`、`lark-cli-auth`，通过 Buildx local exporter 导出 |

CLI 不编译进 backend 或 sandbox 镜像。`docker_data/lark-cli/` 整个目录只读挂到两者的 `/opt/xpeech/bin`，并加入 PATH。更新 CLI 时：

```bash
bash scripts/build-lark-cli.sh
```

脚本先在临时目录编译，成功后以 rename 替换二进制。已启动进程使用旧版本，后续调用使用新版本，无需重建或重启 Python/Node 容器。编译使用 Dockerfile 中的定制源代码版本和 `conf.toml` 的飞书 app_id；修改本地 Go 模板后重新运行脚本即可，也可用 `LARK_CLI_REF=v1.0.90 bash scripts/build-lark-cli.sh` 切换上游版本。只为本机架构编译；交叉构建时显式指定 `LARK_CLI_PLATFORM=linux/arm64` 等。

每用户 Python 环境为 `<workspace>/.venv-opensandbox`，不会复用原宿主环境的 `.venv`。`python`、`python3`、`pip`、`uv run` 与 `uv pip` 默认使用同一个环境，没有必须通过 uv 调用 Python 的限制。Miniforge 负责运行时发行，默认用户环境采用轻量 venv，避免每个用户都复制一份 Node 和完整 Conda 发行版。需要独立 Conda 环境时可以用 `conda create` 或 `mamba create`，默认环境目录位于 HOME 下。HOME 保持 `<workspace>/home`，backend 和 bash 使用相同绝对路径；Conda、pip、npm、uv 的加速配置从 `data/sandbox-home` 只读映射进 HOME，Miniforge 构建时的 Conda 源配置在 `docker/sandbox/condarc`。uv/Conda 缓存、npm 用户安装目录仍在用户 HOME 中。容器回收后，这些内容保留。

## 挂载和网络边界

Compose 自动注入 `[tool.shell]` 的四个 `host_*_root`：workspace、内置 skills、HOME 公共配置、CLI 二进制。只有当前用户 workspace 可写，后三者只读。内置 skills 的覆盖路径与本地文件工具一致；请求是否提供飞书用户 token 改变时，重新创建对应挂载布局。

命令环境只传递明确列出的变量，飞书用户 token 在每次调用时读取，不保存到容器创建配置或注册文件。托管认证的 CLI 通过 Docker bridge 网关访问 token-manager 的 7883 端口。

Docker runtime 不支持 `secureAccess`。本部署使用官方 egress sidecar 的 `dns+nft` 网络策略：允许公网，允许指定网关的 token-manager 端口，拒绝其他私网和链路本地目的地，防止访问其他用户的 execd、backend、管理 API 和云元数据。执行容器不加入应用 Compose 网络。使用私网业务服务时，应先设计明确的放行规则，不要直接放开整个内网。

OpenSandbox 服务持有 Docker socket，仅提供给内部 backend。token-manager 只将 7883 发布到 Docker bridge 网关，不发布到公网网卡。注意该内部服务沿用项目现有的用户标识认证约定。

## 全机容量与生命周期

32 核 / 128G 单机的默认预算为 64 个用户沙箱、32 个执行槽、4 个创建槽，每沙箱 CPU 上限 500m、内存上限 1Gi；每个 egress sidecar 另有 100m / 128Mi 上限。这些值是压测起点，并非已经验证的 200 并发容量。

所有 backend 使用共享的 `/app/data/session/opensandbox`。POSIX flock 协调每用户串行、全机执行数量、创建数量和 resident slots；原子 JSON 文件保存 sandbox ID 和挂载指纹。不要把 state_path 放在用户 workspace 中，不支持将该目录放在不同机器的独立磁盘或任意 NFS 上。

首次 shell 调用创建容器；成功后服务端 TTL 续为 300 秒，后续使用重新续期。满额时优先回收未执行用户中最近最少使用的容器。请求排队默认最多 30 秒。所有副本必须使用相同的域名与数量限制，启动时会核对 topology.json。

backend 取消、超时或连接中断时，尝试中断命令并销毁该用户沙箱；不会自动重放命令。如果无法确认创建或销毁结果，保留容量槽，直到远端过期或后续成功销毁，避免越过容量上限。backend 崩溃留下的 busy 标记会触发销毁重建。

执行镜像中的监督程序对每路输出限制 128KiB，并在正常退出、超时、SIGTERM 时清理其子孙进程，包括 setsid 后台任务。Python 环境初始化也在同一用户锁内执行。

修改 CPU、内存、镜像标签或挂载布局时，下次使用按指纹重建。更新同一 CLI 挂载目录里的二进制不会触发重建。改变数量限制或管理端域名时，先停止 backend 并等待最后一次续期的沙箱 TTL 到期，确认旧沙箱清理完毕，再删除该 state_path 下的 topology.json 并启动全部副本。不要在任务运行时删除锁文件。

## 验证

```bash
docker compose ps
docker compose logs --tail=100 opensandbox backend
```

通过聊天调用 shell 检查 `node --version`、`python --version`、`pip --version`、`lark-cli --version`，再验证本地 write_file 写出的文件能被 shell 读取，shell 生成的文件能通过 send_file 下载。

可选真实服务集成测试（使用当前 conf.toml 和环境变量）：

```bash
RUN_OPENSANDBOX_TESTS=1 uv run pytest tests/test_opensandbox_live.py -v
```

独立单元测试覆盖 SDK 调用、共享容量、取消清理、挂载与 token，以及本机实际运行的命令监督程序；它们不代表 Docker 镜像、网络和生产容量已经通过验证。
