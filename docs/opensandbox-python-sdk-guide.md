# OpenSandbox Python SDK 完整使用手册

> 适用范围：`opensandbox==1.1.1`、Python 3.10+
>
> 调研日期：2026-10-19
>
> 官方文档：[Python SDK](https://open-sandbox.ai/sdks/python) · [快速开始](https://open-sandbox.ai/getting-started/) · [源码](https://github.com/opensandbox-group/OpenSandbox/tree/main/sdks/sandbox/python)

本文介绍的是 OpenSandbox 的核心 Python SDK `opensandbox`。它负责远程沙盒的创建、连接、命令执行、文件操作、网络出口控制、监控与销毁。它不是 Python 标准库的 `sandbox`，也不是本项目当前的本机 Bubblewrap 封装。

官方站点跟随 GitHub 默认分支，可能早于 PyPI 发布包展示新功能。生产环境应固定 SDK、OpenSandbox Server 和沙盒内 `execd` 的版本，并在升级前检查 [Release 页面](https://github.com/opensandbox-group/OpenSandbox/releases)。

## 1. 核心概念

OpenSandbox 不是一个单独的 Python 进程隔离库，而是一套客户端/服务端架构：

| 组件 | 职责 |
|---|---|
| Python SDK | 从业务程序发起生命周期、命令、文件和监控请求 |
| OpenSandbox Server | 管理沙盒创建、查询、暂停、恢复、销毁和端点路由 |
| Runtime | 通过 Docker、Kubernetes、fast-sandbox 等后端创建隔离环境 |
| `execd` | 在沙盒内提供命令、文件、健康检查、指标及隔离会话 API |
| Egress sidecar / control plane | 实施网络出口策略和 Credential Vault |

调用链通常是：

```text
业务程序 -> opensandbox SDK -> OpenSandbox Server -> 沙盒 Runtime
                                      |
                                      +-> execd：命令、文件、指标、会话
                                      +-> egress：网络策略、凭据注入
```

SDK 有两套等价接口：

- 异步：`Sandbox`、`SandboxManager`、`ConnectionConfig`，适合 FastAPI、asyncio、并发 Agent。
- 同步：`SandboxSync`、`SandboxManagerSync`、`ConnectionConfigSync`，适合脚本和传统同步服务。

本文优先使用异步接口。同步接口的方法名和数据模型基本相同，只需去掉 `await` 并使用 `Sync` 类。

## 2. 安装与服务准备

### 2.1 安装 SDK

```bash
# 推荐固定版本
uv add "opensandbox==1.1.1"

# 或使用 pip
python -m pip install "opensandbox==1.1.1"
```

客户端池使用 Redis 时安装额外依赖：

```bash
uv add "opensandbox[pool-redis]==1.1.1"
```

检查安装：

```bash
python -c "import opensandbox; print(opensandbox.__version__)"
```

### 2.2 启动本地 OpenSandbox Server

SDK 本身不会启动服务。最小本地环境需要 Docker Engine 20.10+：

```bash
uvx opensandbox-server init-config ~/.sandbox.toml --example docker
uvx opensandbox-server
curl http://127.0.0.1:8080/health
```

期望返回：

```json
{"status":"healthy"}
```

部署、认证和 Runtime 的完整配置见[官方安装文档](https://open-sandbox.ai/getting-started/installation)与[服务端配置](https://open-sandbox.ai/getting-started/configuration)。

## 3. 连接配置

### 3.1 最小配置

```python
from datetime import timedelta

from opensandbox.config import ConnectionConfig

config = ConnectionConfig(
    domain="localhost:8080",
    protocol="http",
    api_key=None,
    request_timeout=timedelta(seconds=30),
    use_server_proxy=True,
)
```

关键参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `domain` | `localhost:8080` | Server 地址；也可直接包含 `http://` 或 `https://` |
| `protocol` | `http` | `domain` 不含 scheme 时使用 |
| `api_key` | `None` | 开启服务端认证后必填 |
| `request_timeout` | 30 秒 | 单次 HTTP 请求超时，不是命令执行时间或沙盒 TTL |
| `headers` | `{}` | 自定义请求头 |
| `debug` | `False` | HTTP 调试日志 |
| `follow_redirects` | `False` | 是否跟随重定向；上传永不跟随重定向 |
| `use_server_proxy` | `False` | 通过 Server 转发 `execd`/端点请求 |
| `retry_policy` | 启用 | 非流式请求的自动重试策略 |
| `transport` | SDK 创建 | 自定义共享 `httpx` transport |
| `disable_metrics` | `False` | 关闭 SDK 创建延迟遥测 |
| `enable_tracing` | `False` | 开启客户端池预热的 OpenTelemetry tracing |

SDK 也读取：

```bash
export OPEN_SANDBOX_DOMAIN=localhost:8080
export OPEN_SANDBOX_API_KEY=your-api-key
export OPENSANDBOX_DISABLE_METRICS=1  # 可选
```

显式构造参数优先于环境变量。

### 3.2 什么时候启用 `use_server_proxy`

- SDK 所在进程能直接访问沙盒容器/Pod 暴露的地址：可使用默认的 `False`。
- SDK 在宿主机外、NAT 后、另一个集群或只允许访问 Server：设为 `True`。
- 本地 Docker 快速开始通常建议设为 `True`，减少容器地址不可达问题。

代理模式下，SDK 会把 API Key 加到经 Server 转发的请求中；直连 `execd` 时不会把控制面 API Key 带入不可信沙盒。

### 3.3 连接复用

大量创建沙盒时可提供共享 transport：

```python
import httpx

transport = httpx.AsyncHTTPTransport(
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=50,
        keepalive_expiry=30.0,
    )
)
config = ConnectionConfig(
    domain="sandbox.example.com",
    protocol="https",
    api_key="...",
    transport=transport,
)

# 自定义 transport 归调用方所有，应用退出时自行关闭：
# await transport.aclose()
```

没有显式传入 transport 时，SDK 会创建并在 `close()` 时关闭它。异步 transport 一般只能在创建它的 event loop 中使用，不要跨多个 pytest event loop 复用。

## 4. 最小可运行示例

### 4.1 异步版

```python
import asyncio
from datetime import timedelta

from opensandbox import Sandbox
from opensandbox.config import ConnectionConfig
from opensandbox.exceptions import SandboxException


async def main() -> None:
    config = ConnectionConfig(
        domain="localhost:8080",
        use_server_proxy=True,
    )

    sandbox = None
    try:
        sandbox = await Sandbox.create(
            "python:3.12",
            timeout=timedelta(minutes=10),
            connection_config=config,
        )
        execution = await sandbox.commands.run(
            ["python", "-c", "print(1 + 1)"]
        )
        if execution.exit_code not in (None, 0):
            raise RuntimeError(str(execution))
        print(execution.text)
    except SandboxException as exc:
        print(f"OpenSandbox error: {exc.error.code}: {exc.error.message}")
        print(f"request_id={exc.request_id}, retryable={exc.is_retryable}")
        raise
    finally:
        if sandbox is not None:
            await sandbox.destroy()


if __name__ == "__main__":
    asyncio.run(main())
```

### 4.2 同步版

```python
from datetime import timedelta

from opensandbox import SandboxSync
from opensandbox.config import ConnectionConfigSync

config = ConnectionConfigSync(
    domain="localhost:8080",
    use_server_proxy=True,
)

sandbox = SandboxSync.create(
    "python:3.12",
    timeout=timedelta(minutes=10),
    connection_config=config,
)
try:
    execution = sandbox.commands.run(["python", "-c", "print(1 + 1)"])
    print(execution.text)
finally:
    sandbox.destroy()
```

### 4.3 `close()`、`kill()`、`destroy()` 的区别

| 方法 | 远端沙盒 | 本地 HTTP 资源 | 典型用途 |
|---|---|---|---|
| `close()` | 保留 | 关闭 | 稍后用 ID `connect()` |
| `kill()` | 请求销毁 | 保留 | 还需继续复用外部 transport/处理清理 |
| `destroy()` | 请求销毁 | 关闭 | 一次性 create-use-discard，最常用 |

特别注意：`async with sandbox` 和 `with sandbox` 退出时只调用 `close()`，不会调用 `kill()`。一次性任务请使用 `try/finally + destroy()`。

## 5. 创建沙盒

### 5.1 常用参数

```python
from datetime import timedelta

from opensandbox import Sandbox
from opensandbox.models.sandboxes import PlatformSpec

sandbox = await Sandbox.create(
    image="python:3.12",
    timeout=timedelta(minutes=30),       # None 表示不自动过期
    ready_timeout=timedelta(seconds=60),
    entrypoint=["tail", "-f", "/dev/null"],
    env={"APP_ENV": "sandbox"},
    metadata={"project": "xpeech", "user": "u-123"},
    resource={"cpu": "2", "memory": "4Gi"},
    resource_requests={"cpu": "500m", "memory": "1Gi"},
    platform=PlatformSpec(os="linux", arch="amd64"),
    secure_access=True,
    connection_config=config,
)
```

要点：

- 默认 TTL 是 10 分钟；显式传 `timeout=None` 才是不自动过期。
- 默认资源限制是 1 CPU、2 GiB 内存。
- 默认入口为 `tail -f /dev/null`，让容器持续运行。
- `ready_timeout` 同时覆盖端点发布与健康检查等待。
- `metadata` 中 `opensandbox.io/` 前缀由系统保留。
- `resource_requests` 不能超过 `resource` 限制，且是否生效取决于 Runtime。
- `skip_health_check=True` 仍会等待端点发布，只跳过健康检查。

### 5.2 私有镜像

```python
from opensandbox.models.sandboxes import SandboxImageAuth, SandboxImageSpec

image = SandboxImageSpec(
    "registry.example.com/team/runner:2026-10",
    auth=SandboxImageAuth(
        username="robot-user",
        password="registry-token",
    ),
)
sandbox = await Sandbox.create(image, connection_config=config)
```

不要把凭据写入源码、日志或异常信息；应从密钥管理系统注入。

### 5.3 自定义健康检查

```python
import httpx


async def app_is_ready(sandbox: Sandbox) -> bool:
    endpoint = await sandbox.get_endpoint(8080)
    url = f"{sandbox.connection_config.protocol}://{endpoint.endpoint}/health"
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(
                url,
                headers=endpoint.build_request_headers(sandbox.connection_config),
            )
        return response.status_code == 200
    except httpx.RequestError:
        return False


sandbox = await Sandbox.create(
    "my-service:latest",
    entrypoint=["/app/server"],
    health_check=app_is_ready,
    connection_config=config,
)
```

异步健康检查不能阻塞 event loop；同步健康检查必须自行设置 I/O 超时。

### 5.4 生命周期 Hook

```python
from opensandbox.models.sandboxes import (
    LifecycleHook,
    PeriodicLifecycleHook,
    SandboxLifecycle,
)

lifecycle = SandboxLifecycle(
    pre_start=LifecycleHook(
        command=["sh", "-c", "test -d /workspace || mkdir /workspace"],
        timeout_seconds=120,
    ),
    periodic=[
        PeriodicLifecycleHook(
            name="checkpoint",
            schedule="@every 5m",
            command=["sh", "-c", "date -u >> /tmp/checkpoints.log"],
            timeout_seconds=60,
        )
    ],
)
```

`pre_start` 在业务入口启动前阻塞执行；`periodic` 在启动后周期执行。同名周期任务不会重叠。Hook 支持范围取决于服务端 Runtime，使用前应检查部署能力。

## 6. 生命周期管理

### 6.1 查询、续期、更新元数据

```python
from datetime import timedelta

info = await sandbox.get_info()
print(info.id, info.status.state, info.expires_at)

# 过期时间重置为“当前时间 + 30 分钟”，可延长也可缩短
renewed = await sandbox.renew(timedelta(minutes=30))
print(renewed.expires_at)

# 字符串新增/覆盖；None 删除
updated = await sandbox.patch_metadata({
    "stage": "running",
    "obsolete": None,
})
```

### 6.2 连接已有沙盒

```python
sandbox = await Sandbox.connect(
    sandbox_id,
    connection_config=config,
    connect_timeout=timedelta(seconds=30),
)
try:
    print((await sandbox.get_info()).status.state)
finally:
    await sandbox.close()  # 有意保留远端实例
```

### 6.3 暂停与恢复

`pause()` 是异步控制面操作：请求返回不表示已经进入 `Paused`。应轮询状态并设置业务侧截止时间。

```python
import asyncio
import time

await sandbox.pause()

deadline = time.monotonic() + 120
while True:
    info = await sandbox.get_info()
    if info.status.state == "Paused":
        break
    if info.status.state == "Failed":
        raise RuntimeError(info.status.message)
    if time.monotonic() >= deadline:
        raise TimeoutError("pause timed out")
    await asyncio.sleep(1)

# 恢复可能改变端点，因此 resume() 返回一个新句柄
resumed = await Sandbox.resume(sandbox.id, connection_config=config)
await sandbox.close()
sandbox = resumed
```

暂停/恢复依赖 Runtime。不要在旧句柄上继续使用已缓存的命令或文件服务。

### 6.4 空闲自动暂停与访问唤醒

fast-sandbox 后端可按控制面观察到的入站流量判断空闲，并在下次访问时自动唤醒：

```python
from datetime import timedelta
from opensandbox.models.sandboxes import (
    ActivityProbe,
    IdlePolicy,
    SandboxLifecycle,
)

sandbox = await Sandbox.create(
    "my-service:latest",
    timeout=timedelta(days=1),
    lifecycle=SandboxLifecycle(
        idle_policy=IdlePolicy(
            idle_timeout_seconds=300,
            on_idle="pause",
            wake_on_access=True,
            active_probe=ActivityProbe(
                port=8080,
                path="/active",
                timeout_seconds=2,
            ),
        )
    ),
    connection_config=config,
)
```

`idle_timeout_seconds` 允许 30–86400 秒。`active_probe` 应返回 `{"active": true}` 表示仍有不产生入站流量的后台工作，防止被错误暂停。省略 idle policy 会保持“既不自动暂停，也不自动唤醒”的传统行为。

访问一个正在恢复的沙盒时，网关可能返回带 `OpenSandbox-Wake: expired` 标记的 503。默认 transport 会按 `Retry-After` 安全重发，预算为最多 5 次/60 秒；耗尽后抛出 `SandboxWakeTimeoutException`，但服务端恢复仍会继续。SSE 流式调用不执行 wake retry。

## 7. 命令执行

### 7.1 Shell 字符串与原生 argv

```python
# 交给 shell 解析，支持管道、重定向、变量展开
result = await sandbox.commands.run("printf '%s\n' \"$HOME\" | wc -l")

# 不经过 shell；用户输入不会被当作 shell 语法
result = await sandbox.commands.run(
    ["printf", "%s\n", "$HOME", "hello world"]
)
```

处理不可信参数时优先使用 argv 列表。原生 argv 需要较新的 `execd`；旧版本不支持时应升级服务端镜像。

### 7.2 结果对象

```python
result = await sandbox.commands.run("echo out; echo err >&2; exit 3")

print(result.id)
print(result.exit_code)
print([item.text for item in result.logs.stdout])
print([item.text for item in result.logs.stderr])
print(result.error)
print(result.text)  # 合并 stdout 与代码执行 result，不含 stderr
print(str(result))  # 人类可读摘要，包含 stderr/error
```

命令非零退出码不一定以 Python 异常形式出现，调用方应显式检查 `exit_code` 或 `error`。

### 7.3 工作目录、环境变量、身份与超时

```python
from datetime import timedelta
from opensandbox.models.execd import RunCommandOpts

opts = RunCommandOpts(
    working_directory="/workspace",
    timeout=timedelta(minutes=2),
    envs={"TASK_ID": "task-123"},
    uid=1000,
    gid=1000,
)
result = await sandbox.commands.run(["python", "main.py"], opts=opts)
```

- `RunCommandOpts.timeout` 是服务端命令执行时限。
- `ConnectionConfig.request_timeout` 是 HTTP 请求时限。
- `Sandbox.create(timeout=...)` 是整个沙盒 TTL。
- 三者互相独立，不应混用。
- 设置 `gid` 时必须同时设置 `uid`。

让环境变量对后续所有命令和 Bash session 生效：

```python
await sandbox.commands.set_env("MY_TOKEN", "value")
```

变量名必须匹配 `[A-Za-z_][A-Za-z0-9_]*`。该环境文件为追加写，最后一次同名赋值生效。敏感值更适合使用 Credential Vault，避免进入环境、日志或命令行。

### 7.4 实时流式输出

```python
from opensandbox.models.execd import ExecutionHandlers


async def on_stdout(message) -> None:
    print(message.text, end="")


async def on_stderr(message) -> None:
    print(f"[stderr] {message.text}", end="")


handlers = ExecutionHandlers(
    on_stdout=on_stdout,
    on_stderr=on_stderr,
    # 长任务可防止 SDK 同时把全部输出积存在内存中
    skip_accumulation=True,
)

await sandbox.commands.run(
    "for i in 1 2 3; do echo $i; sleep 1; done",
    handlers=handlers,
)
```

异步 API 的 handler 必须是 `async def`。同步 API 使用 `ExecutionHandlersSync` 和普通同步函数。SSE/流式请求不会执行普通自动重试。

### 7.5 后台命令

```python
import asyncio
import time
from datetime import timedelta
from opensandbox.models.execd import RunCommandOpts

execution = await sandbox.commands.run(
    "python long_job.py",
    opts=RunCommandOpts(
        background=True,
        timeout=timedelta(minutes=10),
    ),
)
if execution.id is None:
    raise RuntimeError("server did not return an execution id")

cursor = 0
deadline = time.monotonic() + 660
while True:
    logs = await sandbox.commands.get_background_command_logs(
        execution.id, cursor
    )
    print(logs.content, end="")
    if logs.cursor is not None:
        cursor = logs.cursor

    status = await sandbox.commands.get_command_status(execution.id)
    if status.running is False:
        if status.exit_code != 0:
            raise RuntimeError(f"exit={status.exit_code}: {status.error}")
        break
    if time.monotonic() >= deadline:
        await sandbox.commands.interrupt(execution.id)
        raise TimeoutError("background command timed out")
    await asyncio.sleep(0.5)
```

### 7.6 持久 Bash session

适合需要保留 `cd`、shell 变量和 shell 状态的交互式流程：

```python
session_id = await sandbox.commands.create_session(
    working_directory="/workspace"
)
try:
    await sandbox.commands.run_in_session(session_id, "export STAGE=test")
    result = await sandbox.commands.run_in_session(
        session_id,
        'printf "%s\\n" "$STAGE"; pwd',
        timeout=timedelta(seconds=30),
    )
    print(result.text)
finally:
    await sandbox.commands.delete_session(session_id)
```

Bash session 只复用 shell 状态，不提供额外命名空间隔离；需要进一步隔离时使用第 12 节的 Isolation Session。

## 8. 文件系统

### 8.1 写、读、列举、搜索、删除

```python
from opensandbox.models.filesystem import (
    DirectoryListEntry,
    SearchEntry,
    WriteEntry,
)

await sandbox.files.create_directories([
    WriteEntry(path="/workspace/app", mode=755),
])
await sandbox.files.write_files([
    WriteEntry(
        path="/workspace/app/main.py",
        data="print('hello')\n",
        mode=644,
    ),
])

text = await sandbox.files.read_file("/workspace/app/main.py")

entries = await sandbox.files.list_directory(
    DirectoryListEntry(path="/workspace", depth=2)
)
for entry in entries:
    print(entry.path, entry.entry_type, entry.size, entry.mode)

matches = await sandbox.files.search(
    SearchEntry(path="/workspace", pattern="*.py")
)

await sandbox.files.delete_files(["/workspace/app/main.py"])
await sandbox.files.delete_directories(["/workspace/app"])
```

`write_file()` 是单文件便捷方法。其默认 mode 是 `755`，普通数据文件建议显式传 `mode=644`：

```python
await sandbox.files.write_file(
    "/workspace/config.json",
    '{"enabled": true}\n',
    mode=644,
)
```

### 8.2 二进制与大文件流式读取

```python
from pathlib import Path

with Path("model.bin").open("rb") as source:
    await sandbox.files.write_file(
        "/workspace/model.bin",
        source,
        mode=644,
    )

data = await sandbox.files.read_bytes("/workspace/model.bin")

with Path("download.bin").open("wb") as target:
    async for chunk in sandbox.files.read_bytes_stream(
        "/workspace/model.bin",
        chunk_size=1024 * 1024,
    ):
        target.write(chunk)
```

`read_file()`、`read_bytes()` 和 `read_bytes_stream()` 支持两种局部读取方式：

```python
# HTTP byte range
head = await sandbox.files.read_bytes(
    "/workspace/model.bin",
    range_header="bytes=0-1023",
)

# 按文本行读取；offset 从 1 开始
lines = await sandbox.files.read_file(
    "/workspace/app.log",
    offset=101,
    limit=50,
)
```

`range_header` 与 `offset`/`limit` 互斥。

### 8.3 移动、权限、替换和元数据

```python
from opensandbox.models.filesystem import (
    ContentReplaceEntry,
    MoveEntry,
    SetPermissionEntry,
)

await sandbox.files.move_files([
    MoveEntry(src="/workspace/a.txt", dest="/workspace/b.txt"),
])
await sandbox.files.set_permissions([
    SetPermissionEntry(path="/workspace/b.txt", mode=600),
])

results = await sandbox.files.replace_contents_detailed([
    ContentReplaceEntry(
        path="/workspace/b.txt",
        old_content="old",
        new_content="new",
    )
])
print(results[0].replaced_count)

info_by_path = await sandbox.files.get_file_info([
    "/workspace/b.txt",
    "/workspace",
])
```

### 8.4 指定 Linux UID/GID

```python
user_files = sandbox.files_with_identity(uid=1001, gid=1001)
await user_files.write_file("/workspace/user.txt", "hello", mode=600)
```

该能力要求 Linux 和支持 filesystem execution identity 的新版 `execd`。它遵循普通 Linux 文件权限，但本身不是租户安全边界。

## 9. 端点与沙盒内服务

沙盒程序监听端口后，可解析其外部地址：

```python
import httpx

endpoint = await sandbox.get_endpoint(8080)
url = f"{sandbox.connection_config.protocol}://{endpoint.endpoint}/api"

async with httpx.AsyncClient() as client:
    response = await client.get(
        url,
        headers=endpoint.build_request_headers(sandbox.connection_config),
    )
    response.raise_for_status()
```

开启 `secure_access=True` 后，访问时必须保留 `endpoint.headers` 中的路由/认证头。需要可分享的短期 URL 时：

```python
import time

signed = await sandbox.get_signed_endpoint(
    8080,
    expires=int(time.time()) + 300,
)
print(signed.endpoint)
```

## 10. 网络出口策略

默认拒绝，仅允许指定域名：

```python
from opensandbox.models.sandboxes import NetworkPolicy, NetworkRule

policy = NetworkPolicy(
    defaultAction="deny",
    egress=[
        NetworkRule(action="allow", target="pypi.org"),
        NetworkRule(action="allow", target="files.pythonhosted.org"),
        NetworkRule(action="allow", target="10.0.0.0/8", ports=[443]),
    ],
)

sandbox = await Sandbox.create(
    "python:3.12",
    network_policy=policy,
    connection_config=config,
)
```

运行中更新：

```python
current = await sandbox.get_egress_policy()

await sandbox.patch_egress_rules([
    NetworkRule(action="allow", target="github.com"),
    NetworkRule(action="deny", target="pypi.org"),
])

await sandbox.delete_egress_rules(["github.com"])
```

Patch 是按 `target` 合并，不会整体替换策略，也不会修改 `defaultAction`。同一 payload 内同 target 的第一条规则生效。域名规则不能同时指定 ports；ports 可配 IP/CIDR 或省略 target 表示所有 IP。

网络策略是否可用及实际实现取决于 Runtime。安全系统应在部署验收中验证策略确实被执行，不能只以 SDK 请求成功作为证据。

## 11. Credential Vault

Credential Vault 让 egress 组件在出站请求时注入凭据，真实 secret 不进入沙盒环境变量、文件、命令和日志。

```python
from opensandbox.models.sandboxes import (
    Credential,
    CredentialBinding,
    CredentialProxyConfig,
    NetworkPolicy,
    NetworkRule,
)

sandbox = await Sandbox.create(
    "python:3.12",
    network_policy=NetworkPolicy(
        defaultAction="deny",
        egress=[NetworkRule(action="allow", target="api.example.com")],
    ),
    credential_proxy=CredentialProxyConfig(enabled=True),
    connection_config=config,
)

state = await sandbox.credential_vault.create(
    credentials=[
        Credential(name="api-token", source={"value": "<secret>"}),
    ],
    bindings=[
        CredentialBinding(
            name="example-api",
            match={
                "schemes": ["https"],
                "hosts": ["api.example.com"],
                "paths": ["/v1/*"],
            },
            auth={
                "type": "apiKey",
                "name": "x-api-key",
                "credential": "api-token",
            },
        )
    ],
)
```

读取接口只返回脱敏元数据。并发 patch 可使用 `expected_revision` 做乐观锁；HTTP 412 表示 revision 已被其他写入者更新，应重新读取后重试。模板型沙盒的 Credential Vault 能力与镜像型沙盒不同，需以当前 Server/Runtime 文档为准。

## 12. Isolation Session：沙盒内的二级隔离

Isolation Session 在一个远端沙盒内部再用 Bubblewrap 创建 PID、mount、tmpfs 和环境命名空间。它适合一个可信 worker 沙盒中重复执行大量相互隔离的小任务，但不替代容器、gVisor 或 Kata 这样的租户安全边界。

要求较新的 `execd`，可先探测：

```python
capabilities = await sandbox.isolation.capabilities()
print(capabilities)
```

### 12.1 一次性执行

```python
from opensandbox.models.isolated import IsolatedRunOpts

result = await sandbox.isolation.run_once(
    "python -c 'print(6 * 7)'",
    workspace="/workspace",
    workspace_mode="overlay",
    profile="strict",
    opts=IsolatedRunOpts(timeout_seconds=30),
)
```

### 12.2 长生命周期隔离会话

```python
from opensandbox.models.isolated import (
    CreateIsolatedSessionRequest,
    IsolatedRunOpts,
    IsolatedWorkspaceSpec,
)

request = CreateIsolatedSessionRequest(
    workspace=IsolatedWorkspaceSpec(
        path="/workspace",
        mode="overlay",  # rw / ro / overlay
    ),
    profile="strict",   # strict / balanced
    share_net=False,
    uid=1000,
    gid=1000,
    idle_timeout_seconds=600,
)

async with sandbox.isolation.session(request) as session:
    await session.run("export STAGE=train")
    result = await session.run(
        'printf "%s\\n" "$STAGE"',
        opts=IsolatedRunOpts(timeout_seconds=30),
    )
    await session.files.write_file("/workspace/input.txt", "data", mode=600)
```

同一 Isolation Session 使用长生命周期 Bash，因此 shell 变量可跨 `run()` 保留。不同 session 相互隔离。`async with sandbox.isolation.session(...)` 退出时会删除 session。

### 12.3 后台任务与重新挂接

```python
import asyncio

session = await sandbox.isolation.create(request)
try:
    run = await session.run_background("python train.py")
    cursor = 0
    while True:
        logs = await session.run_logs(run.run_id, cursor=cursor)
        print(logs.text, end="")
        cursor = logs.cursor
        status = await session.run_status(run.run_id)
        if not status.running:
            if status.exit_code != 0:
                raise RuntimeError(status.error)
            break
        await asyncio.sleep(1)
finally:
    await session.delete()

# 进程重启后，只保存了 ID 也可恢复句柄
session = await sandbox.isolation.attach(known_session_id)
```

单次日志读取最多 16 MiB；后台任务运行时应持续增量拉取。前台 SSE 连接断开会中断当前 run，后台 run 才会继续。

## 13. Volume

SDK 模型支持 Host、PVC/命名卷和 OSSFS；后端可用性、允许路径和自动创建行为由服务端 Runtime 决定。

```python
from opensandbox.models.sandboxes import Host, PVC, Volume

volumes = [
    Volume(
        name="workspace",
        host=Host(path="/srv/opensandbox/workspaces/u-123"),
        mount_path="/workspace",
        read_only=False,
    ),
    Volume(
        name="models",
        pvc=PVC(claim_name="shared-models", create_if_not_exists=False),
        mount_path="/models",
        read_only=True,
    ),
]

sandbox = await Sandbox.create(
    "python:3.12",
    volumes=volumes,
    connection_config=config,
)
```

Host path 必须在 Server 的允许列表中。不要允许用户直接指定任意宿主机路径。一个 `Volume` 必须且只能设置 `host`、`pvc`、`ossfs` 中的一种后端。

## 14. 管理器：查询与批量管理

`SandboxManager` 不连接某个沙盒的 `execd`，因此适合后台管理，以及沙盒尚未 ready 时读取状态和诊断。

```python
from opensandbox.manager import SandboxManager
from opensandbox.models.sandboxes import SandboxFilter

manager = await SandboxManager.create(connection_config=config)
try:
    page_number = 1
    while True:
        page = await manager.list_sandbox_infos(
            SandboxFilter(
                states=["Running", "Paused"],
                metadata={"project": "xpeech"},
                page=page_number,
                page_size=100,
            )
        )
        for info in page.sandbox_infos:
            print(info.id, info.status.state)
        if not page.pagination.has_next_page:
            break
        page_number += 1
finally:
    await manager.close()
```

主要方法：

| 类别 | 方法 |
|---|---|
| 沙盒 | `list_sandbox_infos`、`get_sandbox_info`、`kill_sandbox` |
| 生命周期 | `renew_sandbox`、`pause_sandbox`、`resume_sandbox` |
| 元数据 | `patch_sandbox_metadata` |
| 诊断 | `get_diagnostic_logs`、`get_diagnostic_events` |
| 快照 | `create_snapshot`、`get_snapshot`、`list_snapshots`、`delete_snapshot` |
| 模板 | `create_template`、`get_template`、`list_templates`、`delete_template` |

## 15. 快照与模板

### 15.1 创建并恢复快照

快照创建是异步过程，返回后必须等待 `Ready`：

```python
import asyncio
import time

snapshot = await sandbox.create_snapshot(name="after-dependencies")

manager = await SandboxManager.create(connection_config=config)
try:
    deadline = time.monotonic() + 900
    while True:
        snapshot = await manager.get_snapshot(snapshot.id)
        if snapshot.status.state == "Ready":
            break
        if snapshot.status.state == "Failed":
            raise RuntimeError(snapshot.status.message)
        if time.monotonic() >= deadline:
            raise TimeoutError("snapshot was not ready in 15 minutes")
        await asyncio.sleep(2)

    restored = await Sandbox.create(
        snapshot_id=snapshot.id,
        connection_config=config,
    )
    try:
        print(restored.id)
    finally:
        await restored.destroy()
finally:
    await manager.close()
```

`image` 与 `snapshot_id` 必须二选一。快照能力依赖 Runtime；若源沙盒剩余 TTL 不足，先续期再创建快照。

### 15.2 从已发布模板创建

```python
templated = await Sandbox.create_from_template(
    template_id="tpl_...",
    timeout=timedelta(minutes=10),  # 模板模式必填
    metadata={"project": "xpeech"},
    connection_config=config,
)
```

模板必须先构建到 `Succeeded`。模板型沙盒的入口、环境、资源、卷、平台和生命周期 Hook 来自模板，创建时不能覆盖；只允许传 metadata、network policy、extensions、idle policy 等有限字段。

## 16. Client Pool：降低冷启动延迟

客户端池预先创建远端沙盒，并在状态存储中保存其 ID。它不是 HTTP 连接池，`max_idle` 也只限制空闲缓冲，不限制已借出或直接创建的总数。

该 API 目前标记为 experimental，生产使用应固定版本。

```python
from datetime import timedelta

from opensandbox import (
    AcquirePolicy,
    InMemoryAsyncPoolStateStore,
    PoolCreationSpec,
    SandboxPoolAsync,
)

pool = SandboxPoolAsync(
    pool_name="xpeech-python-workers-v1",
    owner_id="worker-01",
    max_idle=2,
    state_store=InMemoryAsyncPoolStateStore(),
    connection_config=config,
    creation_spec=PoolCreationSpec(image="python:3.12"),
)

await pool.start()
try:
    sandbox = await pool.acquire(
        sandbox_timeout=timedelta(minutes=30),
        policy=AcquirePolicy.RETRY_NEXT_IDLE_THEN_CREATE,
    )
    try:
        await sandbox.commands.run("echo pool-ok")
    finally:
        # acquire 是一次性消费，不归还池中；使用者负责销毁
        await sandbox.destroy()
finally:
    await pool.shutdown(graceful=True)
```

四种获取策略：

| 策略 | 空闲实例失效/为空时 |
|---|---|
| `FAIL_FAST` | 尝试一个空闲实例后立即失败 |
| `DIRECT_CREATE` | 尝试一个空闲实例，失败则直接新建 |
| `RETRY_NEXT_IDLE` | 最多尝试 `max_acquire_retries` 个空闲实例，然后失败 |
| `RETRY_NEXT_IDLE_THEN_CREATE` | 重试空闲实例后仍失败则直接新建 |

单进程开发可用内存 store。Gunicorn/Uvicorn 多 worker、Celery 或 Kubernetes 多副本必须使用 Redis store；同一逻辑池共享 `pool_name` 和 Redis `key_prefix`，每个进程使用唯一 `owner_id`。

## 17. 指标与诊断

### 17.1 资源指标

```python
metrics = await sandbox.get_metrics()
print({
    "cpu_count": metrics.cpu_count,
    "cpu_percent": metrics.cpu_used_percentage,
    "memory_total_mib": metrics.memory_total_in_mib,
    "memory_used_mib": metrics.memory_used_in_mib,
    "timestamp_ms": metrics.timestamp,
})
```

这是时点数据，不是持续流。周期采集时应控制频率，并在监控系统中计算时间序列。

### 17.2 远程诊断

```python
logs = await sandbox.get_diagnostic_logs(scope="container")
events = await sandbox.get_diagnostic_events(scope="runtime")

if logs.delivery == "inline":
    print(logs.content)
else:
    print(logs.content_url, logs.expires_at)
```

日志 scope 常见值为 `container`、`lifecycle`、`all`；事件常见值为 `runtime`、`lifecycle`、`all`。返回可能是 inline 内容，也可能是临时 URL。沙盒的 `execd` 未 ready 时，改用 `SandboxManager.get_diagnostic_*`。

## 18. 异常与自动重试

### 18.1 异常处理

```python
from opensandbox.exceptions import (
    SandboxApiException,
    SandboxException,
    SandboxRateLimitException,
    SandboxReadyTimeoutException,
    SandboxWakeTimeoutException,
)

try:
    sandbox = await Sandbox.create("python:3.12", connection_config=config)
except SandboxRateLimitException as exc:
    print(exc.retry_after)
    raise
except SandboxWakeTimeoutException as exc:
    print(exc.attempts, exc.waited)
    raise
except SandboxReadyTimeoutException:
    raise
except SandboxApiException as exc:
    print(exc.status_code, exc.request_id, exc.response_body)
    raise
except SandboxException as exc:
    print(exc.error.code, exc.error.message, exc.is_retryable)
    raise
```

保留 `request_id`，它是定位服务端日志最有用的关联字段。

### 18.2 默认重试语义

- 对 `GET/HEAD/PUT/DELETE/OPTIONS`，默认重试 429、502、503 和请求发送前的 DNS/TCP/TLS 等连接失败。
- 对 `POST/PATCH`，默认不因 HTTP 状态码重试，避免重复创建或重复写；仅在任何请求字节发送前的传输失败时重试。
- 默认最多重试 3 次，使用带抖动的指数退避，并尊重最多 60 秒的 `Retry-After`。
- SSE/流式调用不走普通自动重试。
- 传入自定义 `transport` 后，SDK 不再自动包裹 retry transport。

关闭重试：

```python
from opensandbox.transport import RetryPolicy

fast_fail_config = ConnectionConfig(
    domain="localhost:8080",
    retry_policy=RetryPolicy.disabled(),
)
```

不要轻易为 POST/PATCH 状态码开启重试；只有服务端提供幂等键或操作天然幂等时才安全。

## 19. 独立的 Code Interpreter SDK

核心 `opensandbox` 可以运行 shell/argv 命令。若需要持久化的 Python、Java、Go、TypeScript 代码上下文、结构化 result 等高级解释器功能，官方另提供 `opensandbox-code-interpreter` 包和专用镜像，它不属于本手册的核心 SDK API。

```bash
uv add "opensandbox-code-interpreter==1.1.1"
```

```python
from code_interpreter import CodeInterpreter, SupportedLanguage

sandbox = await Sandbox.create(
    "opensandbox/code-interpreter:v1.1.0",
    entrypoint=["/opt/code-interpreter/code-interpreter.sh"],
    connection_config=config,
)
try:
    interpreter = await CodeInterpreter.create(sandbox=sandbox)
    result = await interpreter.codes.run(
        "sum([1, 2, 3])",
        language=SupportedLanguage.PYTHON,
    )
    print(result.text)
finally:
    await sandbox.destroy()
```

镜像 tag、包版本和 Server/SDK 应按官方兼容矩阵配套固定，不建议直接使用浮动的 `latest`。

## 20. 生产实践清单

1. 固定客户端、Server、`execd` 和镜像版本，不以主分支文档替代版本契约。
2. 所有沙盒设置有限 TTL；只有明确的长生命周期工作流才用 `timeout=None`。
3. 每次创建后立即进入 `try/finally`，一次性沙盒统一 `destroy()`。
4. 用户输入作为 argv 参数传递，必须用 shell 时先做严格建模，不拼接字符串。
5. 同时设置 HTTP、ready、命令和业务总截止时间。
6. 长输出使用 handler + `skip_accumulation=True`，后台日志使用 cursor 增量读取。
7. 大文件用文件对象上传、流式下载，不在内存中一次性缓冲。
8. 网络默认拒绝，只允许任务所需目标；宿主机网络和云元数据地址单独验证阻断。
9. API Key、镜像密钥和业务 secret 不写日志；出站 secret 优先使用 Credential Vault。
10. `metadata` 携带 tenant/user/task/request 标识，方便审计、查询和异常回收。
11. 定期通过 Manager 扫描超龄、无 owner 或异常状态的沙盒。
12. 多进程池使用 Redis，不把内存池误认为进程共享池。
13. 容器隔离不是绝对边界；高风险多租户执行应评估 gVisor、Kata 或 VM Runtime。

## 21. 与 xpeech 迁移前 Bubblewrap 实现的关系

迁移前的 `xpeech/agent/tools/sandbox.py` 把单条本机命令包装为 `bwrap ... bash -c ...`。
`opensandbox` 分支已采用完整 loop 容器化：运行时源码位于 `runtime/src/`，后端负责 Channel、调度和沙盒委托。具体实现与部署见 [runtime/README.md](../runtime/README.md)。以下比较保留作原方案对照：

该实现中，SDK 管理容器生命周期、挂载和 Runtime HTTP 端点，不是让后端逐次调用 `commands.run()` 执行工具。一次消息创建一个容器，完整的“模型 → 工具 → 模型 → … → 最终回答”循环、自定义工具和 MCP 都在容器内运行；问答通过 `/answer` 回到同一个 Loop。后端仅转发 SSE、持有 Channel 身份并提供受限的调度/统计回调。消息结束后删除容器，工作区和历史通过用户级 Volume 保留。

| 当前本机 `bwrap` | OpenSandbox SDK |
|---|---|
| 同进程所在主机本地执行 | 通过 Server 管理远端容器/Pod/VM |
| 工作区 bind mount | 文件 API、Volume 或远程 workspace |
| 进程结束后沙盒即结束 | 有独立 ID、TTL、暂停/恢复、快照和重连 |
| 主要隔离文件系统/进程视图 | 可叠加容器、gVisor、Kata、VM 和网络策略 |
| 命令输出来自本地子进程 | SSE 流式输出或后台命令轮询 |
| 无控制面依赖，低延迟 | 需部署 Server/Runtime，存在网络与冷启动开销 |

因此 OpenSandbox 不是 `wrap_command()` 的直接语法替换。迁移通常要把“返回一条 shell 命令字符串”的接口改成异步执行器，并引入：

- `user_id/session_id -> sandbox_id` 的所有权映射；
- TTL、续期、异常回收和服务退出清理；
- 工作区上传/下载或持久 Volume；
- 流式输出到现有工具消息的桥接；
- 命令超时、取消到 `interrupt()`/`destroy()` 的映射；
- 本地 bwrap 与远程 OpenSandbox 的配置开关和降级策略。

如果只是希望在现有单机上加强命令隔离，保留当前 `bwrap` 更简单；如果需要多租户、远程弹性资源、沙盒生命周期、快照、网络策略或跨机器调度，OpenSandbox 更合适。

## 22. API 速查

| 对象 | 常用入口 |
|---|---|
| `Sandbox` | `create`、`create_from_template`、`connect`、`resume` |
| 生命周期 | `get_info`、`renew`、`pause`、`kill`、`close`、`destroy` |
| 命令 | `commands.run`、`interrupt`、`get_command_status`、`get_background_command_logs` |
| Bash session | `create_session`、`run_in_session`、`delete_session` |
| 文件 | `read_file`、`read_bytes_stream`、`write_file(s)`、`search`、`list_directory` |
| 隔离会话 | `isolation.create`、`session`、`run_once`、`attach`、`list`、`capabilities` |
| 网络 | `get_endpoint`、`get_signed_endpoint`、`get_egress_policy`、`patch_egress_rules` |
| 安全凭据 | `credential_vault.create/get/patch/delete` |
| 可观测性 | `get_metrics`、`get_diagnostic_logs`、`get_diagnostic_events` |
| 管理 | `SandboxManager` 的 list/get/kill/renew/pause/resume |
| 持久化 | `create_snapshot`、`Sandbox.create(snapshot_id=...)`、模板 API |
| 池 | `SandboxPoolAsync`、`SandboxPoolSync` |

## 23. 常见问题

### 创建成功但命令/文件请求连不上

客户端大概率无法直连沙盒端点。设置 `use_server_proxy=True`，并检查 Server 代理、端口发布、DNS、防火墙和 API Key。

### `destroy()` 后仍看到 Runtime 资源

`kill()`/`destroy()` 在服务端接受删除请求后即可返回，底层清理可能继续进行。用 Manager 轮询终态，不要假设调用返回即物理资源已消失。

### 上下文管理器退出后沙盒还在

这是设计行为：上下文管理器只 `close()` 本地资源。需要销毁时显式调用 `destroy()`。

### 命令超时为什么没有销毁沙盒

命令 timeout 只终止命令；沙盒 TTL 和销毁是另一层生命周期。业务可根据风险在命令超时后调用 `interrupt()`，必要时再 `destroy()`。

### SDK 有方法但调用返回 unsupported

公开 SDK 入口不代表当前 Server、Runtime 或 `execd` 已实现该能力。优先核对三者版本及 Runtime 支持矩阵，尤其是 pause/resume、snapshot、volume、lifecycle hook、egress、Credential Vault 和 Isolation Session。

### 为什么上传遇到 3xx 直接失败

文件上传 body 可能不可重放，因此即使启用 `follow_redirects=True`，SDK 也不会跟随上传重定向。应直接配置最终地址或修复反向代理。

## 24. 参考资料

- [OpenSandbox Python SDK 官方文档](https://open-sandbox.ai/sdks/python)
- [SDK 能力矩阵](https://open-sandbox.ai/sdks/)
- [Python SDK 源码与 README](https://github.com/opensandbox-group/OpenSandbox/tree/main/sdks/sandbox/python)
- [Getting Started](https://open-sandbox.ai/getting-started/)
- [Client Pool](https://open-sandbox.ai/guides/client-pool)
- [Isolation Sessions](https://open-sandbox.ai/guides/isolation-sessions)
- [Lifecycle Hooks](https://open-sandbox.ai/guides/lifecycle-hooks)
- [PyPI: opensandbox](https://pypi.org/project/opensandbox/)
