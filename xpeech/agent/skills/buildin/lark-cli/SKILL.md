---
name: lark-cli
description: 通过 lark-cli 查询或操作飞书/Lark 开放平台资源；当用户要求读取、搜索、创建或修改飞书云文档、云盘、知识库、多维表格、日历、任务、消息、通讯录、邮件、会议等数据时使用。
---

# 飞书 lark-cli

使用内置 `lark-cli` 执行飞书开放平台操作。应用凭据已在镜像构建时注入，优先使用用户身份执行；用户令牌由独立的 `lark-oauth` 命令通过 Device Authorization Flow 获取、刷新和管理。

## 执行流程

1. 根据用户目标确定命令域，并先查看该域的帮助：

   ```bash
   lark-cli <domain> --help
   ```

2. 优先使用匹配任务的 `+shortcut`。没有合适 shortcut 时，使用类型化 API 命令；不确定参数、权限或风险等级时先查看 schema：

   ```bash
   lark-cli schema <service.resource.method>
   ```

3. 只有在 shortcut 和类型化命令都不适用时，才使用原始 API：

   ```bash
   lark-cli api <method> <path> --params '<json>' --data '<json>'
   ```

4. 执行命令并根据结果向用户返回必要信息。结果较大时使用 `--jq` 筛选，不要把无关的大段 JSON 全部返回给用户。

## 用户授权

当命令输出包含飞书设备授权 URL 时，这是正常的 Device Authorization Flow。记住是哪一条命令生成了这个 URL，然后按以下流程处理：

- 立即把完整 URL 原样发送给用户，确保可以直接点击。
- 不要打开 URL、替用户授权或修改 URL。把 URL 发出后结束当前飞书任务，后续只根据命令返回的授权状态处理，不要求用户额外回复“已完成授权”。
- 再次执行触发授权的命令时，`lark-oauth` 会使用已保存的 `device_code` 轮询，单次最多等待 60 秒。取得令牌后，业务命令会继续执行。
- 如果命令返回“用户未授权（第 1/2 次等待结束）”，只允许再执行同一命令一次；不要询问用户是否已经完成授权，也不要增加额外重试。
- 如果命令返回“用户未授权”且提示 `device_code 已删除`（通常是第 2/2 次等待结束，也可能是用户拒绝或授权已过期），停止重试并把结果告诉用户。下一次执行会生成新的授权 URL。
- 如果授权由显式的 `lark-oauth --scope ...` 发起，后续轮询也必须使用 scope 集合相同的 `lark-oauth` 命令；获得令牌后再重新执行原业务命令。不要并发执行多个轮询命令。

不要运行 `lark-cli auth login`、`lark-cli config` 或自行创建 profile。当前构建使用自定义 Credential Provider，CLI 自带登录所保存的令牌不会回退给该 Provider 使用。不要读取、修改或展示 lark-cli、`lark-oauth` 的令牌或设备授权缓存。

## 写操作与风险

- 读取操作可以在用户请求范围内直接执行。
- 创建或修改操作先用命令帮助确认风险等级；适合预览时使用 `--dry-run`。
- 只有用户已经明确授权相应高风险写操作时，才传入 `--yes`。
- 删除、覆盖、发送消息、变更权限等不可轻易撤销的操作，在执行前向用户确认准确目标和影响范围。

## 错误处理

- 参数不确定：查看对应 `--help` 或 `schema`，不要猜测参数名。
- 权限不足：报告缺少的 scope，并执行 `lark-oauth --scope <缺少的 scope>` 发起增量设备授权；把输出的授权 URL 发送给用户。后续使用 scope 集合相同的命令轮询，每次最多 60 秒且最多两次；取得令牌后再重跑原业务命令。不要切换到其他凭据或绕过权限。
- 授权 URL 之外的命令失败：保留有用的错误信息，修正命令后最多重试一次；仍失败则向用户说明原因。
- 不要执行 `lark-cli update`，也不要安装另一个 lark-cli 覆盖镜像内的定制版本。

## 常用发现命令

```bash
lark-cli --help
lark-cli <domain> --help
lark-cli schema <service.resource.method>
lark-cli calendar +agenda
```
