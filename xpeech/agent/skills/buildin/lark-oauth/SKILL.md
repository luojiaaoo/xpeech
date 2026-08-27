---
name: lark-oauth
description: 当用户需要执行 lark-cli 命令之前必须读取本技能。禁止运行 `lark-cli auth login`、`lark-cli config` 或自行创建 profile，只允许通过 lark-oauth 命令授权，授权之后 lark-cli 命令才能正常执行。
---

# 飞书 lark-oauth

使用内置 `lark-cli` 执行飞书开放平台操作。应用凭据已在镜像构建时注入，优先使用用户身份执行；用户令牌由独立的 `lark-oauth` 命令通过 Device Authorization Flow 获取、刷新和管理。

## 用户授权

当命令输出包含飞书设备授权 URL 时，这是正常的 Device Authorization Flow。记住是哪一条命令生成了这个 URL，然后按以下流程处理：

- `lark-oauth` 使用单次轮换的 refresh token。在任何时刻只执行一个 `lark-oauth`，也不要并发执行多个可能触发 `lark-oauth` 的 `lark-cli` 命令；必须等待当前工具调用结束后再执行下一条。
- 立即把完整 URL 原样发送给用户，确保可以直接点击。
- 不要打开 URL、替用户授权或修改 URL。把 URL 发出后结束当前飞书任务，后续只根据命令返回的授权状态处理，不要求用户额外回复“已完成授权”。
- 再次执行触发授权的命令时（和第一次的命令严格保持一致！），`lark-oauth` 会使用已保存的 `device_code` 轮询，单次最多等待 60 秒。取得令牌后，业务命令会继续执行。
- 如果命令返回“用户未授权（第 1/2 次等待结束）”，只允许再执行同一命令一次；不要询问用户是否已经完成授权，也不要增加额外重试。
- 如果命令返回“用户未授权”且提示 `device_code 已删除`（通常是第 2/2 次等待结束，也可能是用户拒绝或授权已过期），停止重试并把结果告诉用户。下一次执行会生成新的授权 URL。
- 如果授权由显式的 `lark-oauth --scope ...` 发起，后续轮询也必须使用 scope 集合相同的 `lark-oauth` 命令；获得令牌后再重新执行原业务命令。

不要运行 `lark-cli auth login`、`lark-cli config` 或自行创建 profile。当前构建使用自定义 Credential Provider，CLI 自带登录所保存的令牌不会回退给该 Provider 使用。不要读取、修改或展示 lark-cli、`lark-oauth` 的令牌或设备授权缓存。

## 写操作与风险

- 读取操作可以在用户请求范围内直接执行。
- 创建或修改操作先用命令帮助确认风险等级；适合预览时使用 `--dry-run`。
- 只有用户已经明确授权相应高风险写操作时，才传入 `--yes`。
- 删除、覆盖、发送消息、变更权限等不可轻易撤销的操作，在执行前向用户确认准确目标和影响范围。

## 错误处理

- 参数不确定：只查看最具体命令的 `--help`；类型化 API 参数才查看 `schema`，不要逐级探测或猜测参数名。
- 权限不足：报告缺少的 scope，并执行 `lark-oauth --scope <缺少的 scope>` 发起增量设备授权；把输出的授权 URL 发送给用户。后续使用 scope 集合相同的命令轮询，每次最多 60 秒且最多两次；取得令牌后再重跑原业务命令。不要切换到其他凭据或绕过权限。
- 授权 URL 之外的命令失败：保留有用的错误信息，修正命令后最多重试一次；仍失败则向用户说明原因。
- 不要执行 `lark-cli update`，也不要安装另一个 lark-cli 覆盖镜像内的定制版本。


## 批量授权策略

当任务涉及多个模块、需要多个 scope 时，按以下流程操作：

### 流程

**第一步：执行命令，收集缺失 scope**
- 运行需要的 lark-cli 命令
- 如果返回 `missing_scopes`，把缺失的 scope 加入一个列表（不要马上授权）
- 继续执行其他不依赖这些 scope 的命令

**第二步：一次性批量授权**
- 如果列表不空，把所有 scope 合并成一个字符串：`scope1 scope2 scope3`
- 执行一次 `lark-oauth --scope "合并后的字符串"`
- 把生成的授权 URL 发给用户（只需用户点击一次）

**第三步：重跑被阻断的命令**
- 授权完成后，轮询等待令牌获取成功
- 重新执行之前因权限被阻断的命令

### 预判清单

开始任务前，根据用户意图预判可能需要的 scope：

| 模块 | 常用读取 scope | 写入 scope |
|------|--------------|-----------|
| base（多维表格） | `base:table:read`, `base:field:read`, `base:record:read`, `base:block:read`, `base:app:read` | `base:table`, `base:record` |
| wiki | `wiki:wiki`, `wiki:wiki:readonly`, `wiki:node:read`, `wiki:node:retrieve` | `wiki:wiki` |
| drive（云盘） | `drive:drive:read`, `drive:drive` | `drive:drive` |
| docs（文档） | `docs:doc:read`, `docs:docx` | `docs:docx` |
| calendar（日历） | `calendar:calendar:read`, `calendar:calendar` | `calendar:calendar` |
| im（即时通讯） | `im:message:read`, `im:message`, `im:chat:read`, `im:chat` | `im:message`, `im:chat` |
| contact（通讯录） | `contact:user.base:readonly`, `contact:user.employee:readonly` | - |
| vc（视频会议） | `vc:meeting:read`, `vc:meeting` | `vc:meeting` |

### 注意事项

- **最多只需授权一次**：任务开始时预判所需 scope，或任务执行中收集完所有缺失 scope 后再一次性授权。
- **不要增量授权**：避免用户反复点击授权链接。
- **如果授权 URL 已生成但用户尚未完成**：使用相同的 scope 集合轮询等待，不新增 scope。

## scope 速查表

当命令返回 `missing_scopes` 时，从以下列表选取对应的 scope：

```
# 多维表格
base:table:read      # 读取表结构
base:field:read      # 读取字段
base:record:read     # 读取记录
base:block:read      # 读取 Block 列表
base:app:read        # 读取 Base 信息

# 文档
docs:doc:read        # 读取文档内容
docs:docx            # 创建/编辑文档

# 云盘
drive:drive:read     # 读取云盘文件
drive:drive          # 上传/管理文件

# 日历
calendar:calendar:read  # 查看日程
calendar:calendar       # 创建/修改日程

# 通讯录
contact:user.base:readonly   # 查看用户基本信息
contact:user.employee:readonly  # 查看员工信息

# 即时通讯
im:message:read      # 读取消息
im:message           # 发送消息
im:chat:read         # 读取群聊信息
im:chat              # 管理群聊

# 知识库
wiki:wiki            # 管理知识库
wiki:wiki:readonly   # 读取知识库
wiki:node:read       # 读取节点
wiki:node:retrieve   # 获取节点详情

# 视频会议
vc:meeting:read      # 查看会议记录
vc:meeting           # 管理会议
```
