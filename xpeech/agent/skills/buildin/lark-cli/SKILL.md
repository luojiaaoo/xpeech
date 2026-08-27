---
name: lark-cli
description: 通过 lark-cli 查询或操作飞书/Lark 开放平台资源；当用户要求读取、搜索、创建或修改飞书云文档、云盘、知识库、多维表格、日历、任务、消息、通讯录、邮件、会议等数据时使用。
---

# 飞书 lark-cli

使用内置 `lark-cli` 执行飞书开放平台操作。应用凭据已在镜像构建时注入，优先使用用户身份执行；用户令牌由独立的 `lark-oauth` 命令通过 Device Authorization Flow 获取、刷新和管理。

## 直接执行原则

- 先按下方“高频任务速查”选择命令。已有匹配项时直接执行，不要先调用 `lark-cli --help`，也不要从顶层域逐级查询帮助。
- 只有速查表没有覆盖，或实际缺少必填参数时，才查询一次最具体命令的帮助：`lark-cli <domain> <command> --help`。不要依次查询顶层帮助、域帮助和子命令帮助。
- 普通的搜索、列表和读取无需先读 CLI 内置 skill。复杂文档编辑、多维表格分析或不熟悉的领域约定，才执行一次 `lark-cli skills read <skill-name>`。
- 优先使用 `+shortcut`；没有合适 shortcut 时使用类型化命令。只有参数仍不明确时才执行 `lark-cli schema <service.resource.method>`，原始 `lark-cli api` 是最后的兜底。
- 用户说“所有”“全部”时要处理分页。支持 `--page-all` 的命令同时设置合理的 `--page-limit`；`drive +search` 不支持 `--page-all`，根据响应的 `page_token` 继续请求。

## 高频任务速查

`docs` 用于读取或修改一篇已知文档；搜索、枚举用户可见的云文档必须直接使用 `drive +search`，不要先去 `docs` 域查找搜索命令。

| 用户目标 | 直接使用 |
|---|---|
| 浏览最近的云文档 | `lark-cli drive +search --as user --query '' --sort edit_time --page-size 20` |
| 搜索云文档 | `lark-cli drive +search --as user --query '<关键词>' --page-size 20` |
| 只搜索标题 | 在 `drive +search` 后加 `--only-title` |
| 我拥有的文档 | 在 `drive +search` 后加 `--mine` |
| 我最初创建的文档 | 在 `drive +search` 后加 `--created-by-me` |
| 限定文档类型 | 在 `drive +search` 后加 `--doc-types docx,sheet,bitable,slides,wiki` |
| 读取已知文档 URL/token | `lark-cli docs +fetch --as user --doc '<URL或token>' --doc-format markdown` |
| 创建 Markdown 文档 | `lark-cli docs +create --as user --title '<标题>' --doc-format markdown --content @<文件>` |
| 列出知识库空间 | `lark-cli wiki +space-list --as user --page-all --page-limit 10` |
| 列出个人文档库 | `lark-cli wiki +node-list --as user --space-id my_library --page-all --page-limit 10` |
| 列出知识库节点 | `lark-cli wiki +node-list --as user --space-id '<space_id>' --page-all --page-limit 10` |
| 解析多维表格 URL | `lark-cli base +url-resolve --as user --url '<URL>'` |
| 列出多维表格记录 | `lark-cli base +record-list --as user --base-token '<token>' --table-id '<ID或名称>' --limit 100` |
| 搜索多维表格记录 | `lark-cli base +record-search --as user --base-token '<token>' --table-id '<ID或名称>' --keyword '<关键词>' --search-field '<字段>' --limit 20` |
| 读取电子表格单元格 | `lark-cli sheets +cells-get --as user --url '<URL>' --sheet-name '<工作表名>' --range 'A1:F20'` |
| 查看今日日程 | `lark-cli calendar +agenda --as user` |
| 查看我的任务 | `lark-cli task +get-my-tasks --as user --page-all --page-limit 20` |
| 列出我加入的群聊 | `lark-cli im +chat-list --as user --page-all --page-limit 10` |
| 读取指定会话消息 | `lark-cli im +chat-messages-list --as user --chat-id '<chat_id>' --page-all --page-limit 10` |
| 跨会话搜索消息 | `lark-cli im +messages-search --as user --query '<关键词>' --page-all --page-limit 20` |

补充约定：

- 个人云盘、知识库、日历、任务和消息默认显式传 `--as user`；只有用户明确要求机器人身份或操作应用级资源时才使用 `--as bot`。
- `drive +search --query ''` 表示按筛选条件浏览文档；单页最多 20 条。`--mine` 是当前所有者，`--created-by-me` 是原始创建者，两者语义不同。
- 已有飞书 URL 时直接交给支持 URL 的 shortcut，不要先手工解析 token。多维表格先用 `base +url-resolve` 获取 base、table、view 坐标。
- 面向用户展示少量数据可使用 `--format table`；程序化筛选使用 `--format json --jq '<表达式>'`。大批量数据优先用命令自身的 `--output` 或 `--output-path` 写入工作区文件。
- 多行内容和复杂 JSON 先写入工作区文件，再使用 `@文件` 参数，避免 shell 引号错误。

## 用户授权

当命令输出包含飞书设备授权 URL 时，这是正常的 Device Authorization Flow。记住是哪一条命令生成了这个 URL，然后按以下流程处理：

- `lark-oauth` 使用单次轮换的 refresh token。在任何时刻只执行一个 `lark-oauth`，也不要并发执行多个可能触发 `lark-oauth` 的 `lark-cli` 命令；必须等待当前工具调用结束后再执行下一条。

- 立即把完整 URL 原样发送给用户，确保可以直接点击。
- 不要打开 URL、替用户授权或修改 URL。把 URL 发出后结束当前飞书任务，后续只根据命令返回的授权状态处理，不要求用户额外回复“已完成授权”。
- 再次执行触发授权的命令时，`lark-oauth` 会使用已保存的 `device_code` 轮询，单次最多等待 60 秒。取得令牌后，业务命令会继续执行。
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

## 低频发现命令

```bash
lark-cli <domain> <command> --help
lark-cli schema <service.resource.method>
lark-cli skills read <skill-name>
```
