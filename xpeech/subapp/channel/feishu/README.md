# 飞书渠道

入口是 `app.run(chat_url, host, port)`，同一进程提供内部任务管理 API 和飞书长连接。`FeishuBridge` 负责组装组件、注册 SDK 回调、拦截渠道控制指令和管理连接生命周期，业务组件通过构造参数显式传递依赖。

| 模块 | 职责 |
| --- | --- |
| `app.py` | 内部任务 API、健康检查、飞书连接和调度器生命周期 |
| `bridge.py` | 组件组装、消息入口、`/stop` 控制指令、飞书连接启动和停止 |
| `channel.py` | 飞书 SDK 兼容处理、卡片元素局部更新 |
| `inbound.py` | 身份查询与缓存、入站消息解析、附件下载 |
| `backend.py` | 提交飞书消息，过滤用户回显并将正文和思考分片转换为飞书流式输出 |
| `conversation.py` | 消费后端聊天事件，分发文本、文件、工具和问题事件，处理后端错误 |
| `delivery.py` | 独立消息和流式卡片发送，校验发送结果 |
| `tools.py` | 工具调用卡片的创建、按调用 ID 更新、结果着色和流式收尾 |
| `questions.py` | 问题卡片发送、回调提交、重复点击保护、执行结果更新 |
| `schedule.py` | PostgreSQL 任务持久化、到期调用 `/chat_sync`、直接发送结果 |
| `formatting.py` | 聊天事件展示规则、Markdown 转义、内容截断 |
| `cards.py` | 表单、执行确认和定时任务结果的纯卡片构建函数 |
| `config.py` | 渠道常量 |

消息路径：SDK → `bridge` → `inbound` → `conversation` → `delivery` / `tools` / `questions`。卡片回调直接交给 `questions`，后台任务独立走 `schedule`。

同一会话可以连续发送：SDK 的 `chat_queue` 关闭，不再等待整轮回复或合并忙时消息；过期检测、去重、
权限检查和单条消息处理锁仍由 SDK 执行。新消息立即通过独立 SSE 请求投递给后端 inbox。
后端通过 `command` 事件返回“这条消息将用于引导后续回复”，有文本时附上开头最多 40 个字符。
飞书复用已有命令卡片展示这条回执，后续输出仍从原 SSE 返回。
各请求直接发送独立消息和卡片，同一用户的其他响应无需等待当前流式卡片发送完成。
流式更新使用各自的卡片 ID 和序号，工具进度由各请求独立管理。
`fragment` 和其他工具一样展示调用与结果。
请求及其资源生命周期仍由 SDK 调度，渠道不创建脱离管理的后台聊天任务。

私聊发送独立文本 `/stop`（允许首尾空白）时，`bridge` 在解析普通消息前拦截，查询发送者身份，
使用用户 JWT 和 `x-session-id: DEFAULT_SESSION_ID` 直接调用 `POST /chat/stop`。
该指令不进入 `/chat`、后端 `InboundMessage` 或 inbox。
停止结果复用 `command` 事件的命令卡片样式，直接发送，无需等待当前流式卡片发送完成。
收到后端确认后回复“已停止当前会话。”；没有运行任务时回复“当前没有正在运行的会话。”；
身份查询、网络、HTTP 或响应格式错误时回复“停止失败，请稍后重试。”。普通文本中提及 `/stop` 不会触发停止。

状态归属：

- `questions.question_cards` 保存等待提交或执行完成的问题卡片。每张卡片独立加锁，避免提交响应覆盖先到达的执行结果。
- `conversation.consume()` 的局部变量保存当前请求的工具卡片和问题关联。请求结束或取消时，清理问题索引并结束工具卡片流。

修改卡片样式优先看 `formatting.py` 和 `cards.py`；新增聊天事件从 `conversation.py` 分发到对应组件。后台任务不经过聊天消费流程。

相关回归测试：`tests/test_feishu.py`、`tests/test_feishu_tool_delivery.py`、`tests/test_feishu_tool_formatting.py`、`tests/test_channel_helper.py`。

任务管理 API 使用内部 JWT 和 `x-user-id`：`POST /schedules` 返回 `{"task": {...}}`，
`GET /schedules` 返回 `{"tasks": [...]}`，`DELETE /schedules/{job_id}` 返回
`{"job_id": "...", "cancelled": true}`。只有飞书交互聊天注册三个定时工具。

部署必须保持单副本、单 worker；任务表为 `feishu_scheduled_jobs`，不加载旧任务表。
关闭时停止调度并取消执行中的 HTTP 请求，再关闭连接和客户端。后台结果不提供持久化补发。
