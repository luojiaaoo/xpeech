# 飞书渠道

入口是 `app.run(chat_url, host, port)`，同一进程提供内部任务管理 API 和飞书长连接。`FeishuBridge` 负责组装组件、注册 SDK 回调、控制消息合批和连接生命周期，业务组件通过构造参数显式传递依赖，不再通过多重继承共享状态。

| 模块 | 职责 |
| --- | --- |
| `app.py` | 内部任务 API、健康检查、飞书连接和调度器生命周期 |
| `bridge.py` | 组件组装、合批入口、飞书连接启动和停止 |
| `channel.py` | 飞书 SDK 兼容处理、卡片元素局部更新 |
| `inbound.py` | 身份查询与缓存、入站消息解析、附件下载 |
| `conversation.py` | 消费后端聊天事件，分发文本、文件、工具和问题事件，处理后端错误 |
| `delivery.py` | 普通消息、流式输出、会话进度卡片发送，校验发送结果 |
| `tools.py` | 工具调用卡片的创建、按调用 ID 更新、结果着色和流式收尾 |
| `questions.py` | 问题卡片发送、回调提交、重复点击保护、执行结果更新 |
| `schedule.py` | PostgreSQL 任务持久化、到期调用 `/chat_sync`、直接发送结果 |
| `formatting.py` | 聊天事件展示规则、Markdown 转义、内容截断 |
| `cards.py` | 表单、执行确认和定时任务结果的纯卡片构建函数 |
| `config.py` | 渠道常量 |

消息路径：SDK → `bridge` → `inbound` → `conversation` → `delivery` / `tools` / `questions`。卡片回调直接交给 `questions`，后台任务独立走 `schedule`。

状态归属：

- `delivery.session_update_message_id` 保存用户当前可覆盖的进度消息，由入站批次结束时清理。
- `questions.question_cards` 保存等待提交或执行完成的问题卡片。每张卡片独立加锁，避免提交响应覆盖先到达的执行结果。
- `conversation.consume()` 的局部变量保存当前请求的工具卡片和问题关联。请求结束或取消时，清理问题索引并结束工具卡片流。

修改卡片样式优先看 `formatting.py` 和 `cards.py`；新增聊天事件从 `conversation.py` 分发到对应组件。后台任务不经过聊天消费流程。

相关回归测试：`tests/test_feishu.py`、`tests/test_feishu_tool_delivery.py`、`tests/test_feishu_tool_formatting.py`。

任务管理 API 使用内部 JWT 和 `x-user-id`：`POST /schedules` 返回 `{"task": {...}}`，
`GET /schedules` 返回 `{"tasks": [...]}`，`DELETE /schedules/{job_id}` 返回
`{"job_id": "...", "cancelled": true}`。只有飞书交互聊天注册三个定时工具。

部署必须保持单副本、单 worker；任务表为 `feishu_scheduled_jobs`，不加载旧任务表。
关闭时停止调度并取消执行中的 HTTP 请求，再关闭连接和客户端。后台结果不提供持久化补发。
