# 本地嵌入演示

用一个本地 HTTP 服务给固定演示用户签发短期 JWT，通过远端 `embed.js` 加载浮动聊天组件。
支持自动续签、展开聊天，以及退出后重新登录。仅使用 Python 标准库，无需安装项目依赖。
页面还演示 `setInput()` 填入输入框、`chatSync()` 同步发送（纯后台交互，不展开窗口、
不渲染到对话框，回复只回传主页面），以及 `onReply()` 订阅模型回复。
「模型回复展示」有两个框：「流式对话最后回复」跟随对话框内的大模型响应同步变动，
每条完整回复完成后更新为最新的 assistant 消息；「同步调用最后回复」显示
`chatSync()` 返回事件中最后一条 assistant 消息。同步发送的完整过程另记录在
「主页面交互记录」区域：请求发起、`onReply` 回传的回复（标注请求 ID）、
以及 Promise 返回的原始事件列表；对照聊天窗口可确认对话框无任何变化。
手动聊天的回复同样记录在此，并与同步回复区分标注。

「后台请求附件」支持多选本地文件，也可点击「添加 JS 示例文件」，使用 `new File()`
生成一份订单 JSON。同步发送会通过 `assistant.chatSync(text, { files })` 上传这些文件，
有附件时文字可留空；「填入对话输入框」只填入文字。附件会保留以便重复测试，点击
「清空附件」可移除；退出登录或重新连接会清空附件、文件路径输入和两个文件列表。

文件演示包含四种场景：

1. **sync 上传附件**：选择本地文件或添加 JS 示例文件，输入「分析附件并生成报告文件」，点击同步发送。
2. **sync 文件发现及直接下载**：通过 `const { events, files } = await assistant.chatSync(...)` 接收结果。在「发现的 sync 文件」框展示返回的 `files` 中的路径和下载 URL，并自动调用 `assistant.downloadFile(path)` 尝试浏览器下载。列表中的「直接浏览器下载」可重复触发；浏览器可能要求允许自动或多文件下载。
3. **按路径获取 File 后下载**：将文件路径粘贴到「填写文件路径下载」框，点击「获取 File 并下载」。主站先调用 `assistant.getFile(path)` 得到 `File` 对象，再创建对象 URL 触发浏览器下载。
4. **对话框文件发现**：打开聊天窗口，让助手生成并发送文件。「发现的对话框文件」框通过 `onChatFile()` 显示路径和下载 URL，不自动下载；可点击下载或获取二进制。

对话框文件通过 `onChatFile()` 接收；sync 文件通过本次 `chatSync()` 返回的 `files` 接收，无需订阅。
返回的 `files` 为 `{ path, url }[]`，无文件时为 `[]`；原始 `events` 中仍保留 `send_file`。
两个列表分别在通知到达和 sync 完成时更新；sync 的原始事件仍显示在交互记录里，不重复添加文件条目。
下载 URL 仍要求当前 JWT 认证，下载按钮通过 SDK 自动携带凭证。失败会显示在状态和交互记录中。
「JS 获取二进制」按钮调用 `assistant.getFile(event.context)`，返回带文件名及 MIME 类型的
浏览器 `File` 对象，并展示名称、大小和类型，不触发浏览器保存。主站可继续把文件放进
`FormData` 上传到自己的后台；演示页面只获取文件，不对接业务存储：

```javascript
const file = await assistant.getFile(event.context);
const form = new FormData();
form.append('file', file, file.name);
// 将 form 作为 fetch 的 body，上传到主站自己的存储接口。
// 也可使用 await file.arrayBuffer() 读取原始字节。
```

目标聊天服务需部署支持文件上传、下载、`onChatFile()` 和 sync 返回 `files` 的新版 `embed.js` 与前端构建产物。

在仓库根目录创建 `data/embed-demo/config.json`（该目录已被 Git 忽略）：

```json
{
  "assistant_url": "http://10.84.24.83:7943/",
  "channel": "lalala",
  "user_id": "embed_demo_user",
  "sender_name": "演示用户",
  "title": "客服助手",
  "session_metadata": {"source": "embed_demo", "department": "客服"},
  "algorithm": "HS256",
  "secret_key": "填写服务端 jwt.channels.lalala.secret_key"
}
```

`algorithm` 应与远端全局 `jwt.algorithm` 一致。密钥只在 Python 服务端使用；浏览器只取得
60 秒有效的 JWT。页面不接受浏览器指定用户 ID，始终使用配置中的固定演示账号。
此签发接口用于本地演示；接入实际业务时，应先验证当前登录身份再签发 JWT。

页面上的「通道」和「签名 Key」都可以输入，**仅供测试方便**：填写的 Key 会通过
`POST /api/assistant/jwt` 发送到本地演示后端，由后端签发 JWT 后返回，浏览器不签名、
不保存密钥（Key 会绕后端一圈）。留空 Key 时使用配置里当前通道的密钥；更换通道时需
填写该通道的密钥。**正式接入时必须由业务后端签发并下发 JWT，绝不能把签名密钥放到
前端，也不能让前端自行签名；此处的 Key 输入只是为了方便本地测试。**

`user_id` 和 `sender_name` 在同一份配置中设置，省略 `sender_name` 时默认使用 `user_id`。
`user_id` 签入 JWT 的 `sub`；
页面通过 `init(loadJWT, sender_name, title, session_metadata)` 设置发送者名称、助手标题及会话元数据。
发送者名称通过 `x-sender-name` 请求头传递，支持中文并自动编码，去除首尾空白后不能为空。
`title` 默认是「AI 助手」，`session_metadata` 默认为 `{}`；元数据必须是 JSON 对象，
聊天请求通过表单字段传递。远端每次请求直接验证业务 JWT，不创建用户或登录会话；
用户 ID 取 JWT 的 `sub`，发送者名称取请求头，不关联远端同名账号。退出演示只清除
组件内存中的 JWT，已签发的 JWT 在过期前仍有效；演示 JWT 的有效期为 60 秒。

```bash
chmod 600 data/embed-demo/config.json
python3 examples/embed_demo/server.py
```

打开 <http://localhost:8080/>，点击「打开聊天」或右下角聊天按钮。
「聊天前端 URL」默认填入配置中的 `assistant_url`，「通道」默认填入配置中的 `channel`，
均可直接修改后点击「连接 / 重新连接」，也可在输入框按 Enter。
切换地址会销毁旧组件并创建新会话，无需重启演示服务；
刷新页面后恢复配置中的地址。目标服务仍需配置与演示 JWT 相匹配的通道和密钥。
浏览器需能访问远端聊天服务。服务仅监听 `127.0.0.1`；远程 IDE 可转发 8080 端口到本机。
按 `Ctrl+C` 停止，修改配置后重新启动生效。可通过 `--port 8081` 更换本地端口。
