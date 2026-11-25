# 本地嵌入演示

用一个本地 HTTP 服务给固定演示用户签发短期 JWT，通过远端 `embed.js` 加载浮动聊天组件。
支持自动续签、展开聊天，以及退出后重新登录。仅使用 Python 标准库，无需安装项目依赖。
页面按功能分为八个标签页。「打开聊天」、登录按钮和状态提示始终可见。

| 标签页 | 测试内容 |
| --- | --- |
| 连接 | 聊天服务地址、JWT 通道和演示用户配置 |
| 网页测试 | Page Agent 插件注册、表单填写、本地步骤切换和手动提交 |
| 对话 | `setInput()`、`addFiles()`、`send()` 和最新的 `onReply()` 片段通知 |
| 会话 | `newConversation()` 和 `switchConversation()` |
| 注册片段 | `registerFragmentHandler()`、填入并发送测试提示词、在主页面绘制折线图 |
| 同步 | `chatSync()` 的独立输入、附件、返回片段和文件 |
| 文件 | 对话框文件通知 `onChatFile()` 和按路径下载 |
| 日志 | 片段通知、同步结果和操作记录 |

在「网页测试」页展开「插件注册」，填写支持工具调用的模型名称、OpenAI 兼容 API 根地址和可选的 API Key，再点击「启用网页操作」。页面通过 `registerPageAgent()` 应用模型设置。目标 Web 客户端须已构建 Page Agent。模型请求由浏览器直连，模型服务须允许演示站点跨域访问。

留在「网页测试」页，打开聊天并点击标题栏的「网页操作」开关。在 Page Agent 原生面板输入「联系人填张三，部门选运营，进入下一步，停在提交前」，按 Enter 执行。助手可以填写字段并进入核对页。核对后亲自点击「模拟提交」，提交结果仅显示在当前页面。

在「注册片段」页点击「注册片段按钮」，再点「填入并发送」。预填的提示词让模型生成六个月的访问量测试数据，
并调用 `fragment` 返回带 `type: "line_chart"`、`title`、`labels` 和 `values` 的 JSON。
工具调用仍在执行过程中正常显示。数据有效时，嵌入对话框额外显示带标题、描述和「绘制折线图」按钮的片段卡；
点击后在主页面当前标签页出图。
本页也可仅填入提示词而不发送。演示会话还提供「记录原文」「填入输入框」和红色的「清空输入框」按钮。

「切换会话」调用 `assistant.switchConversation(session_id)`。可粘贴已有会话的 UUIDv7，或从候选列表选取
本次登录用过的会话。测试时可先在第一个会话发送一条消息，再新建第二个会话，然后用该表单切回。
切换结果会显示在状态和交互记录中，不会自动展开聊天窗口。

可输入「请调用 fragment，text 为『你好』」测试片段通知；普通模型回复只显示在对话框内，不触发通知。
「同步」页只显示同步结果，不展开窗口或渲染到对话框；完整的 `{ session_id, fragments, files }` 在「日志」页查看。

「对话」和「同步」分别保存自己的文字和附件，切换标签页不会清空它们。附件支持多选，
也可点击「添加 JS 示例文件」生成订单 JSON。对话附件需先插入对话框再发送；同步附件随本次
`assistant.chatSync(text, { files })` 上传。有附件时文字可留空。退出登录或重新连接会清空两组附件、
文件路径输入和文件列表。

文件演示包含四种场景：

1. **sync 上传附件**：选择本地文件或添加 JS 示例文件，输入「分析附件并生成报告文件」，点击同步发送。
2. **sync 文件发现及直接下载**：通过 `const { files } = await assistant.chatSync(...)` 取得文件路径。在「同步」页的「发现的同步文件」中展示路径，并自动调用 `assistant.downloadFile(path)` 尝试浏览器下载。列表中的「直接浏览器下载」可重复触发；浏览器可能要求允许自动或多文件下载。
3. **按路径获取 File 后下载**：在「文件」页粘贴路径，点击「获取 File 并下载」。主站先调用 `assistant.getFile(path)` 得到 `File` 对象，再创建对象 URL 触发浏览器下载。
4. **对话框文件发现**：打开聊天窗口，让助手生成并发送文件。「发现的对话框文件」框通过 `onChatFile()` 显示文件路径，不自动下载；可点击下载或获取二进制。

对话框文件通过 `onChatFile()` 接收；sync 文件通过本次 `chatSync()` 返回的 `files` 接收，无需订阅。
返回的 `files` 为路径字符串数组，无文件时为 `[]`。两个文件框分别在通知到达和 sync 完成时更新。
下载按钮通过 SDK 自动携带当前 JWT。失败会显示在状态和交互记录中。
「JS 获取二进制」按钮调用 `assistant.getFile(path)`，返回带文件名及 MIME 类型的
浏览器 `File` 对象，并展示名称、大小和类型，不触发浏览器保存。主站可继续把文件放进
`FormData` 上传到自己的后台；演示页面只获取文件，不对接业务存储：

```javascript
const file = await assistant.getFile(path);
const form = new FormData();
form.append('file', file, file.name);
// 将 form 作为 fetch 的 body，上传到主站自己的存储接口。
// 也可使用 await file.arrayBuffer() 读取原始字节。
```

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
