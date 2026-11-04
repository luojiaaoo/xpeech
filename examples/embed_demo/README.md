# 本地嵌入演示

用一个本地 HTTP 服务给固定演示用户签发短期 JWT，通过远端 `embed.js` 加载浮动聊天组件。
支持自动续签、展开聊天，以及退出后重新登录。仅使用 Python 标准库，无需安装项目依赖。
页面还演示 `setInput()` 填入输入框、`chatSync()` 同步发送，以及 `onReply()` 将每条模型回复显示在主页面。

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
