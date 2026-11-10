# 网站嵌入浮动聊天组件

业务网站通过 ES Module `embed.js` 接入聊天组件，无需对接 OAuth2。使用 `import()` 动态
加载模块，不需要插入 `<script src>`，也不向 `window` 注册全局 API。登录成功后通过
`init(jwtOrProvider, senderName, title, sessionMetadata)` 传入业务后端签发的 JWT、必填的发送者名称、
助手标题及会话元数据，组件自动初始化
并显示浮动按钮；点击按钮展开聊天窗口。
退出登录时调用 `destroy()`，移除按钮和窗口，并清除内存中的 JWT。

普通页面与 iframe 页面共用同一套会话工作区，都提供 **历史会话** 和 **＋ 新建会话**。
PC 宽度下历史会话显示在左侧，窄屏下通过抽屉打开；点击历史会话会加载其中保存的消息，
并沿用该会话的 UUIDv7 `session_id` 继续对话。已经打开的会话只隐藏、不卸载，切换或新建会话不会
取消回复，也不会清空消息、文字草稿和待发送附件；正在回复的会话在菜单中显示加载圈。
普通页面的认证仍使用本地账号登录 Cookie，嵌入页面仍只使用 JS 注入的业务 JWT，
两者不会互换认证方式或开放本地账号、管理员接口。

所有会话共用一条 `/api/chat/ws` WebSocket。请求通过 `request_id` 关联响应，并注入 `session_id`
区分会话，多个会话可并发流式接收而互不影响。取消只针对对应 `request_id`，不会断开共享连接。
嵌入 JWT 随请求 JSON 发送，不放在 WebSocket URL 中；普通登录仍使用同源 Cookie。

普通页面和嵌入页面的聊天请求及表单回答都接受规范的小写 UUIDv7 `x-session-id`。普通页面为兼容
旧调用仍允许省略该请求头，此时使用主会话 `00000000-0000-7000-8000-000000000000`；嵌入请求必须
显式提供非主会话 ID。同一对话内 ID 保持不变，后端按用户 ID、通道与会话 ID 隔离上下文。
关闭再打开嵌入窗口会保留当前对话，销毁组件后重新初始化会创建新的对话。

Web 对话的原始事件独立保存，不与 Agent 的模型消息历史混用：

```toml
[web_client]
session_history_path = "data/web_client/session_history"
max_history_days = 30
```

文件按 URL 编码后的用户 ID 分目录，文件名为
`channel=[channel]&session_id=[session_id].yaml`。通道和用户 ID 均从已验证的 Cookie 用户或 JWT
取得；普通页面通道为 `web_client`，嵌入页面通道为 JWT `kid`。每次流式对话会记录一条用于界面
还原的 `user` 事件（包含附件名称）以及后端 SSE 原始事件，同一会话续聊时追加保存；
`/api/chat_sync` 的独立后台执行不写入该历史。查询或保存时按文件修改时间清理超过保留天数的文件，
并清理由非法文件名标识的历史文件。

登录或嵌入认证完成后，前端通过 `GET /api/history/sessions` 列出当前用户、当前通道的会话，
通过 `GET /api/history/sessions/{session_id}` 读取事件。两个接口都返回 `Cache-Control: no-store`；
读取尚未创建或已清理的会话返回空 `events`。历史中的提问表单仅供查看，新的消息仍使用选中的
`session_id` 发送。

嵌入功能默认可用，允许任意业务网站接入，无需启用开关或配置来源白名单。
Web 客户端和 backend 均允许任意来源的 CORS 请求、请求方法及请求头；
跨域调用使用 `Authorization: Bearer`，不使用跨域 Cookie。JWT 验签和账号权限校验仍然生效。

在业务网站的初始化代码中先动态加载一次模块，再在原有的登录成功、退出登录回调中调用
组件方法。将模块地址替换为实际部署的 Web 客户端地址：

```javascript
const assistant = await import("https://assistant.example.com/embed.js");

// 业务网站自行实现此接口：先验证当前登录身份，再为当前用户签发 JWT。
async function loadAssistantJWT() {
  const response = await fetch("/api/assistant/jwt", {
    credentials: "same-origin",
    cache: "no-store",
  });
  if (!response.ok) throw new Error("无法获取聊天登录凭证");
  return (await response.json()).jwt;
}

// 每次登录只调用一次；首次连接和自动续期都会调用获取函数。
function onLoginSuccess(currentUser) {
  assistant.init(loadAssistantJWT, currentUser.display_name, "客服助手", { source: "portal" });
  // 自动初始化并显示浮动按钮，聊天窗口默认收起。
}

// 用户退出业务网站时调用。
function onLogout() {
  assistant.destroy();
  // 继续执行业务网站原有的退出登录逻辑。
}
```

浮动按钮的创建和点击事件均由组件自动处理，业务网站不需要创建按钮或绑定 DOM 事件。
按钮大小为 46px，支持鼠标和触屏拖动，拖动不会展开聊天；位置限制在视口内，窗口缩放时
会自动调整。按钮带轻微入场和短暂呼吸动画，并遵循系统的减少动态效果设置。
页面刷新后，如果业务网站恢复了登录状态，在模块加载完成后调用 `onLoginSuccess(currentUser)`。
单页应用退出时先调用 `destroy()`，下次登录再调用 `init()`；浏览器会复用同一 URL 的模块实例。
单独导入模块不会显示组件或发起认证请求，只有调用 `init()` 才会初始化。
必须遵循 `init(...) → destroy() → init(...)` 的顺序；初始化后再次调用 `init()`
会抛出异常，即使用户、通道或 JWT 完全相同。切换用户或通道必须先销毁组件，不能自动切换。
自动续期的获取函数也只能返回同一用户、同一通道的 JWT；身份变化会报错，不会替换当前会话。

组件提供以下方法：

| 方法 | 行为 |
| --- | --- |
| `assistant.init(jwtOrProvider, senderName, title = "AI 助手", sessionMetadata = {})` | 初始化凭证、发送者名称、助手标题和会话元数据；每次销毁前只允许设置一次 |
| `assistant.hideButton()` | 隐藏浮动按钮，不关闭已打开的聊天窗口 |
| `assistant.showButton()` | 显示浮动按钮，不自动打开聊天窗口 |
| `assistant.open()` | 通过代码主动展开聊天 iframe，无需用户点击浮动按钮 |
| `assistant.close()` | 收起聊天窗口，保留 iframe、聊天内容和会话 |
| `assistant.setInput(text, { timeoutMs? } = {})` | 替换当前对话输入框内容（允许空字符串清空），不自动发送；返回 `Promise<{ session_id }>` |
| `assistant.addFiles(files, { timeoutMs? } = {})` | 将 `File[]` 追加到当前对话框的待发送附件，显示附件列表，不自动上传或发送；返回 `Promise<{ session_id }>` |
| `assistant.newConversation({ timeoutMs? } = {})` | 新建会话，清空对话框及草稿，等待新输入框就绪后返回 `Promise<{ session_id }>` |
| `assistant.send({ timeoutMs? } = {})` | 发送当前输入框的文字及待发送附件，消息与流式回复显示在对话框，提交后返回 `Promise<{ session_id }>` |
| `assistant.chatSync(text, { files?, timeoutMs? } = {})` | 在当前会话后台调用 `/api/chat_sync`，支持 `File[]` 附件，不渲染到对话框，返回 `Promise<{ session_id, events, files }>`，其中返回的 `files` 为 `{ path, url }[]` |
| `assistant.downloadFile(path, { timeoutMs? } = {})` | 下载 `send_file` 事件 `context` 指定的文件，自动携带当前 JWT 并触发浏览器保存，返回 `Promise<{ session_id }>` |
| `assistant.getFile(path, { timeoutMs? } = {})` | 获取文件二进制，返回 `Promise<File>`，包含文件名和 MIME 类型，供主站 JS 处理，不触发浏览器保存 |
| `assistant.onReply(callback)` | 订阅每条完整模型回复，返回取消订阅函数 |
| `assistant.onChatFile(callback)` | 仅订阅对话框的 `send_file`，回传 `{ session_id, path, url, request_id: null }`，返回取消订阅函数 |
| `assistant.destroy()` | 移除按钮和 iframe，取消请求与定时器，清除内存中的身份和 JWT，不发送服务端撤销请求 |

`setUserId()` 和 `setChannelName()` 已移除。`init()` 的无效字符串参数会同步抛出异常，
不初始化组件；验签失败、凭证过期或获取函数失败会显示错误并触发 `xpeech:error`。
也可以直接调用 `assistant.init(jwt, senderName, title, sessionMetadata)`，但组件无法签发新的业务 JWT；该凭证到期后
必须先 `destroy()`，再用新 JWT 初始化。需要自动续期时应使用上述获取函数，业务后端接口
应返回 `{ "jwt": "..." }` 并设置 `Cache-Control: no-store`。
组件尚未初始化或已经销毁时，显示、隐藏、打开、关闭操作不会创建组件；销毁可以重复调用。

### 主页面与 iframe 双向交互

调用 `init(...)` 后即可使用以下接口，SDK 会等待认证和输入框就绪再投递请求。
交互函数不会自动展开窗口；需要展示时调用 `open()`。
`chatSync()` 是纯后台交互：不在对话框渲染消息或回复，也不改变输入框和加载状态。

控制对话框可按顺序调用以下函数，`files` 必须为浏览器 `File` 对象数组：

```javascript
assistant.open();
const { session_id } = await assistant.newConversation();
await assistant.setInput('请分析附件');
await assistant.addFiles(Array.from(fileInput.files));
await assistant.send();
```

`addFiles()` 追加附件，不替换已有附件；空数组不改变附件。附件在 `send()` 时上传，
只发送附件时可不调用 `setInput()`。文字和附件均为空，或对话框正在回复时，`send()` 会拒绝 Promise。
`send()` 返回表示已开始发送，不等待模型回复；使用 `onReply()` 接收完整回复（`request_id` 为 `null`）。
后续网络错误显示在对话框中，与点击发送按钮一致。
`newConversation()` 与点击「新建会话」使用相同流程：生成新的 UUIDv7 会话 ID，隐藏当前对话并保留
其消息、文字草稿、待发送附件和后台回复，登录凭证保持不变；切换期间的新命令排队投递到新会话。

```javascript
// 包括用户在 iframe 手动聊天、主页面调用 chatSync 产生的回复。
const unsubscribe = assistant.onReply(({ session_id, content, request_id }) => {
  console.log("模型回复", session_id, content);
  // 手动聊天的 request_id 为 null；chatSync 回复带对应交互请求 ID。
});

assistant.open();
await assistant.setInput("请帮我分析当前订单"); // 用户可以继续编辑再发送

try {
  const { session_id, events } = await assistant.chatSync("总结当前订单的风险", {
    timeoutMs: 300000,
  });
  // events 是后端原始事件数组，例如 [{ event: "assistant", context: "..." }]。
  console.log(session_id, events);
} catch (error) {
  console.error("同步交互失败", error);
}

// 不再需要接收回复时调用；destroy() 也会清除全部 onReply 订阅。
unsubscribe();
```

通过 `files` 传入浏览器 `File` 对象数组，文件随本次请求上传，无需提前上传或转 Base64：

```javascript
// fileInput 是 <input type="file" multiple> 元素。
const { events } = await assistant.chatSync('分析这些文件', {
  files: Array.from(fileInput.files),
});

// 也可发送 JS 生成的文件。
await assistant.chatSync('总结这些数据', {
  files: [new File([JSON.stringify({ orders: [1, 2, 3] })], 'orders.json', {
    type: 'application/json',
  })],
});
```

有附件时允许 `text` 为空字符串；文本和附件不能同时为空。

`chatSync()` 在保留原始 `events` 的同时，将 `send_file` 事件整理到返回的 `files` 数组中：

```javascript
const { session_id, events, files } = await assistant.chatSync('生成报告', {
  files: Array.from(fileInput.files),
});
for (const file of files) {
  console.log(file.path, file.url);
  await assistant.downloadFile(file.path);
}
```

`files` 每项为 `{ path, url }`，按 `send_file` 事件顺序返回；没有文件时为 `[]`。
它是文件信息列表，二进制内容可通过 `getFile(file.path)` 单独获取。
输入选项的 `files` 仍是待上传的 `File[]`，与返回的文件信息列表用途不同。

下载复用 `GET /api/files?path=...`，服务端只允许访问当前用户工作区中的文件。
`downloadFile` 不展开聊天窗口；默认超时 30 秒，可通过 `timeoutMs` 调整。
Promise 在文件获取成功并触发浏览器下载后完成，不表示用户已保存到磁盘。
浏览器可能要求用户允许多文件下载，也可在业务页面的下载按钮中调用该接口。

主站需要把文件存入自己的后台时，调用 `getFile()` 获取 `File`，再使用业务网站自己的认证上传：

```javascript
const file = await assistant.getFile(event.context);
const form = new FormData();
form.append('file', file, file.name);
const response = await fetch('/api/your-file-storage', { // 替换为主站自己的存储接口。
  method: 'POST', credentials: 'same-origin', body: form,
});
if (!response.ok) throw new Error('保存到业务后台失败');

// 需要原始字节时也可读取：const bytes = await file.arrayBuffer();
```

`getFile()` 使用当前 JWT 获取完整文件后，通过 iframe 消息返回 `File` 对象，不做 Base64 编码。
默认超时 30 秒，可通过 `timeoutMs` 调整；切换会话或销毁组件会取消尚未完成的文件获取。

对话框中的 `send_file` 通过 `onChatFile()` 通知主站，无需点击对话框里的下载按钮。
sync 的文件直接通过 `chatSync()` 返回的 `files` 接收：

```javascript
const unsubscribeChatFiles = assistant.onChatFile(({ session_id, path, url }) => {
  console.log('对话框文件', session_id, path, url);
});

const { files } = await assistant.chatSync('生成报告');
console.log('sync 文件', files);
// 主站可按需调用 assistant.getFile(path) 或 assistant.downloadFile(path)。
```

`path` 原样保留事件中的文件路径；`url` 是带编码路径参数的绝对下载接口 URL，
例如 `https://assistant.example.com/api/files?path=reports%2Freport.pdf`。
该 URL 仍要求 JWT 认证；使用 `getFile(path)` / `downloadFile(path)` 会自动携带当前凭证。
`onChatFile()` 只接收对话框文件，通知的 `request_id` 为 `null`。
也可监听主页面的 `CustomEvent('xpeech:chat-file')`，其 `detail` 与该订阅参数相同；选择一种通知方式即可。
sync 文件只随本次调用的结果返回，不派发文件订阅或文件事件通知。
SDK 本身收到通知不会自动下载；调用方决定如何处理。`destroy()` 会清除文件订阅。

回复还会通过主页面的 `window` 派发 `CustomEvent("xpeech:reply")`，其 `detail`
与 `onReply` 参数相同。选择一种订阅方式即可，避免重复处理。
流式文本在一条回复结束时合并回传，一轮工具调用产生多条模型回复时分别通知；
思考内容、工具事件、token 统计和出错时尚未完成的回复不作为模型回复通知。
同步请求的回复在后端返回全部事件后逐条通知。

`chatSync()` 使用当前 JWT、发送者名称及初始化时的 `sessionMetadata`，
保留输入框草稿和待发送附件，只携带调用时通过 `files` 显式传入的附件。它使用后端的非交互模式，不等待用户填写问题表单；
后端为每次同步执行生成独立会话，不读取也不写入对话历史。
本轮消息和回复只经 Promise 与 `onReply` 回传主页面，不渲染到对话框，窗口收起时也可执行。
需要模型输出 JSON 对象时，在消息文本中加入完整标记 `::请严格以JSON格式回复::`，例如
`assistant.chatSync("总结当前订单的风险。::请严格以JSON格式回复::")`。仅本次请求生效，需模型配置支持；
不支持时返回 HTTP 400。
同步调用可与对话框手动聊天、其他同步调用并发执行，互不阻塞。
HTTP 错误会拒绝 Promise；HTTP 200 返回的 `error` 事件保留在 `events` 中，调用方应自行检查。

`setInput`、`addFiles`、`newConversation`、`send` 默认等待 30 秒，`chatSync` 默认等待 5 分钟，可通过 `timeoutMs` 调整。
超时只停止主页面等待，不代表后端执行已取消，请勿据此自动重复提交。
未初始化、认证失败、会话切换、页面重载或销毁时，相应 Promise 会拒绝；JWT 自动续期不切换会话。
双方都校验 `postMessage` 的 origin 和窗口来源，请求和响应还使用 `request_id`、`session_id` 关联。

Web Client 同时提供 `POST /api/chat_sync`，与 `/api/chat` 使用相同的 multipart 字段
（`content`、`session_metadata`、`files`）和认证 / 会话请求头，
成功时返回完整 JSON 事件数组；嵌入请求仍必须提供合法 UUIDv7 的 `x-session-id` 和 `x-sender-name`。

JWT 必须由业务后端先验证网站当前用户的登录身份，再为该用户签发，不能直接采用浏览器提交的
任意用户 ID。Header 的 `kid` 必须与服务端 `jwt.channels` 的自定义通道名一致，例如 `crm`
对应 `[jwt.channels.crm]`，使用该通道的密钥签名。禁止使用 `web_client`、`feishu`、`internal`。
通道名为 1–64 个字符，支持字母、数字、点、下划线、冒号和连字符，首字符须为字母或数字。
Payload 的 `sub` 是实际用户 ID，必须为 1–128 字符的字符串，不自动去除空格或转换数字。
`exp` 为必填的数值类型 Unix 秒时间戳；若包含 `nbf`、`iat`，也会校验时间有效性。
签名算法使用全局 `jwt.algorithm`；业务后端可按 `jwt.access_token_expire_seconds` 签发短期 JWT，
业务 JWT 的实际过期时间以其 `exp` 为准，无需换取 Web 客户端登录 token。
不要求或校验 `aud`，JWT 中的角色字段也不会授予权限。以下代码仅在业务后端运行，
独立部署的业务后端需使用相同的全局时间、算法及通道密钥配置：

```python
from datetime import UTC, datetime, timedelta
import jwt
from xpeech.config.settings import settings


def issue_assistant_jwt(current_user):
    # current_user 由业务后端自己的登录认证取得。
    channel_name = "crm"
    return jwt.encode(
        {
            "sub": str(current_user.user_id),
            "exp": datetime.now(UTC) + timedelta(seconds=settings.jwt.access_token_expire_seconds),
        },
        settings.jwt.channels[channel_name].secret_key,
        algorithm=settings.jwt.algorithm,
        headers={"kid": channel_name},
    )
```

`user_id` 由 JWT 的 `sub` 设置；`sender_name` 通过 `init` 的第二个必填参数 `senderName` 设置：

```javascript
assistant.init(loadAssistantJWT, "张三", "客服助手", { source: "portal", department: "客服" });
```

组件发送聊天请求时通过 `x-sender-name` 请求头传递发送者名称，自动进行 UTF-8 URL 编码，
支持中文。名称必须为字符串，去除首尾空白后不能为空；未传或无效时会同步抛出异常，不初始化组件。
名称直接使用请求中的值；自动续期保留初始化配置，`destroy()` 后清除。
`title` 默认是「AI 助手」，去除首尾空白后不能为空，用于嵌入浮窗标题、iframe 标题和欢迎文案。
普通网页顶部只显示系统名称，不再额外显示「AI 助手」。
`sessionMetadata` 默认是 `{}`，必须是可 JSON 序列化的对象，支持嵌套数据；每条嵌入聊天请求
通过 `session_metadata` 表单字段传递，不使用请求头。元数据中的 `channel` 由后端忽略，
实际通道仍取 JWT `kid`。标题和元数据在初始化时固定，自动刷新 JWT 时保留，销毁后清除。

组件服务和聊天后端需配置相同的通道及密钥。嵌入请求通过
`Authorization: Bearer <业务 JWT>` 携带凭证，Web 客户端每次请求都验证签名、时间、`sub`
和自定义通道 `kid`。校验通过后，聊天、表单回答和文件下载将原业务 JWT 原样转发给后端，
保留原有 `exp` 和其他声明，不重新签发或延长有效期。普通网页登录仍由 Web 客户端签发后端 JWT。
通道直接取已验证的 `kid`，前端不传递通道名，`x-channel-name` 不参与通道选择；
用户 ID 直接取已验证的 `sub`，不接受浏览器通过独立请求头指定用户 ID。

初始化和刷新 JWT 时，SDK 使用 `credentials: omit` 调用 `GET /api/auth/embed`，
通过 `Authorization: Bearer` 传入原 JWT。此接口只校验当前凭证，返回 `user_id` 和剩余
有效秒数 `expires_in`，不创建用户、登录会话或新 token。`/api/auth/embed/session`
兑换及撤销接口已移除，嵌入认证不使用 `EmbeddedSession` 表。

校验通过后，SDK 通过校验来源和窗口的 `postMessage` 将原 JWT、用户 ID 和发送者名称
交给聊天 iframe。SDK 和 iframe 均只在内存中保存凭证，不写 Cookie、URL、localStorage
或 sessionStorage；所有嵌入请求使用 `credentials: omit`。
使用 JWT 获取函数时，SDK 会在过期前重新调用它，并校验新 JWT 仍属于同一用户和通道；
iframe 和当前对话保持不变。请求返回 401 时也会触发重新获取。传入固定 JWT 字符串时，
无法自动签发新 JWT，过期后需 `destroy()` 再初始化。

`destroy()` 只销毁前端组件并清除内存凭证，不撤销已签发的 JWT；原 JWT 在 `exp` 到期前
仍有效。普通网页登录继续使用 HttpOnly Cookie 和 `sessions` 表，其退出、停用及密码
管理逻辑保持原有行为。

接入错误会显示在聊天窗口中，同时触发业务页面的 `xpeech:error` 事件，详情为
`event.detail.message`。关闭窗口不会退出登录；业务网站退出时必须调用 `destroy()`。
若反向代理设置了 `X-Frame-Options` 或 CSP，必须同时允许业务网站嵌入；HTTPS 网站应通过
HTTPS 加载组件。`/embed.js` 作为公开模块提供跨域加载响应头；嵌入认证接口允许
任意来源访问，但仍校验 JWT。业务网站配置 CSP 时，需要允许从组件服务地址加载脚本、发起
连接并嵌入 iframe。

嵌入会话独立于本地账号：不创建用户、不查找同名用户，也不读取本地账号的名称、
角色或启用状态。聊天、表单回答和文件下载直接使用已验签 JWT 的 `sub` 作为用户 ID；
聊天发送者名称直接取每次请求的 `x-sender-name`，嵌入聊天缺少此请求头会返回 422。
用户 ID 沿用后端 `UserJWTAuth` 的校验规则：1–128 个 ASCII 字母、数字、下划线、连字符
或空格，不能全部为空格。

业务 JWT 仅用于嵌入认证和聊天相关接口，
不能访问本地账号资料、修改密码、管理用户或查看管理员统计；即使 `sub` 与本地管理员
相同也不会获得管理员权限。本地账号的停用、删除或角色变化不影响同名嵌入身份，
业务网站负责其用户的登录认证和 JWT 签发。普通页面继续使用本地账号登录。

持有通道密钥的业务后端是受信任的身份签发方，必须保护密钥和 JWT 签发接口；浏览器仅接收
签好的短期 JWT。不要把服务端密钥或管理员密码放入嵌入代码。

旧的 `[web_client.embed].allowed_origins` 和
`XPEECH_WEB_CLIENT__EMBED__ALLOWED_ORIGINS` 来源白名单配置已不再使用。

验证命令（仓库根目录）：

```bash
.venv/bin/python -m pytest tests/test_web_client* xpeech/subapp/channel/web_client/tests
cd xpeech/subapp/channel/web_client/frontend
npm test
npm run build
```
