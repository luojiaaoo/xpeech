# 嵌入聊天：三步接入

通过 `embed.js` 在业务网站显示浮动聊天窗口。业务系统负责登录和签发 JWT，组件负责 iframe、聊天、附件、追问表单和文件下载。宿主页面还可以操作会话、接收结果，或在窗口收起时运行同步任务。用户沿用业务系统账号，无需另行登录 Xpeech。

准备好 **Web 客户端地址**（默认端口 `7939`），然后完成下面三步。本文以 `https://assistant.example.com`、通道 `crm`、算法 `HS256` 为例。

## 1. 配置通道

在 Xpeech 现有的 `conf.toml` 中添加：

```toml
[jwt.channels.crm]
secret_key = "replace-with-a-distinct-crm-secret-of-at-least-32-characters"
```

**Web 客户端、Agent backend 和业务后端使用相同的通道密钥与 `jwt.algorithm`。** 密钥至少 32 个字符，且与内部密钥及其他通道密钥不同；只保存在后端。

嵌入必须使用自定义通道，不能使用 `web_client`、`feishu` 或 `internal`。

## 2. 业务后端提供 JWT 接口

实现你自己的 `GET /api/assistant/jwt`：验证当前业务登录态，返回 `{"jwt": "..."}`，响应设置 `Cache-Control: no-store`。未登录返回 `401`。

FastAPI 示例（安装 `pyjwt`，替换登录依赖，并将 `router` 挂载到业务应用）：

```python
import os
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from your_app.auth import get_current_user  # 替换为业务系统现有的登录认证

router = APIRouter()
JWT_SECRET = os.environ["ASSISTANT_JWT_SECRET"]  # 与 jwt.channels.crm.secret_key 相同


@router.get("/api/assistant/jwt")
def assistant_jwt(current_user=Depends(get_current_user)):
    token = jwt.encode(
        {
            "sub": f"crm_{current_user.id}",  # 当前登录用户的稳定 ID
            "exp": datetime.now(UTC) + timedelta(seconds=60),
        },
        JWT_SECRET,
        algorithm="HS256",
        headers={"kid": "crm"},
    )
    return JSONResponse({"jwt": token}, headers={"Cache-Control": "no-store"})
```

JWT 只需关注三个字段：

| 字段 | 要求 |
| --- | --- |
| Header `kid` | 已配置的自定义通道名，如 `crm` |
| Payload `sub` | 稳定用户 ID，1–128 个 ASCII 字母、数字、空格、`-` 或 `_`，不能全为空格 |
| Payload `exp` | 过期时间，以数值表示的 Unix 秒时间戳。示例有效期 60 秒，实际签发时可与 `jwt.access_token_expire_seconds` 保持一致 |

用户 ID 从业务登录态取得，不接受浏览器任意指定。同一 `sub` 的文件工作区和记忆共享，通道不会隔离它们；建议加业务前缀，避免与其他业务或本地账号重名。显示名称在下一步单独传入。

## 3. 业务前端关联登录和退出

模块只导入一次。普通 HTML 放在 `<script type="module">` 中；React / Vue 放在应用级模块或登录态管理层，确保调用时 `document.body` 已存在。

```javascript
const assistant = await import("https://assistant.example.com/embed.js");

async function loadAssistantJWT() {
  const response = await fetch("/api/assistant/jwt", {
    credentials: "same-origin",
    cache: "no-store",
  });
  if (!response.ok) throw new Error("无法获取聊天凭证");
  return (await response.json()).jwt;
}

// 接到现有的登录成功 / 刷新后恢复登录回调，每次登录只调用一次。
function onLoginSuccess(user) {
  assistant.init(loadAssistantJWT, user.display_name, "客服助手", {
    source: "portal",
  });
}

// 接到现有的退出 / 切换账号回调。
function onLogout() {
  assistant.destroy();
}

window.addEventListener("xpeech:error", (event) => {
  console.error("聊天组件：", event.detail.message);
});
```

完成后，右下角会出现可拖动的聊天按钮，点击展开窗口。

**把 `loadAssistantJWT` 函数传给 `init()`，组件会在需要时调用它获取或续签 JWT。** 续签必须返回同一用户、同一通道的 JWT，当前对话会保留。请求因 `401` 失败后，组件会尝试续签，失败的业务请求由宿主决定是否重新发起。

每次初始化都要有对应的 `destroy()`。SPA 路由切换或组件重渲染不要重复调用 `init()`；切换账号时，先销毁，再完成业务账号切换并重新初始化。

## 初始化与窗口控制

```javascript
assistant.init(jwtOrProvider, senderName, title = "AI 助手", sessionMetadata = {});
```

| 参数 | 说明 |
| --- | --- |
| `jwtOrProvider` | 必填。返回 JWT 字符串的函数，支持异步；也接受固定 JWT 字符串，但无法自动续签 |
| `senderName` | 可省略，也可传 `undefined` 或 `null`。聊天服务会按 JWT `sub` 使用 `[extra_user_info].username_claim` 提取名字。若提供，必须是非空字符串，支持中文 |
| `title` | 窗口标题及欢迎文案，默认 `AI 助手`，不能为空 |
| `sessionMetadata` | 初始化时设置的 JSON 对象，随流式聊天和同步调用一起发送，默认 `{}`。通道取自 JWT 的 `kid`，已配置的 `[extra_user_info].session_metadata_claims` 同名字段由服务端提供 |

`init()` 启动异步认证并立即返回。重复初始化或参数非法会同步抛错，异步连接错误通过 `xpeech:error` 报告。修改初始化参数需先 `destroy()`。

例如 `assistant.init(loadAssistantJWT, undefined, "客服助手")` 会保留指定标题，由服务端在聊天时查名字。要省略姓名，请同时配置 `[extra_user_info]` 的 `command_extra_user_info_template` 和 `username_claim`；查不到可用的名字时聊天会返回错误，不使用 OAuth2 的 `fallback_username`。

| 方法 | 用途 |
| --- | --- |
| `updateJWT(jwt)` | 验证并更新同一用户、同一通道的 JWT，返回 `Promise<void>`。传入固定 JWT 初始化时可用此方法手动续期 |
| `open()` / `close()` | 展开 / 收起窗口，保留当前对话 |
| `maximize()` / `restore()` | 展开窗口并最大化 / 恢复为浮动窗口 |
| `minimize()` | `close()` 的别名 |
| `hideButton()` / `showButton()` | 隐藏 / 显示浮动按钮，不改变窗口展开状态 |
| `destroy()` | 移除组件，清理凭证、订阅和注册配置。可重复调用，已签发的 JWT 到期前仍有效 |

使用业务页面自己的按钮：

```javascript
assistant.hideButton(); // 在 init() 后调用
document.querySelector("#open-assistant").addEventListener("click", () => {
  assistant.open();
});
```

## 控制对话与恢复会话

调用 `init()` 后即可发起下列操作。SDK 会等待认证和输入框就绪，窗口保持当前展开状态，需要显示时调用 `open()`。
这些方法都返回 `Promise<{ session_id: string }>`，可在最后一个参数中传入 `{ timeoutMs }`，默认等待 30 秒。

| 方法 | 用途 |
| --- | --- |
| `newConversation(options?)` | 打开一个新对话，可能复用已打开且尚未开始、没有历史记录的会话。使用返回的 `session_id` 获取实际会话 ID |
| `switchConversation(session_id, options?)` | 切换到指定会话并等待输入框就绪。ID 必须是规范的小写 UUIDv7 |
| `setInput(text, options?)` | 替换当前会话的输入框内容，传 `""` 可清空 |
| `addFiles(files, options?)` | 追加待发送附件，参数为浏览器 `File[]`。附件在发送时上传 |
| `send(options?)` | 提交当前输入框和附件。Promise 完成表示提交已开始，流式回复和后续错误显示在聊天窗口 |

窗口收起和 JWT 续期会保留当前会话。切换时，已打开的会话保留草稿、待发送附件和正在进行的回复。
刷新页面或销毁后重新初始化时，组件默认选中新会话。当前用户和通道已保存的对话可以从「历史会话」恢复，
也可以由宿主记录 `session_id`，在下次进入业务页面时主动切回。首次打开没有历史记录的会话时，界面显示为空白。

```javascript
// savedSessionId 来自宿主保存的业务记录，files 为用户选取的 File[]。
async function analyzeOrder(savedSessionId, files) {
  assistant.open();
  const { session_id } = savedSessionId
    ? await assistant.switchConversation(savedSessionId)
    : await assistant.newConversation();
  await assistant.setInput("请分析订单附件");
  await assistant.addFiles(files);
  await assistant.send();
  return session_id; // 宿主可把实际会话 ID 保存到订单记录。
}
```

## 同步调用与结果

`chatSync(text, { files?, timeoutMs? })` 使用当前 JWT、发送者名称和初始化时的元数据，在后台独立执行。
它保持聊天窗口、草稿和待发送附件的状态，上传的附件由本次调用的 `files` 指定。有附件时可传空文本。
后端为每次同步执行创建独立的非交互会话，无需用户回答追问，同步结果也不会加入界面的对话历史。

```typescript
type ChatSyncResult = Promise<{
  session_id: string;
  text: string;
  fragments: string[];
  files: string[];
}>;
```

| 字段 | 内容 |
| --- | --- |
| `session_id` | 接收请求的嵌入会话 ID，用于宿主关联结果。后端本次执行使用自己的独立会话 |
| `text` | 本次执行最后一条助手正文，保留原始格式。没有正文时返回 `""` |
| `fragments` | 本次 `fragment` 工具调用传出的文本，按调用顺序排列 |
| `files` | 本次 `send_file` 产出的工作区路径，按出现顺序排列 |

未产生对应内容时，`fragments` 和 `files` 各自返回空数组。

```javascript
// fileInput 是宿主页面的文件输入框。
const { session_id, text, fragments, files } = await assistant.chatSync(
  "请分析附件并生成报告",
  { files: Array.from(fileInput.files) },
);
console.log({ session_id, text, fragments, files });
```

需要 JSON 对象时，启用 `llm.support_json_output`，并在请求正文中加入完整标记
`::请严格以JSON格式回复::`。此时 `text` 仍是字符串，有正文时可用 `JSON.parse(text)` 解析。
每个同步 Promise 对应本次调用的结果，可与界面聊天或其他同步请求并发执行。
默认等待 5 分钟，超时只结束宿主页面的等待，后端可能仍在运行。

## 片段通知和操作卡

`onReply(callback)` 接收 `{ session_id, text }`。这里的 `text` 来自 `fragment` 工具调用，界面聊天和成功的同步调用都会发送此通知。
它也可通过 `window` 上的 `xpeech:reply` 事件接收，任选一种方式即可。订阅方法返回取消订阅函数。

要在聊天中显示由宿主定义的按钮，对目标会话调用 `registerFragmentHandler(session_id, handler)`。
`handler(text)` 同步返回 `{ title, description, buttons }`。标题和按钮标签须为非空字符串，描述可为空字符串。
按钮包含 `label`、`style`、`callback`，样式支持 `default`、`primary`、`dashed`、`text`、`link`、`danger`、`warning`、`success`。
用户点击后，宿主的 `callback(text)` 收到原始片段。同步调用的片段通过返回值和通知交给宿主，操作卡显示在界面聊天中。

```javascript
// session_id 来自 newConversation() 或 switchConversation()，suggestionInput 是宿主的表单控件。
const unregister = assistant.registerFragmentHandler(session_id, () => ({
  title: "工单建议",
  description: "将片段填入业务表单，确认后再保存。",
  buttons: [{
    label: "填入处理建议",
    style: "primary",
    callback(text) {
      suggestionInput.value = text;
    },
  }],
}));
// 离开业务页面时可调用 unregister()。
```

测试时可在对话中发送「请调用 fragment，把处理建议作为 text 传入」。
同一会话再次注册会替换原处理函数，`destroy()` 会清理注册和订阅。

## 获取和下载文件

界面聊天发出的文件通过 `onChatFile(callback)` 通知宿主，参数为 `{ session_id, path }`，返回取消订阅函数。
也可监听 `window` 上的 `xpeech:chat-file` 事件。同步调用的文件路径通过该次结果的 `files` 取得。

| 方法 | 用途 |
| --- | --- |
| `downloadFile(path, options?)` | 按工作区路径获取文件并触发浏览器下载，返回 `Promise<{ session_id: string }>` |
| `getFile(path, options?)` | 返回 `Promise<File>`，包含文件名、MIME 类型和二进制内容，可用 `FormData` 上传到业务后端 |

两种方法都自动携带当前 JWT，默认等待 30 秒，可通过 `{ timeoutMs }` 调整。
路径直接使用文件通知或同步结果中的值。下载 Promise 在浏览器下载触发后完成。

## 启用网页操作

调用 `registerPageAgent({ model, baseURL, apiKey })` 启用网页操作，可以在 `init()` 前后注册。
`model` 和 `baseURL` 必填，`apiKey` 可选。组件在首次点击「网页操作」时加载 Page Agent。
`baseURL` 为 OpenAI 兼容 API 的根地址，组件会请求其下的 `/chat/completions`。模型须支持工具调用。 [Page Agent 接入说明](https://github.com/alibaba/page-agent/blob/main/docs/README-zh.md)

```javascript
assistant.init(loadAssistantJWT, user.display_name, "客服助手", { source: "portal" });
assistant.registerPageAgent({
  model: "your-model-name",
  baseURL: "https://model.example.com/v1",
});
```

浏览器直连模型服务时，可传浏览器可用的 `apiKey`，模型服务需允许业务站点跨域请求。网页操作运行于业务页面，任务和可见页面内容会发送到配置的模型服务。请勿将长期模型密钥放到初始化代码里。

重复注册会替换模型配置并停止当前网页任务。调用 `destroy()` 会清除注册和模型配置，下次登录时可重新注册。

网页操作直接适配现有控件，可浏览只读链接、切换标签页、查询和筛选、填写常用输入框、选择选项及进入本地步骤。插件根据控件文字、无障碍名称和表单语义拦截可识别的保存、提交、删除等动作。原生表单仅放行可识别的 GET 查询和本地对话框操作。涉及站点自定义写入或自动保存时，助手根据页面提示判断，并将可能写入后台的步骤交由用户操作。

```html
<label>联系人 <input name="contact"></label>
<button type="button">下一步</button>
<button type="submit">提交</button>
```

打开悬浮聊天后，可用标题栏的「网页操作」开关显示或收起 Page Agent 原生面板。面板打开后，输入任务并按 Enter 执行。两个窗口可以同时打开。收起网页操作面板会停止当前网页任务，已填写的内容保留在业务页面上，提交由用户亲自点击。

## 部署与排错

嵌入默认可用，无需配置来源白名单。由 SDK 创建浮动窗口、iframe 并传递 JWT，用户可拖拽调整窗口大小。

浏览器通过 Web 客户端的 `/api/chat/ws` 接收流式回复，反向代理需支持 WebSocket Upgrade。
Web 客户端再通过 SSE 从 Agent backend 接收事件，backend 前方的代理需关闭 SSE 响应缓冲并保留足够的读取超时。
同步调用通过普通 HTTP 等待整次执行完成，也需为代理设置足够的超时。

| 现象 | 检查 |
| --- | --- |
| `embed.js` 返回 404 或 HTML | 使用 Web 客户端地址；确认前端已构建、代理未转到 Agent API 或 SPA 回退页 |
| 页面、静态资源或 API 404 | SDK 从模块的 origin 根路径加载资源；需代理 `/embed.js`、`/`、`/assets/*`、`/api/*`，不支持只代理 `/assistant/embed.js` |
| 模块加载失败 / iframe 被拦截 | HTTPS 页面使用 HTTPS 组件；宿主 CSP 允许组件域名的 `script-src`、`connect-src`、`frame-src`，并兼容 SDK 内联样式；代理允许跨域加载及 iframe 嵌入 |
| `/api/auth/embed` 返回 401 | 通道、密钥、算法、用户 ID、过期时间和服务端时钟 |
| 认证成功，聊天返回 401 | Agent backend 是否配置了同一通道、密钥和算法 |
| 认证成功，流式聊天无法连接或中途断开 | 检查 `/api/chat/ws` 的 WebSocket Upgrade 和代理读取超时 |
| `init 只能初始化一次` | 重复调用了 `init()`；先 `destroy()` 再初始化 |
| `更新 JWT 不能切换用户或通道` | 检查续签接口返回的 `sub` 和 `kid`，切换身份时先 `destroy()` 再初始化 |
| 流式正文最后一次性出现 | 检查 Web 客户端到 Agent backend 之间的代理是否缓冲 SSE 响应 |
| 同步调用返回「无效的同步聊天响应」 | 确认 `embed.js` 和 iframe 前端产物来自同一版本，并检查代理或 CDN 是否仍在返回旧产物 |

本地验证直接使用 [嵌入演示](../examples/embed_demo/README.md)。部署命令见 [项目 README](../README.md)，组件实现细节见 [Web 客户端说明](../xpeech/subapp/channel/web_client/README.md)。需要自己开发聊天 UI 时，参阅 [后端 API 接入](backend-integration.md)。
