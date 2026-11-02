# Xpeech iframe 嵌入接入文档

面向需要把 Xpeech 聊天界面集成到自有业务系统、门户或管理后台的开发者。本文介绍服务端配置、业务用户免重复登录、前端初始化、JWT 自动续签和退出清理。

当前推荐通过 **`embed.js` 加载浮动聊天组件**：组件自动创建 iframe，内置消息展示、附件上传、表单追问和文件下载，业务系统只需提供当前用户的短期 JWT。

- 组件地址：`https://assistant.example.com/embed.js`，替换为实际部署的 **Web 客户端地址**。
- Web 客户端默认端口为 `7939`；Agent API 默认端口为 `7878`，两者用途不同。
- 如果需要自行开发聊天 UI、直接消费 SSE，请参阅 [后端接入文档](backend-integration.md)。

---

## 1. 整体交互模型

```text
业务系统页面                  业务系统后端             Xpeech Web 客户端       Agent API
  |                              |                           |                    |
  |-- 使用原有方式登录 ---------->|                           |                    |
  |-- 加载 /embed.js --------------------------------------->|                    |
  |                              |                           |                    |
  |-- init(loadJWT, ...)          |                           |                    |
  |-- GET /api/assistant/jwt ---->| 校验业务登录身份           |                    |
  |<-- {"jwt":"..."} -----------| 为当前用户签发短期 JWT      |                    |
  |                              |                           |                    |
  |-- GET /api/auth/embed（Bearer JWT）---------------------->| 校验原 JWT         |
  |<-- user_id / expires_in ---------------------------------|                    |
  |-- 组件创建 iframe，使用 postMessage 传递凭证 ------------>|                    |
  |                              |                           |                    |
  |                    iframe 内发起聊天 / 答题 / 文件请求 -->|-- 原 JWT 转发 ----->|
  |                    iframe 内渲染 SSE、表单和下载入口 <----|<-- 事件 / 文件 -----|
  |                              |                           |                    |
  |-- 到期前再次调用 loadJWT ---->|                           |                    |
  |<-- 同一用户、同一通道的新 JWT |                           |                    |
  |-- 更新 iframe 内存凭证 --------------------------------->| 对话保持不变       |
```

核心约定：

1. **业务系统负责用户登录**。Xpeech 根据业务后端签发的 JWT 识别用户，无需用户再次登录 Xpeech，也无需预先创建同名账号。
2. **嵌入必须使用自定义通道**，例如 `crm`；不能使用 `web_client`、`feishu` 或 `internal`。
3. **浏览器只接收签好的短期 JWT**。通道密钥保存在业务后端和 Xpeech 服务端。
4. **使用 JWT 获取函数支持自动续期**。退出、切换用户或切换通道时先 `destroy()`，再按需重新初始化。
5. **iframe 及其通信由组件管理**。仅插入 `<iframe src="https://assistant.example.com/">` 不会完成本文的业务身份接入。

---

## 2. 配置 Xpeech 服务

### 2.1 配置独立通道

在 Xpeech 的 `conf.toml` 中增加业务通道，例如：

```toml
[jwt]
secret_key = "replace-with-an-internal-secret-of-at-least-32-characters"
algorithm = "HS256"
access_token_expire_seconds = 60

[jwt.channels.crm]
secret_key = "replace-with-a-distinct-crm-secret-of-at-least-32-characters"
channel_tags = ["crm"]
```

将示例合并到已有配置，保留原有通道。内部密钥与所有通道密钥必须互不相同，且每个密钥至少 32 个字符。

**Web 客户端和 Agent backend 必须配置相同的 `crm` 通道、密钥及全局算法**。业务后端也使用这份通道密钥和算法签名。Web 客户端将原 JWT 转发给 backend，不会重新签发或延长有效期。

`channel_tags` 为可选项，用于匹配内建技能和 MCP 的通道标签，不是 iframe 来源白名单。通道权限说明见 [后端接入文档](backend-integration.md)。

### 2.2 确认 Web 客户端可访问

已有部署可以直接使用。首次本地启动时，在仓库根目录构建前端：

```bash
npm --prefix xpeech/subapp/channel/web_client/frontend install
npm --prefix xpeech/subapp/channel/web_client/frontend run build
uv run -m xpeech web_client --host 0.0.0.0 --port 7939 --backend-url http://127.0.0.1:7878
```

Agent API 及其依赖服务需已运行。生产环境建议由 HTTPS 域名代理 Web 客户端，下文统一使用 `https://assistant.example.com`。

浏览器应能访问以下资源：

| 资源 | 用途 |
| --- | --- |
| `/embed.js` | ES Module 组件入口，公开访问 |
| `/?embed=1&parent_origin=...` | SDK 创建的 iframe 页面 |
| `/assets/*` | 构建后的页面静态资源 |
| `/api/auth/embed` | 校验业务 JWT |
| `/api/chat`、`/api/answer_question`、`/api/files` | iframe 内的聊天、答题和下载请求 |

当前 SDK 按模块地址的 **origin** 拼接根路径。请把上述资源发布在同一个 origin 的根路径下；仅将模块代理到 `/assistant/embed.js` 不会让其余请求自动带上 `/assistant` 前缀。

---

## 3. 业务后端：为当前登录用户签发 JWT

### 3.1 凭证约定

业务系统自行提供一个获取凭证的接口，例如：

```http
GET /api/assistant/jwt
Cookie: <业务系统原有登录凭证>
```

此路径属于**业务系统**，不是 Xpeech 内置接口。接口必须先验证业务登录身份，再签发该用户的 Token：

```json
{"jwt":"eyJhbGciOiJIUzI1NiIs..."}
```

响应设置 `Cache-Control: no-store`。未登录或登录已失效时返回 `401`，不要根据浏览器提交的任意用户 ID 签发 Token。

JWT 结构：

| 位置 | 字段 | 约定 |
| --- | --- | --- |
| Header | `kid` | 已配置的自定义通道名，如 `crm` |
| Payload | `sub` | 稳定用户 ID，字符串，长度 1–128，只允许 ASCII 字母、数字、空格、连字符和下划线，不能全为空格 |
| Payload | `exp` | 数值类型的 Unix 秒时间戳；建议按全局配置签发 1–60 秒的短期凭证，默认 60 秒 |
| 签名 | 算法与密钥 | 与 Xpeech 全局 `jwt.algorithm` 和对应通道密钥一致 |

嵌入不要求 `aud`。若携带 `nbf`、`iat`，服务端也会校验时间有效性，签发方与 Xpeech 应保持时钟同步。

建议使用带业务前缀的稳定 ID，如 `crm_10001`。显示名称单独通过前端 `senderName` 传入，不要用可变昵称作为 `sub`。同一 `sub` 对应同一用户工作区，应避免不同业务用户或本地账号意外使用相同 ID。

### 3.2 Python 接口示例

以下示例适用于已有 FastAPI 业务系统。将 `your_app.auth.get_current_user` 替换为现有登录认证依赖，并把 `router` 挂载到业务应用；示例假设 `current_user.id` 是稳定的数字用户主键。

业务后端需安装 `pyjwt`，并设置 `ASSISTANT_JWT_SECRET` 环境变量，值为 `jwt.channels.crm.secret_key`。示例使用默认 `HS256` 和 60 秒有效期；服务端修改配置时需同步调整。

```python
import os
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from your_app.auth import get_current_user

router = APIRouter()
JWT_SECRET = os.environ["ASSISTANT_JWT_SECRET"]


@router.get("/api/assistant/jwt")
def assistant_jwt(current_user=Depends(get_current_user)):
    # get_current_user 必须验证当前登录身份，并在未登录时返回 401。
    user_id = f"crm_{current_user.id}"
    token = jwt.encode(
        {
            "sub": user_id,
            "exp": datetime.now(UTC) + timedelta(seconds=60),
        },
        JWT_SECRET,
        algorithm="HS256",
        headers={"kid": "crm"},
    )
    return JSONResponse(
        {"jwt": token},
        headers={"Cache-Control": "no-store"},
    )
```

业务端原有的 HttpOnly 登录 Cookie 只用于浏览器向自身后端获取 JWT。浏览器向 Xpeech 发起的嵌入请求使用 Bearer Token，不依赖跨站 Cookie。

---

## 4. 业务前端：初始化 iframe 组件

### 4.1 加载模块并关联登录生命周期

在业务网站的前端模块中动态导入一次，再在现有登录成功、恢复登录和退出回调中调用：

```javascript
const assistant = await import("https://assistant.example.com/embed.js");

window.addEventListener("xpeech:error", (event) => {
  // 可替换为业务系统的提示组件；不要记录 JWT 本身。
  console.error("聊天组件连接失败：", event.detail.message);
});

async function loadAssistantJWT() {
  const response = await fetch("/api/assistant/jwt", {
    credentials: "same-origin",
    cache: "no-store",
  });
  if (!response.ok) throw new Error("无法获取聊天凭证，请检查业务登录状态");
  const result = await response.json();
  if (typeof result.jwt !== "string" || !result.jwt) {
    throw new Error("业务后端未返回有效 JWT");
  }
  return result.jwt;
}

// 登录成功或页面刷新后恢复登录状态时调用一次。
function onLoginSuccess(currentUser) {
  assistant.init(
    loadAssistantJWT,
    currentUser.display_name,
    "客服助手",
    { source: "portal", department: currentUser.department ?? "" },
  );
}

// 用户退出或准备切换账号时调用，再执行业务系统原有的退出逻辑。
function onLogout() {
  assistant.destroy();
}
```

`currentUser` 来自业务系统自己的登录态，`display_name` 必须为非空字符串。将这两个函数接入现有应用回调后即可使用；SPA 路由切换不要重复初始化。

普通 HTML 页面可把上述代码放在 `<script type="module">` 中；React、Vue 等应用可以放在应用级模块或登录态管理层。DOM 的 `body` 应已创建。如果使用组件生命周期挂载，在卸载时调用 `destroy()`，保证每次初始化都有对应清理。

导入模块本身不会显示组件或发起认证请求，也不会注册 `window.Xpeech` 等全局 API。调用 `init()` 后，SDK 自动创建右下角浮动按钮并开始认证，聊天窗口默认收起；用户点击即可展开。浮动按钮支持鼠标和触屏拖动。

### 4.2 初始化参数

```javascript
assistant.init(jwtOrProvider, senderName, title = "AI 助手", sessionMetadata = {});
```

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `jwtOrProvider` | 是 | JWT 字符串，或返回 JWT 字符串的函数，支持异步；建议使用获取函数 |
| `senderName` | 是 | 发送者显示名称，支持中文，去除首尾空白后不能为空；组件自动进行 UTF-8 URL 编码并通过 `x-sender-name` 传递 |
| `title` | 否 | 默认 `AI 助手`，非空字符串；用于浮窗标题、iframe 标题和欢迎文案 |
| `sessionMetadata` | 否 | 默认 `{}`，必须是可 JSON 序列化的对象，支持嵌套数据；随聊天表单发送，不应包含凭证 |

用户 ID 和通道分别来自已验签 JWT 的 `sub`、`kid`，不用另外调用设置方法。元数据中的 `channel` 不参与通道选择。名称、标题和元数据在初始化时固定，JWT 自动续期时保留；修改这些参数需先销毁再初始化。

`init()` 会立即返回，不返回表示认证完成的 Promise。重复初始化和非法参数会同步抛出异常；异步获取凭证、验签或 iframe 加载失败时，通过窗口内提示及 `xpeech:error` 事件报告。

### 4.3 按钮与窗口控制

| 方法 | 行为 |
| --- | --- |
| `assistant.open()` | 展开聊天窗口 |
| `assistant.close()` | 收起窗口，保留 iframe 和当前对话 |
| `assistant.hideButton()` | 隐藏浮动按钮，不关闭已展开窗口 |
| `assistant.showButton()` | 显示浮动按钮，不自动展开窗口 |
| `assistant.destroy()` | 移除按钮和 iframe，清理组件凭证、定时器和内部事件监听 |

如果使用业务系统自己的「智能助手」按钮，可以在 `init()` 后隐藏默认按钮，再将业务按钮绑定到 `open()`：

```javascript
assistant.hideButton();
document.querySelector("#open-assistant").addEventListener("click", () => {
  assistant.open();
});
```

上例假设页面已有 `id="open-assistant"` 的按钮。显示、隐藏、打开、关闭方法在未初始化或已销毁时不会创建组件；`destroy()` 可重复调用。

### 4.4 直接放置 iframe 的限制

SDK 内部打开的页面为：

```text
https://assistant.example.com/?embed=1&parent_origin=https%3A%2F%2Fportal.example.com
```

该页面会等待父窗口的凭证消息，并检查消息来源和发送窗口。只设置 `src`、把 JWT 加到 URL 或依赖 Xpeech 登录 Cookie，均不能替代 SDK 的初始化流程。

当前 SDK 提供浮动窗口，没有 `container`、自定义尺寸、指定 `session_id` 或主动发送聊天消息的公开参数。业务系统可通过公开方法控制窗口，不应依赖组件内部 DOM 或自行拼接凭证消息。

---

## 5. 自动续签、会话与退出

### 5.1 使用获取函数自动续签

推荐传入 `loadAssistantJWT`，不要写成 `loadAssistantJWT()`：

```javascript
assistant.init(loadAssistantJWT, "张三", "客服助手");
```

SDK 在首次认证、凭证到期前以及 iframe 请求返回 `401` 时调用获取函数，校验并更新凭证。新 JWT 必须属于**同一用户、同一通道**，刷新期间 iframe 和当前会话保持不变。

`401` 会触发刷新，但已经失败的聊天、答题或下载请求不会因此自动重放；用户可以在凭证恢复后按界面提示重试。

也可直接传入固定 JWT 字符串，但到期后无法自动获取新凭证，需要 `destroy()` 后重新初始化。不要通过定时重复调用 `init()` 的方式续签。

### 5.2 会话行为

| 操作 | 行为 |
| --- | --- |
| 首次初始化 | iframe 内生成 UUIDv7 临时会话 ID，聊天和答题自动携带同一 ID |
| 关闭窗口后再次打开 | 保留 iframe、消息和当前会话 |
| 自动刷新 JWT | 保留当前会话 |
| 点击「新建会话」 | 取消当前回复，清空当前展示，生成新的临时会话 ID；不删除旧会话的后端记录 |
| `destroy()` 后重新 `init()`，或刷新业务页面 | 创建新的 iframe 和临时会话，不自动恢复原来的消息展示 |

嵌入页面只提供临时会话，不使用普通网页登录的 `default` 主会话。不同临时会话隔离对话历史，但同一用户的文件工作区和记忆仍共享。

### 5.3 退出与切换身份

严格遵循以下顺序：

```javascript
assistant.destroy();
assistant.init(loadAssistantJWT, nextUser.display_name, "客服助手");
```

上例用于业务系统已完成账号切换，获取函数将返回新用户凭证的场景。即使用户或 JWT 相同，销毁前再次调用 `init()` 也会抛错。

业务退出时调用 `destroy()`，同时按原有流程清除业务系统登录态。组件销毁只清理前端状态，不撤销已签发的 JWT，其在 `exp` 到期前仍有效。

---

## 6. 跨域、代理与身份边界

### 6.1 跨域与页面策略

- 嵌入默认可用，无需开关或来源白名单；旧的 `[web_client.embed].allowed_origins` 配置已不再使用。
- `/embed.js` 提供跨域加载响应头，嵌入请求使用 `Authorization: Bearer` 和 `credentials: omit`，不需要开放跨站 Cookie。
- HTTPS 业务页面也应通过 HTTPS 加载组件，避免混合内容被浏览器拦截。
- 宿主页面的 CSP 需允许组件 origin：`script-src` 允许加载模块，`connect-src` 允许 JWT 校验请求，`frame-src` 允许 iframe。SDK 会创建内联样式，严格限制样式的策略也需兼容该行为。
- Xpeech 嵌入页面设置 `Content-Security-Policy: frame-ancestors *`。反向代理若额外添加 `X-Frame-Options: DENY/SAMEORIGIN` 或更严格的 `frame-ancestors`，仍可能阻止嵌入，需要协调代理策略。
- 聊天是长连接。Web 客户端前方的代理和 backend 前方的代理均需关闭响应缓冲，并允许足够长的读取超时。

### 6.2 用户身份与权限

嵌入认证不创建 Xpeech 本地用户或登录会话，不查找同名本地账号，也不读取其名称、角色或启用状态。业务系统负责当前用户是否可以继续获取 JWT。

嵌入身份只能访问聊天相关接口，不能凭此访问本地账号资料、密码管理、用户管理或管理员统计；JWT 中的角色字段不会授予管理员权限。

这与工作区隔离是两回事：文件和记忆仍以用户 `sub` 定位。通道名不会自动给相同 `sub` 建立独立工作区，因此业务后端应统一规划用户 ID。

SDK 和 iframe 仅在内存中保存 JWT，不把凭证写入 URL、Cookie、localStorage 或 sessionStorage。业务系统也不应把共享密钥写入前端代码。

---

## 7. 本地完整演示

仓库提供 [本地嵌入演示](../examples/embed_demo/README.md)，使用 Python 标准库启动业务页面和 JWT 签发接口，无需安装项目依赖。

在仓库根目录创建 `data/embed-demo/config.json`，此目录已被 Git 忽略：

```json
{
  "assistant_url": "http://localhost:7939/",
  "channel": "crm",
  "user_id": "crm_demo_user",
  "sender_name": "演示用户",
  "title": "客服助手",
  "session_metadata": {"source": "embed_demo", "department": "客服"},
  "algorithm": "HS256",
  "secret_key": "填写 jwt.channels.crm.secret_key 的真实值"
}
```

确保 Web 客户端与 backend 已配置同一通道，且浏览器能访问 `assistant_url`。远端部署时应替换为浏览器可达的地址。

```bash
chmod 600 data/embed-demo/config.json
python3 examples/embed_demo/server.py
```

打开 `http://localhost:8080/`，可验证展开聊天、自动续签、退出和重新初始化。可使用 `--port 8081` 切换演示端口，按 `Ctrl+C` 停止。

演示始终为配置中的固定用户签发 JWT，仅监听 `127.0.0.1`，用于本地验证。实际业务接入时，应使用第 3 节所述的登录态校验接口。

---

## 8. 常见问题与排查

| 现象 | 排查方向 |
| --- | --- |
| `/embed.js` 返回 `404` 或内容是 HTML | 使用 Web 客户端地址，检查代理是否误转到 Agent API 或 SPA 回退页 |
| 模块加载失败或浏览器报 CORS 错误 | 检查模块响应类型、跨域响应头、HTTPS 及宿主 CSP |
| `senderName 必须为非空字符串` | 传入实际显示名称，不能缺失或只含空白 |
| `JWT 格式无效，必须包含自定义通道 kid 和用户 sub` | 检查获取函数返回的是 JWT 字符串，而不是整个 JSON 对象；确认使用自定义 `kid` |
| `/api/auth/embed` 返回 `401` | 检查通道是否配置、签名密钥和算法是否一致、`sub` 是否合法、`exp` 是否过期及服务端时钟 |
| 认证通过但聊天返回 `401` | 检查 backend 是否也配置了同一通道与密钥，以及 JWT 在转发时是否仍有效 |
| `init 只能初始化一次` | 登录恢复、路由挂载或组件重渲染导致重复初始化；先销毁再初始化 |
| `续期 JWT 不能切换用户或通道` | 获取函数返回了不同身份；切换账号时先调用 `destroy()` |
| iframe 空白或提示页面未能加载 | 检查前端构建、`/assets/*`、iframe URL、代理的防嵌入响应头和浏览器控制台 |
| 聊天内容很久后一次性出现 | 检查两层代理的 SSE 缓冲和超时配置 |
| 关闭窗口后仍保留对话 | 正常行为；关闭只收起窗口，退出使用 `destroy()` |
| 本地账号已停用，但嵌入仍能使用 | 嵌入身份独立验证业务 JWT，应由业务系统停止为该用户签发新凭证 |

嵌入错误会触发 `window` 上的 `xpeech:error`，通过 `event.detail.message` 读取原因。业务后端签发失败时也应返回清晰的 HTTP 状态，便于区分业务登录失效和 Xpeech 认证失败。

---

## 9. 接入速查

1. 部署并构建 Web 客户端，确认浏览器可访问其 `/embed.js`、页面和 API。
2. 在 Web 客户端、backend 和业务后端配置相同的自定义通道密钥与算法。
3. 业务后端实现受登录态保护的 `/api/assistant/jwt`，只为当前登录用户签发短期凭证。
4. 前端导入模块，在登录完成后调用 `init(loadAssistantJWT, senderName, title, metadata)`。
5. 退出或切换用户时调用 `destroy()`，按需重新初始化。
6. 验证聊天、追问、文件、自动续签和退出；确认 HTTPS、CSP 与代理长连接配置可用。

组件实现细节与验证命令见 [Web 客户端说明](../xpeech/subapp/channel/web_client/README.md)；自定义聊天 UI 的接口协议见 [后端接入文档](backend-integration.md)。
