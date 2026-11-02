# 嵌入聊天：三步接入

通过 `embed.js` 在业务网站显示浮动聊天窗口。业务系统负责登录和签发 JWT，组件负责 iframe、聊天、附件、表单和文件下载，无需创建 Xpeech 账号。

准备好 **Web 客户端地址**（默认端口 `7939`，不是 Agent API 的 `7878`），然后完成下面三步。本文以 `https://assistant.example.com`、通道 `crm`、算法 `HS256` 为例。

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
| Payload `exp` | 过期时间，Unix 秒时间戳；示例有效期 60 秒 |

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

**传入 `loadAssistantJWT` 函数，组件会自动续签；不要写成 `loadAssistantJWT()`。** 续签必须返回同一用户、同一通道的 JWT，不影响当前对话。请求因 `401` 失败后会触发续签，但不会自动重发该请求。

每次初始化都要有对应的 `destroy()`。SPA 路由切换或组件重渲染不要重复调用 `init()`；切换账号时，先销毁，再完成业务账号切换并重新初始化。

## API 速查

```javascript
assistant.init(jwtOrProvider, senderName, title = "AI 助手", sessionMetadata = {});
```

| 参数 | 说明 |
| --- | --- |
| `jwtOrProvider` | 必填。返回 JWT 字符串的函数，支持异步；也接受固定 JWT 字符串，但无法自动续签 |
| `senderName` | 必填。非空显示名称，支持中文，如 `张三` |
| `title` | 窗口标题及欢迎文案，默认 `AI 助手`，不能为空 |
| `sessionMetadata` | 随聊天发送的 JSON 对象，默认 `{}`；其中的 `channel` 不参与通道选择 |

`init()` 立即返回，不返回认证完成的 Promise。重复初始化、非法参数会同步抛错；异步连接错误通过 `xpeech:error` 报告。修改初始化参数需先 `destroy()`。

| 方法 | 用途 |
| --- | --- |
| `open()` / `close()` | 展开 / 收起窗口，保留当前对话 |
| `hideButton()` / `showButton()` | 隐藏 / 显示浮动按钮，不改变窗口展开状态 |
| `destroy()` | 移除组件，清理内存凭证；可重复调用，已签发 JWT 到期前仍有效 |

使用业务页面自己的按钮：

```javascript
assistant.hideButton(); // 在 init() 后调用
document.querySelector("#open-assistant").addEventListener("click", () => {
  assistant.open();
});
```

关闭再打开、自动续签均保留对话；点击「新建会话」、销毁后重建或刷新页面会开启新对话，不恢复旧消息展示。旧会话的后端记录不会被删除。

## 部署与排错

嵌入默认可用，无需来源白名单。SDK 自动管理 iframe 和凭证通信，直接插入 `<iframe>` 无法完成业务身份接入。当前仅支持浮动窗口，没有 `container`、自定义尺寸或指定会话 ID 的公开参数。

| 现象 | 检查 |
| --- | --- |
| `embed.js` 返回 404 或 HTML | 使用 Web 客户端地址；确认前端已构建、代理未转到 Agent API 或 SPA 回退页 |
| 页面、静态资源或 API 404 | SDK 从模块的 origin 根路径加载资源；需代理 `/embed.js`、`/`、`/assets/*`、`/api/*`，不支持只代理 `/assistant/embed.js` |
| 模块加载失败 / iframe 被拦截 | HTTPS 页面使用 HTTPS 组件；宿主 CSP 允许组件域名的 `script-src`、`connect-src`、`frame-src`，并兼容 SDK 内联样式；代理允许跨域加载及 iframe 嵌入 |
| `/api/auth/embed` 返回 401 | 通道、密钥、算法、用户 ID、过期时间和服务端时钟 |
| 认证成功，聊天返回 401 | Agent backend 是否配置了同一通道、密钥和算法 |
| `init 只能初始化一次` | 重复调用了 `init()`；先 `destroy()` 再初始化 |
| `续期 JWT 不能切换用户或通道` | 账号切换前未销毁组件 |
| 聊天内容最后一次性出现 | Web 客户端与 backend 前方的代理均关闭 SSE 响应缓冲，并延长读取超时 |

本地验证直接使用 [嵌入演示](../examples/embed_demo/README.md)。部署命令见 [项目 README](../README.md)，组件实现细节见 [Web 客户端说明](../xpeech/subapp/channel/web_client/README.md)。需要自己开发聊天 UI 时，参阅 [后端 API 接入](backend-integration.md)。
