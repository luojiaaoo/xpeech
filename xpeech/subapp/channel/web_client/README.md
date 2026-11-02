# 网站嵌入浮动聊天组件

业务网站通过 ES Module `embed.js` 接入聊天组件，无需对接 OAuth2。使用 `import()` 动态
加载模块，不需要插入 `<script src>`，也不向 `window` 注册全局 API。登录成功后通过
`init(jwtOrProvider, senderName, title, sessionMetadata)` 传入业务后端签发的 JWT、必填的发送者名称、
助手标题及会话元数据，组件自动初始化
并显示浮动按钮；点击按钮展开聊天窗口。
退出登录时调用 `destroy()`，移除按钮和窗口，并清除内存中的 JWT。

普通页面只提供主会话（`default`），不提供会话切换和新建临时会话入口。
页面不显示「主会话 / 延续主会话上下文」标题栏。
普通页面的聊天请求和表单回答仅允许省略 `x-session-id` 或使用 `default`，不接受临时会话 ID。

iframe 页面只提供临时会话，不显示会话标题栏。左上角半透明悬浮的 **＋ 新建会话** 按钮用于创建独立的临时会话；
新建时清空当前界面、取消当前回复并生成新的 UUIDv7
会话 ID。每条嵌入聊天请求和表单回答都必须携带该 ID（`x-session-id` 请求头），同一对话内
保持不变，后端按用户 ID 与会话 ID 隔离上下文。旧会话的后端记录不会被删除；关闭再打开窗口
会保留当前对话，销毁组件后重新初始化会创建新的对话。

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
| `assistant.destroy()` | 移除按钮和 iframe，取消请求与定时器，清除内存中的身份和 JWT，不发送服务端撤销请求 |

`setUserId()` 和 `setChannelName()` 已移除。`init()` 的无效字符串参数会同步抛出异常，
不初始化组件；验签失败、凭证过期或获取函数失败会显示错误并触发 `xpeech:error`。
也可以直接调用 `assistant.init(jwt, senderName, title, sessionMetadata)`，但组件无法签发新的业务 JWT；该凭证到期后
必须先 `destroy()`，再用新 JWT 初始化。需要自动续期时应使用上述获取函数，业务后端接口
应返回 `{ "jwt": "..." }` 并设置 `Cache-Control: no-store`。
组件尚未初始化或已经销毁时，显示、隐藏、打开、关闭操作不会创建组件；销毁可以重复调用。

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

嵌入临时会话独立于本地账号：不创建用户、不查找同名用户，也不读取本地账号的名称、
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
