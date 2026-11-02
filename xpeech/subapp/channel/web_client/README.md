# 网站嵌入浮动聊天组件

业务网站通过 ES Module `embed.js` 接入聊天组件，无需对接 OAuth2。使用 `import()` 动态
加载模块，不需要插入 `<script src>`，也不向 `window` 注册全局 API。登录成功后通过
`setJWT()` 传入业务后端签发的 JWT，组件自动初始化并显示浮动按钮；点击按钮展开聊天窗口。
退出登录时调用 `destroy()`，移除按钮和窗口，并清理会话。

iframe 内聊天界面左上角的 **＋** 用于新建对话：点击后清空当前界面、取消当前回复并生成新的 UUIDv7
会话 ID。每条嵌入聊天请求和表单回答都必须携带该 ID（`x-session-id` 请求头），同一对话内
保持不变，后端按用户 ID 与会话 ID 隔离上下文。旧会话的后端记录不会被删除；关闭再打开窗口
会保留当前对话，销毁组件后重新初始化会创建新的对话。

嵌入功能默认可用，无需启用开关。在 Web 客户端配置中列出允许接入的业务网站来源
（协议、域名、端口必须匹配，不包含路径）；未配置来源时不允许任何网站嵌入：

```toml
[web_client.embed]
allowed_origins = ["https://portal.example.com", "http://localhost:3000"]
```

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
function onLoginSuccess() {
  assistant.setJWT(loadAssistantJWT);
  // 自动初始化并显示浮动按钮，聊天窗口默认收起。
}

// 用户退出业务网站时调用。
function onLogout() {
  assistant.destroy();
  // 继续执行业务网站原有的退出登录逻辑。
}
```

浮动按钮的创建和点击事件均由组件自动处理，业务网站不需要创建按钮或绑定 DOM 事件。
页面刷新后，如果业务网站恢复了登录状态，在模块加载完成后调用 `onLoginSuccess()`。
单页应用退出时先调用 `destroy()`，下次登录再调用 `setJWT()`；浏览器会复用同一 URL 的模块实例。
单独导入模块不会显示组件或请求会话，只有调用 `setJWT()` 才会初始化。
必须遵循 `setJWT(...) → destroy() → setJWT(...)` 的顺序；初始化后再次调用 `setJWT()`
会抛出异常，即使用户、通道或 JWT 完全相同。切换用户或通道必须先销毁组件，不能自动切换。
自动续期的获取函数也只能返回同一用户、同一通道的 JWT；身份变化会报错，不会替换当前会话。

组件提供以下方法：

| 方法 | 行为 |
| --- | --- |
| `assistant.setJWT(jwtOrProvider)` | 传入 JWT 字符串，或返回 JWT 字符串的同步/异步函数；每次销毁前只允许设置一次 |
| `assistant.hideButton()` | 隐藏浮动按钮，不关闭已打开的聊天窗口 |
| `assistant.showButton()` | 显示浮动按钮，不自动打开聊天窗口 |
| `assistant.open()` | 通过代码主动展开聊天 iframe，无需用户点击浮动按钮 |
| `assistant.close()` | 收起聊天窗口，保留 iframe、聊天内容和会话 |
| `assistant.destroy()` | 移除按钮和 iframe，取消请求与定时器，清除内存身份并撤销已知 token |

`setUserId()` 和 `setChannelName()` 已移除。`setJWT()` 的无效字符串参数会同步抛出异常，
不初始化组件；验签失败、凭证过期或获取函数失败会显示错误并触发 `xpeech:error`。
也可以直接调用 `assistant.setJWT(jwt)`，但组件无法签发新的业务 JWT；该凭证到期后
必须先 `destroy()`，再用新 JWT 初始化。需要自动续期时应使用上述获取函数，业务后端接口
应返回 `{ "jwt": "..." }` 并设置 `Cache-Control: no-store`。
组件尚未初始化或已经销毁时，显示、隐藏、打开、关闭操作不会创建组件；销毁可以重复调用。

JWT 必须由业务后端先验证网站当前用户的登录身份，再为该用户签发，不能直接采用浏览器提交的
任意用户 ID。Header 的 `kid` 必须与服务端 `jwt.channels` 的自定义通道名一致，例如 `crm`
对应 `[jwt.channels.crm]`，使用该通道的密钥签名。禁止使用 `web_client`、`feishu`、`internal`。
通道名为 1–64 个字符，支持字母、数字、点、下划线、冒号和连字符，首字符须为字母或数字。
Payload 的 `sub` 是实际用户 ID，必须为 1–128 字符的字符串，不自动去除空格或转换数字。
`exp` 为必填的数值类型 Unix 秒时间戳；若包含 `nbf`、`iat`，也会校验时间有效性。
签名算法和有效期统一使用全局 `jwt.algorithm`、`jwt.access_token_expire_seconds`；
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

组件服务和聊天后端需配置相同的通道及密钥。聊天、表单回答和文件下载转发时，服务端使用
会话绑定通道的密钥签名，JWT header 的 `kid` 原样使用完整通道名；密钥不传给浏览器。
后续请求不能改用其他通道。
沿用后端的用户 JWT 协议：聊天、表单和文件接口从签名后的 `sub` 读取用户身份，
不从浏览器请求头接受用户 ID。统计接口沿用内部 JWT，由 Web 客户端校验管理员权限后转发。

SDK 使用 `credentials: omit` 向嵌入会话接口提交 `{ "jwt": "..." }`，不接受独立的
`user_id` 或 `channel_name`。服务端按 `kid` 查找配置密钥，验签通过后才使用 `sub`
查找或创建账号。换出的会话有效期取 JWT 剩余时间与全局 `jwt.access_token_expire_seconds`
的较短者。
SDK 通过校验来源和窗口的 `postMessage` 把随机会话 token 交给聊天 iframe，不传递原始业务 JWT。
SDK 和 iframe 均只在内存中保存凭证，不写 Cookie、URL、
localStorage 或 sessionStorage；请求通过 `Authorization: Bearer` 携带。
会话过期前自动续期，页面刷新后重新初始化。

接入错误会显示在聊天窗口中，同时触发业务页面的 `xpeech:error` 事件，详情为
`event.detail.message`。关闭窗口不会退出登录；业务网站退出时必须调用 `destroy()`。
若反向代理设置了 `X-Frame-Options` 或 CSP，必须同时允许业务网站嵌入；HTTPS 网站应通过
HTTPS 加载组件。`/embed.js` 作为公开模块提供跨域加载响应头；嵌入会话接口仍按
`allowed_origins` 限制来源。业务网站配置 CSP 时，需要允许从组件服务地址加载脚本、发起
连接并嵌入 iframe。

验签后的 JWT `sub` 直接对应 Web 客户端的实际账号 ID，不加前缀、不做哈希或来源隔离。
已有账号会复用其资料和后端身份；账号不存在时自动创建普通用户，初始显示名称为用户 ID，
密码为服务端随机值。聊天、表单和文件下载均使用这个实际用户 ID，因此与普通页面访问同一
账号时使用相同的后端身份。用户 ID 沿用后端 `UserJWTAuth` 的校验规则：1–128 个 ASCII 字母、数字、
下划线、连字符或空格，不能全部为空格。

普通账号和管理员账号均可通过嵌入接口登录，保留账号原有角色和权限；已停用账号不能登录。
嵌入会话不能调用 `/api/auth/password` 修改自身密码；管理员账号仍保留用户管理权限。
普通页面原有的密码登录不受影响。

持有通道密钥的业务后端是受信任的身份签发方，必须保护密钥和 JWT 签发接口；浏览器仅接收
签好的短期 JWT。来源白名单不能替代身份认证。不要把服务端密钥或管理员密码放入嵌入代码。

配置由 `web_client/config.py` 独立读取 `conf.toml` 中的 `[web_client.embed]`；
也可用 `XPEECH_WEB_CLIENT__EMBED__ALLOWED_ORIGINS='["https://portal.example.com"]' 覆盖。

验证命令（仓库根目录）：

```bash
.venv/bin/python -m pytest tests/test_web_client* xpeech/subapp/channel/web_client/tests
cd xpeech/subapp/channel/web_client/frontend
npm test
npm run build
```
