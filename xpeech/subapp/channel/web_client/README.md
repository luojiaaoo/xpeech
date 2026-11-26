# 网站嵌入浮动聊天组件

业务网站通过 ES Module `embed.js` 接入聊天组件，无需对接 OAuth2。使用 `import()` 动态
加载模块，不需要插入 `<script src>`，也不向 `window` 注册全局 API。登录成功后通过
`init(jwtOrProvider, senderName, title, sessionMetadata)` 传入业务后端签发的 JWT、可选的发送者名称、
助手标题及会话元数据，组件自动初始化
并显示浮动按钮；点击按钮展开聊天窗口。
退出登录时调用 `destroy()`，移除按钮和窗口，并清除内存中的 JWT。

普通页面与 iframe 页面共用同一套会话工作区，都提供 **历史会话** 和 **＋ 新建会话**。
PC 宽度下历史会话显示在左侧，窄屏下通过抽屉打开；点击历史会话会加载其中保存的消息，
并沿用该会话的 UUIDv7 `session_id` 继续对话。已经打开的会话只隐藏、不卸载，切换或新建会话不会
取消回复，也不会清空消息、文字草稿和待发送附件；正在回复的会话在菜单中显示加载圈。
普通页面的认证仍使用本地账号登录 Cookie，嵌入页面仍只使用 JS 注入的业务 JWT，
两者不会互换认证方式或开放本地账号、管理员接口。

在输入框中键入 `@` 时，通过 `initial_load=true` 请求工作区一级文件和目录，由后端应用与文件搜索相同的排除名单；继续输入文件名即可搜索文件或文件夹。
搜索自动匹配各级目录中名称包含关键词的条目，例如 `@bb` 可以找到 `aabbcc.py`；只匹配文件或目录本身的名称，不匹配路径。
无法匹配时显示空结果；搜索格式无效的错误由前端捕获，按没有匹配文件展示。
搜索接口最多返回 100 个匹配项，引用以带引号的工作区相对路径插入正文。
首次加载、搜索候选和文件选择树均按单行分列显示条目、修改日期（浏览器本地时间）和文件大小：搜索候选先显示名称，后面以灰色显示完整的工作区相对路径，文件选择树只显示名称。文件夹的大小列留空，文件大小自动换算为 B、KB、MB 等单位；被截断的路径或名称可悬停查看。
搜索通过异步 `walk(top_down=True)` 遍历，在进入子目录前剪枝排除 `.git`、`.idea`、`.vscode`、`.venv`、`.local_lib`、`node_modules`、Python 的 `site-packages` 和 `dist-packages` 库目录、Go 的 `pkg/mod` 依赖缓存、`pkg/sum` 校验缓存和常见缓存等目录，再用 `fnmatch` 匹配名称；文件树仍可查看、展开并选择这些目录及文件。
点击输入框旁的 `@` 按钮则打开独立的文件选择树，不触发搜索；首次打开加载根目录，
展开文件夹时才请求其直接子项，已展开的目录在本次打开期间复用结果。
点击文件或文件夹名称将引用插入当前光标或替换选中文字，文件夹引用以 `/` 结尾；点击文件夹左侧箭头展开或收起目录。
每级超过 100 项时可点击「加载更多」。
输入 `/` 可选择技能；只有正文第一个字符为 `/` 时，候选中才包含 `/help`、`/new` 和 `/clear` 指令，
正文中空格后的 `/` 只显示技能。指令按名称前缀搜索，技能按名称和描述搜索；输入与发送均使用普通字符串。
候选支持 ↑ / ↓、Enter / Tab 选择，Esc 关闭；Shift + Enter 仍用于换行。
文件名查询采用 500ms 防抖，连续输入时重新计时，停止输入后再查询；单独输入 `@` 时立即加载一级列表。
输入 `/` 时立即请求全部技能和指令，继续输入时在前端即时筛选，不取消或重发目录请求，也不调用技能或指令搜索接口。退出候选后再次输入 `/` 会重新获取目录。
技能候选沿用后端的通道、组织权限过滤，隐藏内部技能；选择后插入 `[使用 技能名 技能]`，作为普通文字发送给模型。
后端按「带 `filter` 且匹配通过的内置技能 → 用户技能 → 公共内置技能」返回，各组保留原有顺序；用户技能的 `filter` 不参与筛选或排序。
技能与指令候选通过 `/api/chat_command` 加载；文件搜索使用 `/api/files/search`，文件树按目录使用 `/api/files/list`。
这些接口均需鉴权，不需要传入用户 ID。

每次回复通过独立的 `/api/chat/ws` WebSocket 流式接收，请求 JSON 注入 `session_id`，连接本身即用于
关联该会话的事件；回复完成或取消后关闭连接，已打开但没有运行任务的会话不保留空闲 WebSocket。
多个会话可并发回复且互不影响。嵌入 JWT 随请求 JSON 发送，不放在 WebSocket URL 中；普通登录仍使用同源 Cookie。

同一会话在回复过程中也可继续发送。后端返回 `user` 事件，Web 按每条事件的 `turn_id` 归组。
同一轮的用户消息集中显示在回复过程上方。后端开启新一轮时分配新的 `turn_id`，即使上一轮连接仍在关闭，也会显示独立的回复过程。
整轮共用一个可折叠的过程面板和计时，历史回放采用相同的显示顺序。
后端通过 `command` 事件返回“这条消息将用于引导后续回复”，有文本时附上开头最多 40 个字符，后续输出仍从原连接返回。
入队确认复用命令消息渲染，按接收顺序显示在折叠过程外。原连接送来的正文增量继续追加到正在输出的消息，
历史回放也保持正文连续并保留确认提示。
每个请求独立结束，同一 `turn_id` 的请求都结束后收尾该轮的回复过程，所有请求结束后清除会话的运行状态。
回复期间，输入文字后显示发送按钮，输入框为空时显示终止按钮。终止操作按 session 停止后端，
收到确认后关闭该 session 的所有聊天连接。断连和请求错误直接显示，不再轮询等待后端空闲。
每条 Web 聊天连接按接收顺序逐条保存和转发事件，首次写入时清理历史目录。

每轮回复的思考、中间回复和工具调用默认合并折叠，标题显示旋转加载图标、
「工作中 N 秒」和展开箭头，点击可查看过程。助手正文逐块显示在面板外，直到下一条助手正文开始输出，
前一条才收进过程面板。运行期间在回复底部的 Token 统计位置显示三个动画点。
收到 `token_usage` 后停止动画并显示统计，最后一条助手正文留在面板外。实时回复从收到 `user` 事件开始计时，
完成后使用 `token_usage` 中的「会话时长」；没有统计事件时，流结束后停止动画。
历史会话使用相同的折叠方式，文件下载、提问表单和错误提示保持可见。

工具结果携带 `tool_call_id` 和布尔值 `success`，工具结果与执行确认卡片通过同一回合内的 `tool_call_id` 关联。
普通工具调用和执行确认的卡片与标题在成功后显示绿色，失败后显示红色；成功结果正文保持原色。

`send_file` 返回的文件以附件卡片显示，提供下载按钮。`.docx`、`.xls`、`.xlsx`、`.pptx`、
`.md`、`.markdown`、`.pdf`、`.csv`、`.jpg`、`.jpeg`、`.png` 和 `.svg` 文件额外提供预览按钮，在弹窗中查看；Markdown 复用 Ant Design X Markdown，
Word 和 Excel 分别使用 `@js-preview/docx`、`@js-preview/excel`，PowerPoint 使用 `@aiden0z/pptx-renderer`。
预览库按需加载，文件通过现有鉴权下载接口读取，不依赖外部 Office 在线服务。
PDF 使用浏览器内置阅读器，图片按可用空间缩放，SVG 作为图片显示；CSV 使用分页表格，支持带引号的逗号、换行和转义双引号。
旧版 `.doc`、`.ppt` 及其他格式仍可下载，不提供浏览器预览。

本地演示：在 `frontend` 目录运行 `npm run demo:preview`，打开
`http://127.0.0.1:5175/preview-demo.html`。演示页直接读取当前用户下载目录中的 Office 和 Markdown 文件，
无需后端或登录；也可通过 `npm run demo:preview -- "其他目录"` 指定目录。
演示文件接口仅在此命令启动的本机服务器上启用，不接入正式应用。

普通页面和嵌入页面的 `/api/chat_sync`、`/api/chat/stop` 和 `/api/answer_question`
都必须显式提供规范的小写 UUIDv7 `x-session-id`，省略、空值或格式非法均返回 422。
WebSocket `/api/chat/ws` 的启动消息必须提供同样格式的 `session_id`。
Web 客户端不设置默认会话；新会话由前端生成 UUIDv7，同一对话内 ID 保持不变，
后端按用户 ID、通道与会话 ID 隔离上下文。`/api/chat_sync` 的请求头用于 Web 接口校验，
核心后端仍为同步执行生成独立会话，不透传此 ID。
关闭再打开嵌入窗口会保留当前对话，销毁组件后重新初始化会创建新的对话。

Web 对话的原始事件独立保存，不与 Agent 的模型消息历史混用：

```toml
[web_client]
session_history_path = "data/web_client/session_history"
max_history_days = 30
```

文件按 URL 编码后的用户 ID 分目录，文件名为
`channel=[channel]&session_id=[session_id].jsonl`，每行保存一个 JSON 事件，不重复保存文件名中的
通道和会话 ID。通道和用户 ID 均从已验证的 Cookie 用户或 JWT 取得；普通页面默认通道为 `web_client`，
嵌入页面通道为 JWT `kid`。每次流式对话会记录一条用于界面还原的 `user` 事件（包含附件名称）以及
后端 SSE 原始事件，同一会话续聊时逐行追加保存；
`/api/chat_sync` 的独立后台执行不写入该历史。查询或保存时按文件修改时间清理超过保留天数的文件，
并清理由非法文件名标识的历史文件。

登录或嵌入认证完成后，前端通过 `GET /api/history/sessions` 列出当前用户、当前通道的会话，
通过 `GET /api/history/sessions/{session_id}` 读取事件。两个接口都返回 `Cache-Control: no-store`；
读取尚未创建或已清理的会话返回空 `events`。历史中的提问表单仅供查看，新的消息仍使用选中的
`session_id` 发送。

普通登录页可通过 `?channel=crm` 选择 `jwt.channels.web_client.allow_extra_channels`
白名单中的自定义通道，系统名和推荐问题的配置见[主文档](../../../../README.md)。
每项可配置 `organization_filter = ["公司/销售/", "公司/客服/"]`，仅允许服务端查询到的
组织匹配任一前缀的用户登录和使用；省略或空列表表示不限组织，查询失败则拒绝受限通道访问。
页面登录后保留 `channel` 参数，OAuth2 链接登录回跳也会恢复该参数；未指定时始终进入
默认 `web_client`，不根据登录 Cookie 推断通道。该选择绑定到登录会话，不能在登录后的
API 上通过 channel 参数覆盖；会话通道与页面 URL 不一致时，需要重新登录。
`channel` 与 `embed` 参数互斥，嵌入模式必须使用本文的 JS/JWT 认证流程。

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
聊天窗口支持通过四条边或四个角拖拽调整大小，鼠标和触屏均可操作。最小尺寸为
320 × 360px，最大尺寸限制在视口内，并保留 12px 边距；视口较小时会自动缩小以适应屏幕。
收起再打开会保留调整后的大小，调用 `destroy()` 后恢复默认尺寸。

标题栏在关闭按钮左侧依次提供最小化、最大化/还原按钮。最大化占满宿主页面的视口，最小化收起为悬浮按钮，
保留 iframe 和当前会话。窗口大小由用户手动控制，打开或关闭文件预览不会切换窗口模式。
页面刷新后，如果业务网站恢复了登录状态，在模块加载完成后调用 `onLoginSuccess(currentUser)`。
单页应用退出时先调用 `destroy()`，下次登录再调用 `init()`；浏览器会复用同一 URL 的模块实例。
单独导入模块不会显示组件或发起认证请求，只有调用 `init()` 才会初始化。
每次登录只调用一次 `init()`。同一用户、同一通道可更新 JWT；切换用户或通道须先调用 `destroy()` 再初始化。

组件提供以下方法：

| 方法 | 行为 |
| --- | --- |
| `assistant.init(jwtOrProvider, senderName, title = "AI 助手", sessionMetadata = {})` | 初始化凭证、可选的发送者名称、助手标题和会话元数据；每次销毁前只允许设置一次 |
| `assistant.updateJWT(jwt)` | 验证并替换同一用户和通道的 JWT，保留当前会话、草稿和注册配置，返回 `Promise<void>` |
| `assistant.hideButton()` | 隐藏浮动按钮，不关闭已打开的聊天窗口 |
| `assistant.showButton()` | 显示浮动按钮，不自动打开聊天窗口 |
| `assistant.open()` | 通过代码主动展开聊天 iframe，无需用户点击浮动按钮 |
| `assistant.close()` | 收起聊天窗口，保留 iframe、聊天内容和会话 |
| `assistant.maximize()` | 展开并最大化聊天窗口 |
| `assistant.restore()` | 还原为浮动窗口并展开 |
| `assistant.minimize()` | `close()` 的兼容别名，收起聊天窗口并保留会话 |
| `assistant.setInput(text, { timeoutMs? } = {})` | 替换当前对话输入框内容（允许空字符串清空），不自动发送；返回 `Promise<{ session_id }>` |
| `assistant.addFiles(files, { timeoutMs? } = {})` | 将 `File[]` 追加到当前对话框的待发送附件，显示附件列表，不自动上传或发送；返回 `Promise<{ session_id }>` |
| `assistant.newConversation({ timeoutMs? } = {})` | 新建会话，清空对话框及草稿，等待新输入框就绪后返回 `Promise<{ session_id }>` |
| `assistant.switchConversation(session_id, { timeoutMs? } = {})` | 切换到指定会话，等待目标会话就绪后返回 `Promise<{ session_id }>` |
| `assistant.send({ timeoutMs? } = {})` | 发送当前输入框的文字及待发送附件，消息与流式回复显示在对话框，提交后返回 `Promise<{ session_id }>` |
| `assistant.chatSync(text, { files?, timeoutMs? } = {})` | 在当前会话后台运行，支持 `File[]` 附件，不渲染到对话框，返回 `Promise<{ session_id: string, text: string, fragments: string[], files: string[] }>` |
| `assistant.downloadFile(path, { timeoutMs? } = {})` | 按工作区路径下载文件，自动携带当前 JWT 并触发浏览器保存，返回 `Promise<{ session_id }>` |
| `assistant.getFile(path, { timeoutMs? } = {})` | 获取文件二进制，返回 `Promise<File>`，包含文件名和 MIME 类型，供主站 JS 处理，不触发浏览器保存 |
| `assistant.onReply(callback)` | 每次 `fragment` 工具调用回传 `{ session_id, text }`，返回取消订阅函数 |
| `assistant.onChatFile(callback)` | 对话框发送文件时回传 `{ session_id, path }`，返回取消订阅函数 |
| `assistant.registerFragmentHandler(session_id, handler)` | 为嵌入会话注册额外的片段操作卡，返回取消本次注册的函数 |
| `assistant.destroy()` | 移除按钮和 iframe，取消请求与定时器，清除内存中的身份和 JWT，不发送服务端撤销请求 |

`init()` 的无效字符串参数会同步抛出异常，
不初始化组件；验签失败、凭证过期或获取函数失败会显示错误并触发 `xpeech:error`。
如果业务站自己控制续期，可用固定 JWT 初始化。业务后端签发新 JWT 后调用
`await assistant.updateJWT(await loadAssistantJWT())`，旧 JWT 已过期也可以更新。
校验失败会拒绝 Promise，不替换当前凭证。用获取函数初始化时，取得初始 JWT 后即可手动更新，
原获取函数仍负责之后的自动续期。
业务后端的凭证接口应返回 `{ "jwt": "..." }` 并设置 `Cache-Control: no-store`。
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
只发送附件时可不调用 `setInput()`。文字和附件均为空，或对话框正在停止回复时，`send()` 会拒绝 Promise。
`send()` 返回表示已开始发送，不等待模型回复。
后续网络错误显示在对话框中，与点击发送按钮一致。
`newConversation()` 与点击「新建会话」使用相同流程：生成新的 UUIDv7 会话 ID，隐藏当前对话并保留
其消息、文字草稿、待发送附件和后台回复，登录凭证保持不变；切换期间的新命令排队投递到新会话。

已知会话 ID 时，可调用 `await assistant.switchConversation(savedSessionId)`。ID 必须是规范的小写
UUIDv7。目标会话只加载当前用户和通道的历史；没有历史时显示空白会话。已打开的会话保留原有草稿和附件，
切换不会重新登录或展开窗口。

```javascript
// 对话框聊天和 chatSync 中的 fragment 调用都会通知。
const unsubscribe = assistant.onReply(({ session_id, text }) => {
  console.log("片段", session_id, text);
});

assistant.open();
await assistant.setInput("请将订单摘要作为 text 调用 fragment"); // 用户可以继续编辑再发送

try {
  const { session_id, fragments } = await assistant.chatSync("总结当前订单的风险，并将摘要作为 text 调用 fragment", {
    timeoutMs: 300000,
  });
  console.log(session_id, fragments);
} catch (error) {
  console.error("同步交互失败", error);
}

// 不再需要接收片段时调用；destroy() 也会清除全部 onReply 订阅。
unsubscribe();
```

### 嵌入对话框片段操作卡

`fragment` 调用和结果与其他工具一样保留在对话的执行过程里。
只有嵌入会话注册后才会额外显示操作卡：主站的 `handler(text)` 对每个 `fragment` 调用执行一次，
同步返回 `{ title, description, buttons }`。`title` 为非空标题，`description` 为描述字符串，可为空。
`buttons` 是 `{ label, style, callback }[]`，按列表顺序渲染。
`label` 为非空文案。`style` 支持 `default`、`primary`、`dashed`、`text`、`link`，
以及 `danger`（红色）、`warning`（橙色）、`success`（绿色）。样式只影响外观，不会自动弹出确认框。
点击时才执行该按钮的 `callback(text)`，参数是原始片段文本。

```javascript
const { session_id } = await assistant.newConversation();
const unregister = assistant.registerFragmentHandler(session_id, () => ({
  title: '订单片段',
  description: '可以记录片段，或把它填回输入框。',
  buttons: [
    {
      label: '记录',
      style: 'default',
      callback(text) {
        console.log('收到片段', text);
      },
    },
    {
      label: '填入输入框',
      style: 'primary',
      callback: (text) => assistant.setInput(text),
    },
  ],
}));

// 不再显示当前会话的片段时调用。
// unregister();
```

未注册或返回 `buttons: []` 时，不显示额外操作卡。生成函数抛错或返回格式无效时，显示「片段卡片生成失败，请重试」。
再次注册会替换该会话的处理函数。片段通知和 `chatSync()` 返回值不受注册影响；
同步请求不在对话框显示卡片。

### 同步调用与文件

通过 `files` 传入浏览器 `File` 对象数组，文件随本次请求上传，无需提前上传或转 Base64：

```javascript
// fileInput 是 <input type="file" multiple> 元素。
await assistant.chatSync('分析这些文件', {
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

`chatSync()` 返回当前嵌入会话的 `session_id`、本次执行最后一条助手正文 `text`、
`fragment` 工具调用的文本列表 `fragments` 和 `send_file` 指定的文件路径列表 `files`。

```javascript
const { session_id, text, fragments, files } = await assistant.chatSync('生成并发送报告，另外将摘要作为 text 调用 fragment', {
  files: Array.from(fileInput.files),
});
console.log(session_id, text, fragments);
for (const path of files) {
  await assistant.downloadFile(path);
}
```

`text` 保留正文原始格式，没有助手正文时为 `""`。两个列表都按事件中的出现顺序返回，未产生对应内容时为 `[]`。
可通过 `getFile(path)` 单独获取文件内容。
输入选项的 `files` 仍是待上传的 `File[]`，与返回的文件路径列表用途不同。

下载复用 `GET /api/files?path=...`，服务端只允许访问当前用户工作区中的文件。
`downloadFile` 不展开聊天窗口；默认超时 30 秒，可通过 `timeoutMs` 调整。
Promise 在文件获取成功并触发浏览器下载后完成，不表示用户已保存到磁盘。
浏览器可能要求用户允许多文件下载，也可在业务页面的下载按钮中调用该接口。

主站需要把文件存入自己的后台时，调用 `getFile()` 获取 `File`，再使用业务网站自己的认证上传：

```javascript
const file = await assistant.getFile(path);
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
const unsubscribeChatFiles = assistant.onChatFile(({ session_id, path }) => {
  console.log('对话框文件', session_id, path);
});

const { files } = await assistant.chatSync('生成报告');
console.log('sync 文件', files);
// 主站可按需调用 assistant.getFile(path) 或 assistant.downloadFile(path)。
```

`path` 原样保留工作区文件路径；调用 `getFile(path)` 或 `downloadFile(path)` 时，SDK 会构造请求并自动携带当前凭证。
也可监听主页面的 `CustomEvent('xpeech:chat-file')`，其 `detail` 与该订阅参数相同；选择一种通知方式即可。
sync 文件只随本次调用的结果返回，不派发文件订阅或文件事件通知。
SDK 本身收到通知不会自动下载；调用方决定如何处理。`destroy()` 会清除文件订阅。

片段还会通过主页面的 `window` 派发 `CustomEvent("xpeech:reply")`，其 `detail`
与 `onReply` 参数相同。选择一种订阅方式即可，避免重复处理。
流式对话收到 `fragment` 工具调用时逐条通知；同步请求成功后按调用顺序逐条通知。
普通模型回复、思考内容和其他工具事件不会触发通知。若需要区分同一会话内的并发同步请求，
直接读取每次 `chatSync()` 返回的 `fragments`。

`chatSync()` 使用当前 JWT、发送者名称及初始化时的 `sessionMetadata`，
保留输入框草稿和待发送附件，只携带调用时通过 `files` 显式传入的附件。它使用后端的非交互模式，不等待用户填写问题表单；
后端为每次同步执行生成独立会话，不读取也不写入对话历史。
最终助手正文、片段和文件路径通过 Promise 返回，不渲染到对话框，窗口收起时也可执行。
需要模型输出 JSON 对象时，在消息文本中加入完整标记 `::请严格以JSON格式回复::`，例如
`assistant.chatSync("总结当前订单的风险。::请严格以JSON格式回复::")`。仅本次请求生效，需模型配置支持；
不支持时返回 HTTP 400。启用后，`text` 中的正文仍以字符串返回，调用方可用 `JSON.parse(text)` 解析。
同步调用可与对话框手动聊天、其他同步调用并发执行，互不阻塞。
请求失败或后台执行出错时，Promise 会以错误拒绝。

`setInput`、`addFiles`、`newConversation`、`switchConversation`、`send` 默认等待 30 秒，
`chatSync` 默认等待 5 分钟，可通过 `timeoutMs` 调整。
超时只停止主页面等待，不代表后端执行已取消，请勿据此自动重复提交。
未初始化、认证失败、会话切换、页面重载或销毁时，相应 Promise 会拒绝；JWT 自动续期不切换会话。
双方都校验 `postMessage` 的 origin 和窗口来源，请求和响应还使用 `request_id`、`session_id` 关联。

业务主站只通过嵌入 SDK 接入。同步结果和通知仅包含上文列出的字段；
Agent 原生事件以及 Web 界面的流式和历史协议是内部实现，不属于嵌入 SDK 契约。

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

`user_id` 由 JWT 的 `sub` 设置；`sender_name` 可以通过 `init` 的第二个参数 `senderName` 设置：

```javascript
assistant.init(loadAssistantJWT, "张三", "客服助手", { source: "portal", department: "客服" });
assistant.init(loadAssistantJWT, undefined, "客服助手"); // 或由服务端查询发送者名称，二选一。
```

指定了名称时，组件会将其进行 UTF-8 URL 编码，通过 HTTP 的 `x-sender-name` 请求头或 WebSocket 的 `sender_name` 字段传递。支持中文；名称必须为字符串，去除首尾空白后不能为空，否则同步抛出异常。自动续期保留初始化配置，`destroy()` 后清除。
省略名称或传 `undefined` 时，Web 服务在聊天请求中按已验证的 JWT `sub` 查询 `[extra_user_info]`，用其中的 `username_claim` 提取名字。需同时配置查询命令；查不到名字时聊天返回错误，不使用 OAuth2 的 `fallback_username`。
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
有效秒数 `expires_in`，不创建用户、登录会话或新 token。

校验通过后，SDK 通过校验来源和窗口的 `postMessage` 将原 JWT、用户 ID 和发送者名称
交给聊天 iframe。SDK 和 iframe 均只在内存中保存凭证，不写 Cookie、URL、localStorage
或 sessionStorage；所有嵌入请求使用 `credentials: omit`。
使用 JWT 获取函数时，SDK 会在过期前重新调用它，并校验新 JWT 仍属于同一用户和通道；
iframe 和当前对话保持不变。请求返回 401 时也会触发重新获取。使用固定 JWT 字符串时，
由业务后端签发新凭证，再通过 `updateJWT()` 更新；无需销毁组件。

`destroy()` 只销毁前端组件并清除内存凭证，不撤销已签发的 JWT；原 JWT 在 `exp` 到期前
仍有效。普通网页登录继续使用 HttpOnly Cookie 和 `web_client_sessions` 表，其退出、停用及密码
管理逻辑保持原有行为。

接入错误会显示在聊天窗口中，同时触发业务页面的 `xpeech:error` 事件，详情为
`event.detail.message`。关闭窗口不会退出登录；业务网站退出时必须调用 `destroy()`。
若反向代理设置了 `X-Frame-Options` 或 CSP，必须同时允许业务网站嵌入；HTTPS 网站应通过
HTTPS 加载组件。`/embed.js` 作为公开模块提供跨域加载响应头；嵌入认证接口允许
任意来源访问，但仍校验 JWT。业务网站配置 CSP 时，需要允许从组件服务地址加载脚本、发起
连接并嵌入 iframe。

嵌入会话独立于本地账号：不创建用户、不查找同名用户，也不读取本地账号的名称、
角色或启用状态。聊天、表单回答和文件下载直接使用已验签 JWT 的 `sub` 作为用户 ID；
聊天发送者名称取每次请求的必填请求头 `x-sender-name`，缺失或为空时返回 422。
用户 ID 沿用后端 `UserJWTAuth` 的校验规则：1–128 个 ASCII 字母、数字、下划线、连字符
或空格，不能全部为空格。

业务 JWT 仅用于嵌入认证和聊天相关接口，
不能访问本地账号资料、修改密码、管理用户或查看管理员统计；即使 `sub` 与本地管理员
相同也不会获得管理员权限。本地账号的停用、删除或角色变化不影响同名嵌入身份，
业务网站负责其用户的登录认证和 JWT 签发。普通页面继续使用本地账号登录。

持有通道密钥的业务后端是受信任的身份签发方，必须保护密钥和 JWT 签发接口；浏览器仅接收
签好的短期 JWT。不要把服务端密钥或管理员密码放入嵌入代码。

验证命令（仓库根目录）：

```bash
.venv/bin/python -m pytest tests/test_web_client* xpeech/subapp/channel/web_client/tests
cd xpeech/subapp/channel/web_client/frontend
npm test
npm run build
```
