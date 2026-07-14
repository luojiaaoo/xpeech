# playwright-mcp 调用协议

所有浏览器操作只允许通过 `playwright-mcp` 完成。

本地 HTML 必须先用内置 `create_browser_preview` 托管。该工具仅复制和托管静态文件，不属于浏览器操作；获得预览 URL 后，才能调用 `playwright-mcp` 导航和验证。

## 工具来源检查

调用前检查已注册 MCP server 与工具来源：

- server 名称应明确对应 `playwright-mcp` / `playwright`。
- 允许的工具族为该 server 暴露的 `browser_*` 工具。
- 同名工具若来自其他 server，不得调用。
- `playwright-mcp` 不存在或连接失败时，将浏览器验证标记为 blocked，不做本地或跨工具 fallback。

## 禁止项

- 禁止 shell 调用 `npx playwright`、`playwright install` 或任何浏览器可执行文件。
- 禁止在 Python/Node.js 中 import/require Playwright、Puppeteer、Selenium。
- 禁止通过 CDP/WebSocket 连接本地浏览器。
- 禁止使用其他浏览器 MCP 替代。
- 禁止自行启动 `python -m http.server`、`npx serve`、Vite dev server 或任何其他 HTTP 服务。
- 禁止仅凭代码阅读宣称页面已通过视觉验证。

## 托管本地 HTML

在调用任何 `playwright-mcp` 工具前：

1. 对工作区内的 deck 目录调用 `create_browser_preview`。
2. 传入目录时，工具返回该预览目录的 URL 前缀；将 `index.html` 追加到该 URL 后作为 deck 入口。
3. 传入单个 HTML 时，工具返回完整文件 URL，直接使用。
4. 将返回的 URL 交给 `playwright-mcp` 的 `browser_navigate`。

`create_browser_preview` 会复制当前文件，因此每次修改页面后都要重新调用，并对新 URL 回归验证。如果 `create_browser_preview` 缺失或托管失败，将浏览器验证标记为 `blocked`；不得自行开启 HTTP 服务兜底。

## 通用验证提示词

> 只使用 playwright-mcp 验证 `<URL>`。先确认当前调用工具来自 playwright-mcp；将视口设为 `<WIDTH>x<HEIGHT>`，导航后等待 `document.fonts.ready`、所有图片完成加载且布局稳定。先读取 accessibility snapshot，再截图；读取控制台 warning/error 与页面异常。检查白屏、破图、字体回落、文字溢出、裁切、重叠、字号、对比度和页面边缘安全区。返回 `status`、`url`、`viewport`、`screenshots`、`console_errors`、`page_errors`、`visual_issues`。不要调用本地浏览器、浏览器 CLI 或其他浏览器 MCP。

## 逐页 deck 提示词

> 只使用 playwright-mcp 打开 `<DECK_URL>`，视口设为 1920×1080。读取首页 snapshot 与截图，然后用 `ArrowRight` 逐页前进，共检查 `<COUNT>` 页。每页等待字体、图片和动画稳定，记录标题、页码、截图和新增控制台错误。检查空白页、页序、错位、裁切、跨页样式污染和 localStorage/hash 恢复。最后确认实际页数等于 `<COUNT>`。不要使用任何本地浏览器控制。

多文件 deck 还应逐个导航 `slides/*.html`，隔离确认每页自身没有报错。

## 多视口提示词

> 只使用 playwright-mcp 对 `<URL>` 依次设置 `<VIEWPORTS>`。每个视口重新加载、等待稳定、获取 snapshot、截图并读取控制台。比较断行、溢出、裁切、重叠和可读性。不要调用其他浏览器能力。

## 交互提示词

> 只使用 playwright-mcp。先 snapshot 获取稳定引用，再按 `<STEPS>` 逐项 click/type/press。每步记录可见变化、截图与新增错误；刷新页面验证需要持久化的状态。无法由 playwright-mcp 完成的步骤标为 blocked，不做本地自动化兜底。

## 结果判定

- `pass`：目标页全部截图并检查，控制台无未解释错误。
- `fail`：存在可复现的视觉、交互或运行时问题。
- `blocked`：`create_browser_preview` 缺失/托管失败，或 playwright-mcp 缺失、不可达、缺少所需能力。
