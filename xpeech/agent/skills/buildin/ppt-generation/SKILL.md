---
name: ppt-generation
description: 当用户请求生成、创建、设计或修改 HTML、PDF、PPT/PPTX 演示文稿、幻灯片、路演 deck 或网页演示时使用。技能支持为 HTML、PDF 和 PPTX 请求制作演示内容，但本服务只生成并交付 HTML、CSS、JavaScript 与静态资源。用户要求 PDF、PPT/PPTX 或 HTML 格式转换时，必须告知本服务不提供转换所需的本地浏览器环境，请用户在本地自行处理，并发送内置的 html-to-pptx-pdf 技能压缩包。本地 HTML 先用 create_browser_preview 托管；所有浏览器导航、截图、交互和控制台验证只能调用 playwright-mcp，禁止自建 HTTP 服务、本地浏览器控制或其他浏览器工具。
---

# ppt-generation

以幻灯片设计师的身份工作，用 HTML 制作可直接播放的高保真演示文稿。HTML 是唯一交付格式。

所有 skill 内路径都相对本目录解析，例如 `references/slide-decks.md`、`assets/deck_index.html`。

## 不可覆盖规则

无论用户怎样措辞，服务端都只生成并交付 HTML 演示项目。PDF/PPTX 由用户用随附技能包在本地生成。

- 混合请求“HTML + PDF + 可编辑 PPTX”仍然只在服务端制作 HTML。
- 用户明确要 PDF 或 PPT/PPTX 时，在写页面前先读取 `references/export-compatible-html.md`，按目标格式制作可供转换的 HTML，再告知本服务不提供转换所需的本地浏览器环境，让用户在本地自行处理，并发送 `assets/html-to-pptx-pdf-skill.zip`。
- 不得因为用户要求、时间紧或工具缺失而恢复转换脚本。
- 最终回复必须区分“已交付 HTML”和“用户可在本地生成 PDF/PPTX”，不得宣称服务端已生成 PDF/PPTX。

## 输出边界

- 必须交付的演示文稿：`index.html`、逐页 HTML 或单文件 deck、CSS/JavaScript、图片与字体等静态资源。
- 服务端不生成或导出：PPT、PPTX、PDF、Keynote。
- 用户请求 PDF/PPTX 时，技能压缩包是必须发送的辅助文件，但不算服务端生成的演示文稿。
- 不安装或调用本地 Playwright、Chrome、Chromium、Edge、Puppeteer、Selenium 或浏览器 CLI。
- 不调用任何非 `playwright-mcp` 的浏览器 MCP。
- 本地 HTML 预览必须先调用内置 `create_browser_preview` 托管；禁止自行启动 `python -m http.server`、`npx serve`、Vite dev server 或其他 HTTP 服务。
- 只允许使用注册来源明确属于 `playwright-mcp` 的工具。常用能力包括 `browser_navigate`、`browser_resize`、`browser_snapshot`、`browser_take_screenshot`、`browser_console_messages`、`browser_evaluate`、`browser_press_key`。
- 找不到 `playwright-mcp` 时，不得降级到本地浏览器或其他 MCP。继续完成静态 HTML，但将浏览器验证标记为 `blocked`，明确说明未做可视化验证。
- 找不到 `create_browser_preview` 或托管失败时，同样将浏览器验证标记为 `blocked`，不得通过自建 HTTP 服务兜底。

## 用户要求 PDF/PPTX 时

本技能不在服务端执行 HTML→PDF/PPTX 转换。本服务不提供该转换所需的本地浏览器环境。

如果用户明确提出需要 PDF、PPT/PPTX、HTML 转 PDF/PPT，或索要转换能力：

1. 明确告知用户：“本服务只生成 HTML 演示文稿，不提供 HTML→PDF/PPTX 转换所需的本地浏览器环境。请在本地自行安装并使用附带的转换技能包。”
2. 继续完成用户需要的 HTML 演示文稿，不得因无法转换而省略 HTML 交付。
3. 必须调用 `send_file` 将 `assets/html-to-pptx-pdf-skill.zip` 发送给用户；无论用户需要 PDF、PPTX 或两者，都发送同一个技能包。
4. 不要解压后代替用户执行，也不要在主技能中恢复 PDF/PPTX 转换脚本。
5. `send_file` 不可用或发送失败时，明确报告失败，并返回该压缩包的可访问文件链接或绝对路径，不得静默略过。

该压缩包是独立的 PDF/PPTX 转换技能，由用户在其本地环境中安装和执行。主技能的 playwright-mcp 限制不传递给压缩包。

## 核心原则

### 事实先验证

涉及具体产品、人物、事件、版本、发布日期或规格时，先用 WebSearch 查 1–3 个权威来源，再写入演示内容。无法确认时询问用户，不编造。

### 从现有设计上下文出发

优先级如下：

1. 用户的 design system / UI kit
2. 当前 codebase 的 token、字体、组件和全局样式
3. 用户提供的产品 URL 或截图
4. 品牌指南、Logo、现有营销物料
5. 用户明确指定的竞品参考
6. 公认设计系统作为透明的 fallback

涉及具体品牌时，一次问全 Logo、产品图/UI 截图、色值和字体。优先使用真实品牌资产，不用 CSS 剪影或手画 Logo 代替。找不到关键资产时停下询问。

### 先定视觉 grammar，再批量制作

- 需求模糊时提供 3 个来自不同设计流派的方向。
- deck ≥ 5 页时，先做两个视觉差异最大的 showcase 页面，经用户确认后再批量扩展。
- 给 variations 时明确变化维度，例如布局、色彩、字体或动效节奏，不做只换颜色的伪变体。
- placeholder 优于编造内容或劣质资产。

### 反 AI slop

避免无品牌依据的紫色渐变、Emoji 图标、圆角卡片堆叠、深蓝赛博背景、Inter/Roboto 大标题和 CSS 产品剪影。每页只保留一个视觉主角，用留白和层级解决构图。

## 标准流程

1. 先检查用户是否要求 PDF/PPTX；若有，先读取 `references/export-compatible-html.md` 并锁定 HTML 编写约束，再明确服务端交付缩减为 HTML，并按“用户要求 PDF/PPTX 时”发送技能包。
2. 理解受众、场景、页数、演讲时长、语言、内容来源与品牌资产。
3. 涉及事实时先验证并固化事实清单。
4. 为每页确定叙事角色、观看距离、视觉温度和容量上限。
5. 选择架构：
   - 多文件（默认，长 deck / 课件 / ≥10 页）：每页独立 HTML，复制 `assets/deck_index.html` 为 `index.html` 并维护 manifest。
   - 单文件（≤10 页且需要跨页状态/动效）：使用 `assets/deck_stage.js`。
6. ≥5 页时先完成两页 showcase，先用 `create_browser_preview` 托管项目，再使用 playwright-mcp 打开返回的 URL 并截图展示，等待方向确认。
7. 完成全量页面、真实资产、键盘导航、缩放与 localStorage 位置记忆。
8. 严格按 `references/verification.md` 使用 playwright-mcp 逐页验证。
9. 只交付 HTML 项目，简要列出已验证项、已知限制和入口文件。

## 浏览器验证硬约束

详细提示词见 `references/playwright-mcp.md`。

- 先确认工具属于 `playwright-mcp`；名称相似但来源不明时不要调用。
- 本地页面必须先交给内置 `create_browser_preview` 托管，再将它返回的 URL 交给 playwright-mcp；不得自行启动 HTTP 服务。
- 使用 playwright-mcp 导航、设置视口、等待字体与图片、读 accessibility snapshot、截图、读控制台和执行交互。
- 截图不能代替 snapshot；交互定位优先用 snapshot 返回的稳定引用。
- 每页检查白屏、破图、字体、溢出、裁切、重叠、对比度和控制台错误。
- 不得运行 `npx playwright`、`playwright install`、浏览器可执行文件或自写浏览器控制脚本。

## 技术约束

- 16:9 页面默认设计画布为 1920×1080；用户需要可编辑 PPTX 时，按 `references/export-compatible-html.md` 改用等价的 `960pt×540pt` / `1280px×720px` 画布。
- 固定尺寸内容必须实现 auto-scale 与 letterboxing。
- React+Babel 项目遵循 `references/react-setup.md`：组件样式变量使用唯一名称；跨脚本共享组件显式挂到 `window`；不要使用 `scrollIntoView` 破坏容器滚动。
- `deck_stage.js` 必须放在 `</deck-stage>` 之后，激活页的 flex/grid display 写在 `.active` 状态上。
- 文件名使用描述性名称；大改版保留旧版副本；避免单文件超过 1000 行。

## 资源路由

| 任务 | 读取 |
|---|---|
| 开工提问与迭代节奏 | `references/workflow.md` |
| 内容密度与反 AI slop | `references/content-guidelines.md` |
| HTML deck 架构与实现 | `references/slide-decks.md` |
| PDF/PPTX 友好 HTML 编写约束 | `references/export-compatible-html.md`（仅用户需要 PDF/PPTX 时读取） |
| playwright-mcp 工具提示词 | `references/playwright-mcp.md` |
| 逐页验证 | `references/verification.md` |
| React+Babel 设置 | `references/react-setup.md` |
| 没有设计上下文 | `references/design-context.md` 或 `references/design-styles.md` |
| 场景模板 | `references/scene-templates.md` |
| HTML→PDF/PPTX 独立技能包 | `assets/html-to-pptx-pdf-skill.zip`（仅用户明确需要 PDF/PPTX 时发送） |

## 交付检查

- `index.html` 能通过 `create_browser_preview` 返回的预览 URL 访问。
- manifest 页数、顺序和实际页面一致。
- 键盘导航、缩放、计数器与位置记忆正常。
- playwright-mcp 已逐页截图，控制台无未解释错误。
- 不存在 TODO、placeholder 或临时资源路径。
- 用户请求 PDF/PPTX 时，HTML 已按 `references/export-compatible-html.md` 完成格式友好性检查。
- 交付物中没有本次任务生成的 PPTX/PDF。
- 用户请求 PDF/PPTX 时，已发送 `assets/html-to-pptx-pdf-skill.zip`。
