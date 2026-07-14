# 可转换 HTML 编写协议

主服务始终只生成并交付 HTML。用户同时需要 PDF/PPTX 时，在写 HTML 前先按本文档选择约束，再把 `assets/html-to-pptx-pdf-skill.zip` 发给用户在本地转换。

## 目录

1. 先确认用户需要的格式
2. PDF 友好 HTML
3. 可编辑 PPTX 友好 HTML
4. PDF + PPTX 同时需要
5. 交接检查

## 先确认用户需要的格式

| 用户需要 | 服务端写法 | 用户本地处理 |
|---|---|---|
| HTML | 默认 1920×1080，可使用完整 Web 能力 | 无 |
| HTML + PDF | 默认 1920×1080，加完整打印布局 | 用转换包导出 PDF |
| HTML + 可编辑 PPTX | 从第一页就按 PPTX 约束写 | 用转换包导出 PPTX |
| HTML + PDF + PPTX | 按更严格的 PPTX 约束写 | 用同一转换包导出两种格式 |

不要先自由写完 HTML，再尝试事后补救可编辑 PPTX。复杂 SVG、CSS 渐变、Web Component 和裸文字 DIV 会导致大量重写。

## PDF 友好 HTML

PDF 是 Chromium 对 HTML 的打印结果，可保留绝大多数视觉能力。

- 使用 `assets/deck_index.html` 或 `assets/deck_stage.js` 内置的打印布局。
- 每页保持固定尺寸、`overflow: hidden` 和不透明背景。
- 不要把关键内容只放在动画中间帧、hover、折叠区或视频里。
- 字体和图片使用可访问的本地资源或稳定 URL，并设置合理 fallback。
- 多文件 deck 的 manifest 顺序必须与期望 PDF 页序一致。
- 单文件 `<deck-stage>` 的每页必须是直接子 `<section>`，不要在中间多包一层。

## 可编辑 PPTX 友好 HTML

可编辑 PPTX 需要把 DOM 翻译为 PowerPoint 原生文本框、形状和图片，所以优先使用每页独立 HTML 的多文件架构。

### 画布

- 每页 `body` 使用 `960pt × 540pt`，等价于 `1280px × 720px` 或 `13.333in × 7.5in`。
- 将 `deck_index.html` 中的 `DECK_WIDTH` / `DECK_HEIGHT` 同步改为 `1280` / `720`，保证 HTML 播放与单页画布一致。
- 不要使用 `<deck-stage>` 作为可编辑 PPTX 的源结构。

### 必须遵守的 DOM/CSS 约束

1. 将文字放在 `<p>`、`<h1>`–`<h6>`、`<ul>`、`<ol>` 或 `<li>` 中；不要让 `<div>` 直接包含裸文字。
2. 不要使用 CSS gradient；使用纯色或预渲染图片。
3. 不要在文字标签上放 background、border 或 shadow；将装饰放在外层 `<div>`。
4. 不要在 `<div>` 上使用 `background-image`；使用显式 `<img>`。
5. 不要用 `::before` / `::after` 插入必须出现在 PPTX 中的文字或图形。
6. 不要依赖 Web Component、复杂 SVG、filter 或关键 transform 实现主视觉。
7. inline 文字元素不要用 margin 实现布局，将间距放到外层容器。

如果这些约束与用户要求的动画或高保真视觉冲突，在开工前说明：PDF 可保留视觉，可编辑 PPTX 必须牺牲部分 Web 效果。

## PDF + PPTX 同时需要

- 以 PPTX 约束作为 HTML 编写上限。
- 用户导出 PDF 时传入 `--width 1280 --height 720`，与 PPTX 源页一致。
- 不要维护两份内容不同的 HTML 源稿；PDF/PPTX 都从同一组页面生成。

## 交接检查

- HTML 入口和所有 `slides/*.html` 都在交付目录内。
- manifest 文件名、顺序和页数正确。
- 所有字体、图片、CSS 和 JavaScript 路径在用户本地仍然可解析。
- 用户需要 PPTX 时，已逐页检查上述 DOM/CSS 约束。
- 用户需要 PDF 时，已检查打印状态不依赖 hover/交互。
- 已发送 `assets/html-to-pptx-pdf-skill.zip`，但主服务未声称已生成 PDF/PPTX。
