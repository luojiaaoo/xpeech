# HTML 演示文稿规范

主服务只生成 HTML 演示文稿。不要在服务端生成或导出 PPTX/PDF；用户需要这些格式时，按 `SKILL.md` 发送本地转换技能包。

## 目录

1. 格式与编写约束
2. 批量制作前的 showcase
3. 架构选择
4. 多文件架构
5. 单文件架构
6. 标签与讲者备注
7. 视觉与内容约束
8. 播放能力
9. 常见故障
10. playwright-mcp 验证

## 格式与编写约束

HTML 始终是服务端唯一生成的演示文稿。但目标格式会影响 HTML 从第一页开始的写法：

- 只需 HTML：默认 1920×1080，可使用完整 Web 能力。
- 还需 PDF：保留视觉自由度，但必须保证打印状态静态、完整且逐页分隔。
- 还需可编辑 PPTX：必须优先多文件架构，并在写 HTML 前读取 `references/export-compatible-html.md`。

不在主服务运行转换脚本。用户需要 PDF/PPTX 时，发送 `assets/html-to-pptx-pdf-skill.zip` 让用户本地处理。

## 批量制作前的 showcase

deck ≥ 5 页时，先完成两个视觉结构差异最大的页面，例如“封面 + 数据页”或“封面 + 产品截图页”。

1. 用真实内容和资产完成两页，不做空壳。
2. 调用 `create_browser_preview` 托管所在目录。
3. 用 playwright-mcp 截图，让用户确认字体、色彩、masthead、间距和页面节奏。
4. 方向确认后再批量扩展，避免整套返工。

需要预制方向参考时，读取 `assets/showcases/INDEX.md`；每个索引条目都必须同时存在 `.html` 源文件和 `.png` 预览图。

## 架构选择

默认使用多文件架构：

| 场景 | 架构 |
|---|---|
| ≥10 页、学术/课件、长 deck、多人协作 | 每页独立 HTML + `deck_index.html` |
| ≤10 页、需要跨页状态或连续动效 | 单 HTML + `deck_stage.js` |

多文件架构优先，因为 iframe 隔离样式，每页可独立验证，批量修改时冲突最少。

## 多文件架构

```text
deck/
├── index.html
├── shared/
│   ├── tokens.css
│   └── assets/
└── slides/
    ├── 01-cover.html
    ├── 02-agenda.html
    └── 03-content.html
```

从 `assets/deck_index.html` 复制 `index.html`，只维护 manifest：

```js
window.DECK_MANIFEST = [
  { file: 'slides/01-cover.html', label: '封面' },
  { file: 'slides/02-agenda.html', label: '目录' },
  { file: 'slides/03-content.html', label: '核心内容' },
];
```

每张 slide 都是完整 HTML 文档。`body` 直接作为 1920×1080 画布，页面私有布局写在页内，共享色板、字体阶和页眉页脚规则放在 `shared/tokens.css`。

不要把单页布局类放进共享 CSS；共享层只放真正跨页一致的 token 和 chrome。

每页必须：

- 可以独立通过预览 URL 打开，不依赖其他页先加载。
- 自己引入所需字体和共享 CSS。
- 设置唯一、可识别的 `<title>`。
- 使用描述性文件名并以两位页码排序，例如 `05-product-demo.html`。
- 在 manifest 中与文件名、标签、页数一一对应。

## 单文件架构

使用 `assets/deck_stage.js`：

```html
<deck-stage>
  <section class="slide active">...</section>
  <section class="slide">...</section>
</deck-stage>
<script src="deck_stage.js"></script>
```

约束：

- `<script>` 放在 `</deck-stage>` 之后。
- section 的可见 display 写在 active 状态，例如 `.slide.active { display: grid; }`。
- 避免全局选择器覆盖未激活页的隐藏规则。
- hash 与 localStorage 恢复顺序必须确定，URL hash 优先于旧播放位置。

不要直接给普通 section 类写 `display: flex/grid`，否则会覆盖未激活页的隐藏规则。使用下列模式之一：

```css
deck-stage > section:not(.active) {
  display: none !important;
}

deck-stage > section.active {
  display: grid;
}
```

或把 flex/grid 布局放到 section 内部 wrapper 上，让 section 只负责显示与隐藏。

## 标签与讲者备注

### Slide 标签

- 多文件：在 manifest 中设置 `{ file, label }`。
- 单文件：在 section 上设置 `data-screen-label`。
- 面向用户的页码从 1 开始，不使用数组的 0-based 位置与用户沟通。

### Speaker notes

只在用户明确要求时添加。在 `index.html` 的 `<head>` 中使用：

```html
<script type="application/json" id="speaker-notes">
[
  "第 1 页的完整讲稿",
  "第 2 页的完整讲稿"
]
</script>
```

备注数组顺序必须与页序一致；写成可直接讲述的口语稿，不要只写提纲。

## 视觉与内容约束

- 每页只讲一个结论，标题使用断言句。
- 一个页面只设一个视觉主角；正文、注解和页码退居层级。
- 观众在 10 米投屏距离仍应读清标题和关键数据。
- 每页控制在 1 个核心信息、3–4 个辅助点和 1 个主视觉以内。
- 页面类型轮换：大字封面、数据页、对比页、时间轴、引语页、产品截图页、结尾 CTA。
- 使用真实 Logo、产品图和 UI 截图；缺失时保留清晰 placeholder。
- 不用 Emoji 充当图标，不用无依据的科技渐变和卡片墙。

先声明整套 deck 的设计系统：背景色数量、display/body 字体、普通页与章节页节奏、图像规则和页边距。同一套 deck 通常只用 4–5 种 layout，但要轮换颜色、密度和主视觉，避免每页像同一张模板。

字号建议：

- 正文不小于 24px，理想范围 28–36px。
- 普通标题 60–120px。
- 主视觉大字 180–240px。
- 数据页让关键数字成为主视觉，其周边解释不超过 3 行。

## 播放能力

HTML 交付至少支持：

- ArrowLeft / ArrowRight 翻页
- Home / End 跳转
- 计数器
- scale + letterbox
- localStorage 记忆播放位置
- hash 深链到指定页
- 全屏或浏览器演讲模式

这些都是 HTML 功能，不应触发任何格式导出。

## 常见故障

- iframe 白屏：直接导航到对应 `slides/*.html`，检查 manifest 相对路径和资源 404。
- 多页叠在一起：检查 section 的 `display` 是否绕过 `.active` 状态。
- 缩放错位：确认每页是固定画布，不用 `vw` / `vh` 做关键定位。
- hash 跳页失效：确认 hash 恢复优先于 localStorage。
- 字体回落：等待 `document.fonts.ready`，再检查字体 URL 和 fallback。
- 图片破损：优先使用 deck 目录内相对路径，确保 `create_browser_preview` 复制后仍然可解析。
- 修复后仍看到旧页面：重新调用 `create_browser_preview`，不要继续使用旧 URL。

## playwright-mcp 验证

先对 deck 目录调用内置 `create_browser_preview`，将 `index.html` 追加到它返回的目录 URL 前缀。禁止自行启动 `python -m http.server`、`npx serve`、Vite dev server 或其他 HTTP 服务。随后必须使用 `playwright-mcp`：

1. `browser_navigate` 打开 `create_browser_preview` 产生的 `index.html` 预览 URL。
2. `browser_resize` 设置 1920×1080。
3. `browser_snapshot` 获取结构和稳定元素引用。
4. 等待 `document.fonts.ready` 与图片加载完成。
5. `browser_take_screenshot` 保存每页截图。
6. `browser_console_messages` 检查 warning/error。
7. 用 `browser_press_key` 逐页前进，并确认页数、顺序和状态恢复。

工具必须来自 `playwright-mcp`。没有该 MCP 时只完成静态代码检查，并把视觉验证标为 blocked；禁止改用本地 Playwright、浏览器 CLI 或其他浏览器 MCP。
修改文件后必须重新调用 `create_browser_preview` 获取新 URL，避免验证旧副本。

完整提示词见 `references/playwright-mcp.md`，检查清单见 `references/verification.md`。
