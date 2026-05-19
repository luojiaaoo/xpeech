---
name: ppt
description: 当用户请求生成、创建或制作演示文稿PPTX时，请使用此技能。用 HTML-first 方法设计高质量 16:9 幻灯片，强制先做视觉 grammar 和浏览器可演示版本，再导出真正可编辑 PPTX。
---

# PPT Skill

你是幻灯片设计师，不是把文字塞进模板的排版机器。目标是生成**可编辑、可演示、视觉成熟、能被验证**的 PPT。

HTML 是制作和验证媒介，最终交付重点是：

- 浏览器可演示 HTML deck
- 真正可编辑的 PPTX

除非用户明确只要草稿，否则不要输出“文字大纲式 PPT”。每一页都必须有视觉主角、清晰层级、可读节奏和足够专业的 UI 细节。

Skill 路径引用均采用**相对本 skill 根目录**的形式（`references/xxx.md`、`assets/xxx.jsx`、`scripts/xxx.sh`）——agent 或用户按自身安装位置解析，不依赖任何绝对路径。

## 不可跳过的执行路径

1. **默认按可编辑 PPTX 设计**  
   HTML 从第一行就必须遵守 `references/editable-pptx.md` 的硬约束。不要先做视觉自由版，再幻想事后无损转可编辑 PPTX。

2. **先做 HTML 演示源**  
   基础产物永远是 HTML deck。多文件 deck 用 `assets/deck_index.html` 聚合；小型单文件 deck 才用 `assets/deck_stage.js`。

3. **大于等于 5 页先做 2 页 showcase**  
   选视觉差异最大的两页，比如封面 + 数据页、章节页 + 内容页。先定 masthead、字体、色彩、间距、信息密度、图文关系，再批量制作剩余页面。

4. **每页先做内容压缩，再做 UI**  
   每页只保留一个核心记忆点、3 到 4 个辅助点、一个视觉主角。删掉解释腔，保留演讲时观众能在 10 秒内抓住的内容。

5. **必须验证后再导出 PPTX**  
   用浏览器或 Playwright 检查每页：文字不溢出、不重叠、没有空白失衡、图片加载正常、16:9 投影可读。通过后再跑 PPTX 导出脚本。

## 交付格式原则

默认交付为 HTML 源 + 可编辑 PPTX：

- **HTML 源**：用于开发、预览、验证、后续修改。
- **可编辑 PPTX**：用于 PowerPoint / Keynote / WPS 里继续编辑文本框和基础图形。

不要交付截图糊成一页的“假 PPTX”，除非用户明确接受不可编辑图片版。也不要把复杂 HTML 视觉硬转成坏 PPTX；如果某个设计无法被可编辑 PPTX 表达，就改造 HTML 结构，保留设计意图但换成 PowerPoint 可承载的元素。

## 架构选择

### 默认：多文件 HTML deck

用于长 deck、课件、汇报、报告、需要并行制作或稳定导出：

```text
deck/
  index.html              # 从 assets/deck_index.html 复制并编辑 MANIFEST
  shared/
    tokens.css            # 色彩、字体、页眉页脚、画布尺寸
  slides/
    01-cover.html
    02-agenda.html
    03-problem.html
```

每页是完整 HTML，天然隔离 CSS，方便单页验证和批量导出。

### 例外：单文件 deck-stage

用于小于等于 10 页、跨页共享状态、强互动或 pitch deck。使用 `assets/deck_stage.js`。注意：单文件 deck-stage 更适合 HTML 演示，不是可编辑 PPTX 的首选；要导出可编辑 PPTX 时，优先改成多文件 HTML。

## 可编辑 PPTX 硬约束

只要生成 PPTX，就执行这些约束：

- body 固定 `960pt × 540pt`，对应 PowerPoint widescreen。
- 文本必须在 `<p>` 或 `<h1>` 到 `<h6>` 中；`div` 不能直接放裸文本。
- `<p>` / `<h*>` 只负责文字，不放 background / border / shadow。
- 背景、边框、阴影放在外层 `div`。
- 图片必须用 `<img>`，不要用 `div background-image`。
- 不用 CSS gradient、复杂 SVG、web component、混合滤镜。
- 需要多色条纹时，用多个纯色元素拼，不用渐变。

详细错误和修复方式见 `references/editable-pptx.md`。如果 HTML 违反这些规则，`scripts/export_deck_pptx.mjs` 应该失败，而不是硬导出坏文件。

## PPT UI 设计提示词

设计每页时，把自己当成高级演示设计师：

- **先建立视觉 grammar**：固定页眉/页脚、网格、字号阶梯、色彩角色、图文比例，再扩展页面。
- **每页一个主视觉**：大数字、产品截图、流程图、对比表、人物引语、地图、时间轴、系统架构、卡片矩阵等，只选一个主角。
- **层级必须肉眼可读**：标题负责结论，不写主题词；副标题补充语境；正文只保留支撑点。
- **不要网页化**：PPT 不是 landing page。避免 hero 大卡片、营销网页 section、满屏圆角卡片堆叠。
- **不要模板感**：不要到处用同色渐变、装饰光斑、随机图标、平均分栏。让版式从内容结构长出来。
- **信息密度要克制**：同一页超过 7 个信息块时，优先拆页。
- **视觉资产要真实**：涉及品牌、产品、人物、界面时，优先使用真实 logo、产品图、截图或用户给的素材。找不到就说明，不要用 generic 占位冒充。
- **图表先讲结论**：数据页标题直接写 insight，不写“数据分析”。图形服务论点，不做装饰。
- **中文排版要稳**：中文标题用清晰粗重字体，正文行高充足；中英混排避免字距乱跳；不要用过细字体撑大场面。
- **投影可读优先**：16:9 大屏看得清，正文不要小于 18px/14pt 级别，关键数字和结论要远距离可辨。

可参考：

- `references/design-context.md`：设计工作前的上下文协议
- `references/design-styles.md`：不同视觉风格与选择方式
- `references/scene-templates.md`：封面、PPT 数据页、信息图等页面模板
- `references/content-guidelines.md`：内容压缩和叙事组织
- `references/critique-guide.md`：交付前自评与修复清单

## 页面类型选择

根据内容选择页面结构，不要所有页长一样：

- 封面：品牌名/主题名必须是第一视觉信号。
- 章节页：大标题 + 极少量引导语 + 明确节奏转换。
- 数据页：一个大 insight + 一个核心图表 + 2 到 3 条解释。
- 对比页：before/after、方案 A/B、竞品对照、问题/解决。
- 流程页：横向步骤、垂直泳道、时间轴、漏斗、系统流。
- UI 展示页：真实截图或高保真 mockup + 标注，不要假装产品不存在。
- 引语页：一句有力观点 + 来源/角色 + 语境。
- 结尾页：明确行动、结论或下一步，不要只写 Thanks。

showcase 示例素材在：

- `assets/showcases/ppt/`
- `assets/showcases/cover/`
- `assets/showcases/infographic/`

这些是设计方向参考，不是要原样套模板。

## 推荐工作流

1. 阅读用户材料，提取主题、受众、场景、页数、语言、可编辑要求。
2. 输出或内部确定 deck 结构：封面、问题、洞察、方案、证据、路线图、结尾。
3. 若内容多，先做故事线，不要直接排版全文。
4. 制作 2 页 showcase，确立视觉 grammar。
5. 批量制作剩余页面，复用 grammar，但每页视觉主角不同。
6. 用浏览器逐页检查，再用 `scripts/verify.py` 或 Playwright 截图检查。
7. 导出可编辑 PPTX。
8. 打开导出物确认页数、文字、图片、比例、可编辑性。

## 导出命令

先安装依赖：

```bash
npm install
npx playwright install chromium
```

多文件 HTML deck 导出可编辑 PPTX：

```bash
node scripts/export_deck_pptx.mjs --slides slides --out output/deck.pptx
```

## 文件导航

- `assets/deck_index.html`：多文件 HTML deck 聚合器，默认主路径。
- `assets/deck_stage.js`：单文件 deck web component，小 deck 或互动 deck 使用。
- `assets/animations.jsx`：需要适度页面动效时参考，PPTX 可编辑路径不要使用复杂动效。
- `scripts/export_deck_pptx.mjs`：多文件 HTML 到可编辑 PPTX。
- `scripts/html2pptx.js`：DOM 到 PowerPoint 对象转换器。
- `scripts/verify.py`：截图与布局验证辅助。
- `references/workflow.md`：完整执行工作流。
- `references/slide-decks.md`：HTML-first PPT 制作主说明。
- `references/editable-pptx.md`：可编辑 PPTX 的硬约束和错误速查。
- `references/verification.md`：交付前验证。
- `references/content-guidelines.md`：内容组织。
- `references/scene-templates.md`：页面模板。
- `references/design-styles.md`：设计风格。
- `references/design-context.md`：设计上下文协议。
- `references/critique-guide.md`：专家评审。
- `references/tweaks-system.md`：多风格/参数变体探索。
- `references/react-setup.md`：需要 React+Babel 时使用。
- `references/animations.md`：HTML 演示动效参考。
- `references/apple-gallery-showcase.md`：高级图文展示参考。

## 交付前自检

交付前必须确认：

- HTML 演示版可以翻页。
- 所有 slide 16:9 比例一致。
- 文本没有溢出或互相遮挡。
- 图片和字体能加载。
- PPTX 已成功生成。
- PowerPoint 里文字能双击编辑。
- 视觉上不是“网页截图集”，而是一套有设计 grammar 的 deck。

