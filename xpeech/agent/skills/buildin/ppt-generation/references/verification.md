# HTML 演示文稿验证

浏览器验证必须遵循 `references/playwright-mcp.md`。只接受 playwright-mcp，不做本地浏览器控制。

## 必检项

### 入口与资源

- `index.html` 可访问，标题与语言正确。
- manifest 文件存在、页数和顺序正确。
- CSS、JavaScript、字体、Logo、图片没有 404。
- 页面无白屏，控制台无未解释 error。

### 逐页视觉

- 1920×1080 视口逐页截图。
- 标题、正文、关键数字没有溢出或裁切。
- 元素不重叠，安全区与页边距一致。
- 字体已加载，没有意外 fallback。
- 图片清晰、比例正确、无拉伸和破图。
- 重点对比度足够，远距离可读。
- 页面节奏有变化，但 design grammar 一致。

### 播放与状态

- ArrowLeft / ArrowRight、Home / End 正常。
- 页码与计数器同步。
- hash 可跳到指定页。
- 刷新后 localStorage 位置恢复符合预期。
- scale 与 letterbox 在缩小视口下正常。
- 最后一页继续前进不会出现空白页。

### 多文件架构

- 每个 `slides/*.html` 可被 playwright-mcp 独立导航。
- 单页控制台干净。
- iframe 加载失败时能定位到具体文件。
- 页面私有 CSS 不依赖其他页的加载顺序。

### PDF/PPTX 交接（仅用户要求时）

- 已读取 `references/export-compatible-html.md`。
- 需要 PDF 时，页面的关键内容不依赖 hover、中间动画帧或视频。
- 需要可编辑 PPTX 时，使用多文件架构和 `960pt×540pt` / `1280px×720px` 画布。
- PPTX 源页不存在裸文字 DIV、CSS gradient、文字标签背景/边框/阴影、DIV `background-image`。
- 仅检查 HTML 是否适合交接，不在主服务运行 PDF/PPTX 转换。

## 验证顺序

1. 调用内置 `create_browser_preview` 托管 deck 目录，使用它返回的 URL；禁止自行启动 HTTP 服务。
2. 确认 playwright-mcp 工具来源。
3. 导航到预览 URL 下的 `index.html`，设置 1920×1080。
4. 等待字体与图片，读取 snapshot、控制台和首页截图。
5. 逐页操作并截图。
6. 随机独立打开至少 3 页；不足 3 页时全部打开。
7. 清理 TODO、placeholder、临时路径和控制台问题。
8. 用户需要 PDF/PPTX 时，完成上述交接静态检查。
9. 修复后重复受影响页面与完整翻页回归。

因为 `create_browser_preview` 会复制文件，第 9 步前必须重新调用它，并对新 URL 验证。该工具不可用时将浏览器验证标记为 `blocked`，不得启动其他静态服务兜底。

## 报告格式

```text
status: pass | fail | blocked
entry: .../index.html
slides_checked: N/N
screenshots:
  - ...
console_errors:
  - ...
visual_issues:
  - slide: 03
    severity: high | medium | low
    issue: ...
    evidence: ...
```

没有 playwright-mcp 时只能报告 `blocked`，不得写成“已通过”。
