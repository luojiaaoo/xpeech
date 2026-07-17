# Verification：远程浏览器验证流程

默认使用 `create_browser_preview` 托管工作区内的 HTML，再用 `agent-browser` 通过注入的远程 CDP 会话完成导航、截图、交互和错误检查。

不要启动 `http.server`，不要打开 `file://` URL，不要安装或启动本地 Chrome/Chromium/Playwright 浏览器。

## 必做流程

### 1. 创建预览 URL

对 deck 根目录或单个 HTML 调用 `create_browser_preview`，记住返回的 HTTP(S) URL。

`create_browser_preview` 会把当前内容复制到隔离的预览目录。**HTML 或资源修改后必须重新调用**，旧 URL 不会自动反映新内容。

### 2. 加载 agent-browser 当前说明

本次任务第一次操作浏览器前执行：

```bash
agent-browser skills get core
```

不要手动填写 `--cdp` 或 `--session`，Shell 执行层会自动注入。

### 3. 打开、等待并检查结构

```bash
agent-browser open <preview-url>
agent-browser set viewport 1920 1080
agent-browser wait --load networkidle
agent-browser snapshot -c
```

先看 snapshot 是否有主标题、正确页数和导航元素，避免只靠截图猜测页面是否成功渲染。

### 4. 截图并检查错误

```bash
agent-browser screenshot screenshots/deck-cover.png
agent-browser console
agent-browser errors
```

- 幻灯片默认截 viewport，不要用 full-page 把多页拼成一张长图。
- 只有需要检查长页溢出时才使用 `agent-browser screenshot --full <path>`。
- 只检查某个区域时可用 `agent-browser screenshot "#hero" screenshots/hero.png`。
- 需要 retina 输出时用 `agent-browser set viewport 1920 1080 2`。
- 截图是工作区产物，需要交付给用户时使用 `send_file`。

## 多视口检查

每个视口都要重新设置尺寸、等待布局稳定后截图：

```bash
agent-browser set viewport 1920 1080
agent-browser wait 500
agent-browser screenshot screenshots/deck-1920x1080.png

agent-browser set viewport 1440 900
agent-browser wait 500
agent-browser screenshot screenshots/deck-1440x900.png
```

幻灯片是固定画布，核心检查是 16:9 视口下的缩放、letterbox、裁切和字体加载。

## 逐页和交互检查

对聚合 deck：

```bash
agent-browser open <preview-index-url>
agent-browser set viewport 1920 1080
agent-browser wait --load networkidle
agent-browser screenshot screenshots/slide-01.png
agent-browser press ArrowRight
agent-browser wait 300
agent-browser screenshot screenshots/slide-02.png
```

按同样方式遍历每页，并额外检查：

- `Home` / `End` / 方向键导航是否正确。
- 计数器、hash 和 localStorage 位置恢复是否一致。
- 按钮、Tweaks、动画和 iframe 内容是否可交互。
- 每次交互后重新执行 `snapshot -c`，必要时再截图。

## 批量脚本（仅需要自动化多视口/多页时）

`scripts/verify.py` 也不启动本地浏览器。它要求：

1. target 是 `create_browser_preview` 返回的 HTTP(S) URL。
2. 通过进程环境变量 `CDP_URL` 传入远程 CDP WebSocket URL。
3. 只安装 Python Playwright 客户端，不运行 `playwright install`。

```bash
pip install playwright
python scripts/verify.py <preview-url> \
  --viewports 1920x1080,1440x900 \
  --slides 10 \
  --output ./screenshots/
```

在 Compose 中 `.env` 默认配置 `CDP_URL=ws://browserless:3000`。如果从宿主机运行脚本，需将环境变量设为 `CDP_URL=ws://localhost:3000`。

## 验证出错时

### 页面白屏

1. 先运行 `agent-browser errors` 和 `agent-browser console`。
2. 检查 React+Babel script tag 的 integrity hash（见 `react-setup.md`）。
3. 检查 `const styles = {...}` 命名冲突。
4. 检查跨文件组件是否 export 到 `window`。
5. 检查 JSX 语法错误。

### 字体不对

- 检查 `@font-face` URL 是否能从 Browserless 容器访问。
- 检查 fallback 字体。
- 截图前执行 `agent-browser wait --load networkidle`，必要时再 `agent-browser wait 3500`。

### 动画卡顿

- 先用 `agent-browser` 实际操作相关按钮或键盘路径，不要只看静态首帧。
- 检查是否有频繁 reflow/layout thrashing。
- 动效优先使用 `transform` 和 `opacity`。

### 布局错位

- 检查 `box-sizing: border-box` 是否全局应用。
- 检查 reset 是否生效。
- 用 snapshot 确认元素存在，再结合截图检查几何布局。

## 交付前最终检查

- 首页和每一页都能渲染，无白屏。
- `agent-browser errors` 无未处理 JavaScript 错误。
- `agent-browser console` 无会影响演示的 error/warning。
- 字体、图片、iframe 和 CDN 资源均已加载。
- 随机抽查页和关键交互已截图人工确认。
- HTML 修改后已重新调用 `create_browser_preview` 并用新 URL 复验。

验证是设计工作的最后一道必做步骤：结构化 snapshot、视觉截图、console/errors 和真实交互四项都要覆盖，不能把“能打开”当成“已验证”。

# 播放器首屏布局（每次必测）

聚合页不能只检查“页面能打开”。至少在一个宽屏视口和一个窄屏视口中检查 stage 的 `getBoundingClientRect()`：

- `left >= -1`、`top >= -1`
- `right <= innerWidth + 1`、`bottom <= innerHeight + 1`
- `abs((left + right) / 2 - innerWidth / 2) <= 1`
- `abs((top + bottom) / 2 - innerHeight / 2) <= 1`

然后让预览容器改变一次尺寸并重复断言。若首屏只出现半张、stage 左上角落在视口中心附近，说明缩放初始化没有执行；不得把它当成内容页问题继续调 CSS。
