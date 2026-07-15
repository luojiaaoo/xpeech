---
name: agent-browser
description: 面向 AI 代理的浏览器自动化 CLI。用于导航网站、点击元素、填写表单、提取页面数据、截图、测试 Web 应用以及其他浏览器自动化任务。浏览器由外部通过注入的 CDP 连接提供；不要安装、启动或管理本地浏览器。

---

# agent-browser

使用 `agent-browser` 通过 CDP 自动化控制外部管理的浏览器。支持无障碍树快照和紧凑的 `@eN` 元素引用。

Install: `npm i -g agent-browser`

## 加载当前说明

在任务中第一次执行 `agent-browser` 操作之前，加载与 CLI 匹配的工作流程：

```bash
agent-browser skills get core
```

仅在必要时加载完整参考：

```bash
agent-browser skills get core --full
```

请遵循返回的说明，因为它们与已安装的 CLI 版本相匹配。

## 截图

```bash
agent-browser open https://example.com && agent-browser snapshot
```

## 约束

- 不要运行 `agent-browser install`。
- 不要安装 Chrome、Chromium、Playwright、Puppeteer 或浏览器二进制文件。
- 不要启动或终止本地浏览器。
- 除非明确要求，否则不要创建另一个浏览器 session。
- 不要手动添加 `--session` 或 `--cdp`；它们会在执行时自动注入。
- 优先使用无障碍快照和 `@eN` 元素引用进行交互。
- 仅在视觉检查或证据有用时截图。
- 如果 CDP 连接失败，请报告错误，不要退回到本地浏览器。
