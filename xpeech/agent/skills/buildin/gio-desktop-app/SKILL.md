```markdown
---
name: gio-desktop-app
short_description: Go Gio 桌面软件开发
description: "Build cross-platform desktop GUI software in pure Go with Gio (gioui.org). Use when the user wants to develop a desktop application or program (Windows/Linux/macOS), a GUI tool with windows and buttons, a standalone exe, or to package and publish a desktop app. Keywords: 桌面软件、桌面应用、桌面程序、图形界面、GUI、窗口程序、Gio、gioui、Golang、Go、exe、打包发布、纯Go。Before writing any code, confirm the user's runtime environment (Windows x64 / Linux x64 / Linux arm64) in plain language, guiding non-technical users to identify their computer."
---

# 桌面软件开发说明书（Gio + 纯 Go）

用 Gio 框架（gioui.org）+ 纯 Go 开发带图形界面的桌面软件（Windows / Linux / macOS），并按目标平台打包交付。
业务代码 100% Go，**不需要 CGO**，不写 C 或其他语言。

## 工作流程

1. **确认运行环境**（写代码前必须完成，见第 1 步）
2. 确认功能需求与界面元素
3. 搭建 Gio 项目骨架
4. 开发并编译验证
5. 按目标平台打包，交付可执行文件 + 大白话使用说明

## 第 1 步：询问运行环境（必做）

**开发前必须知道软件最终运行在哪种电脑上。** 用户已经说清的（例如"给我办公室的 Windows 电脑用"）直接记录进入第 2 步，不要重复追问。

用户很可能不懂"x64""架构"这类词。**用生活语言提问**："软件要装在什么电脑上？"，而不是"目标 GOARCH 是什么"。按 ask-user-question 技能的表单规范调用 `ask_user_question`：

```json
{
  "type": "form",
  "title": "确认软件运行环境",
  "subtitle": "请选择这款软件最终要装在哪种电脑上，我会按对应平台开发和打包",
  "submit_label": "开始开发",
  "fields": [
    {
      "type": "select",
      "name": "target_platform",
      "label": "软件要在什么电脑上运行？",
      "placeholder": "请选择",
      "options": [
        { "label": "Windows 电脑（最常见）", "value": "windows_x64" },
        { "label": "Linux 电脑或服务器", "value": "linux_x64" },
        { "label": "树莓派 / 开发板 / 信创电脑", "value": "linux_arm64" },
        { "label": "Mac 电脑", "value": "darwin_arm64" },
        { "label": "多种都要支持", "value": "multi_platform" },
        { "label": "不确定，帮我判断", "value": "unknown" }
      ]
    },
    {
      "type": "input",
      "name": "device_hint",
      "label": "若选"不确定"，简单描述一下这台电脑（可留空）",
      "placeholder": "例如：公司的 Windows 笔记本 / 家里的树莓派 4 / 一台黑色小主机"
    }
  ]
}
```

### 选项与构建目标对照

约 90% 的实际需求是前三种。把用户的大白话翻译成构建目标：

| 用户的说法 | 构建目标 | 说明 |
|---|---|---|
| Windows 电脑 / 笔记本 / 办公电脑 | `windows/amd64`（Windows x64） | 主流默认 |
| Linux 电脑 / 服务器 / Ubuntu / Debian | `linux/amd64`（Linux x64） | |
| 树莓派 3/4/5、开发板、工控机、信创电脑（鲲鹏 / 飞腾 / 麒麟 / 统信） | `linux/arm64`（Linux arm64） | 树莓派 1/2/Zero 是 armv6/armv7，按 `GOARCH=arm` 单独处理 |
| MacBook / iMac | `darwin/arm64` 或 `darwin/amd64` | |
| 骁龙笔记本 / 明确说 ARM 版 Windows | `windows/arm64` | 少见 |
| 多种都要支持 | 逐个平台打包 | 文件名带平台名 |

### 用户说"不确定"时的引导方法（大白话）

1. **看外观**：巴掌大的绿色电路板是树莓派（Linux arm64）；贴着"麒麟 / 统信 / 鲲鹏 / 飞腾"标牌的主机是信创电脑（多为 Linux arm64）；普通笔记本 / 台式机继续下一步。
2. **Windows 自查**：开始菜单 → 设置 → 系统 → 系统信息（或"关于"）→ 看"系统类型"：写 `x64-based processor` 就是 Windows x64，写 `ARM-based` 就是 Windows ARM。
3. **Linux 自查**：打开终端输入 `uname -m` 回车 —— `x86_64` 是 Linux x64，`aarch64` 是 Linux arm64。
4. 仍无法判断、线索指向普通家用 / 办公电脑时，**默认按 Windows x64**（占比最大），并建议顺带给 Linux x64 也打一个包（成本很低）。

确认后先向用户复述一句"好的，软件将运行在 XXX 上"，再开始开发；多平台时列出全部构建目标。

## 第 2 步：确认需求

用一两句话复述软件要做什么，列出主要窗口和控件（按钮、表格、输入框、托盘图标……）。关键信息缺失时用表单补问一次即可，不要反复追问；先做出能跑的最小版本再迭代。

## 第 3 步：项目骨架（纯 Go）

```bash
mkdir myapp && cd myapp
go mod init example.com/myapp
go get gioui.org@latest
go get gioui.org/x@latest
```

最小 `main.go`：

```go
package main

import (
	"log"
	"os"

	"gioui.org/app"
	"gioui.org/font/gofont"
	"gioui.org/io/system"
	"gioui.org/layout"
	"gioui.org/op"
	"gioui.org/widget/material"
)

func main() {
	go func() {
		w := app.NewWindow(app.Title("我的桌面软件"), app.Size(600, 400))
		if err := run(w); err != nil {
			log.Fatal(err)
		}
		os.Exit(0)
	}()
	app.Main()
}

func run(w *app.Window) error {
	th := material.NewTheme()
	th.Shaper = gofont.Collection()

	var ops op.Ops
	var btn widget.Clickable

	for {
		switch e := w.Event().(type) {
		case system.DestroyEvent:
			return e.Err
		case system.FrameEvent:
			gtx := layout.NewContext(&ops, e)
			if btn.Clicked(gtx) {
				os.Exit(0)
			}
			layout.Center.Layout(gtx, func(gtx layout.Context) layout.Dimensions {
				return layout.Flex{Axis: layout.Vertical}.Layout(gtx,
					layout.Rigid(material.H1(th, "你好，桌面软件已运行").Layout),
					layout.Rigid(material.Button(th, &btn, "退出").Layout),
				)
			})
			e.Frame(gtx.Ops)
		}
	}
}
```

> 注意：`run` 函数里需要 `"gioui.org/widget"` 包来使用 `widget.Clickable`。

- 目录保持简单：`main.go` + 按功能分包（`ui/`、`core/`），不要过度设计。
- Gio 是**即时模式**（每帧重绘），状态用局部变量 / 结构体保存，UI 函数每帧执行一次。
- 界面含中文时捆绑中文字体（`go:embed` 引入 .ttf），追加到 `go font.FontFace` 列表后设 `th.Shaper`，保证精简系统上也能正常显示中文。
- **Gio 不需要 CGO**，纯 Go 编译，无 C 编译器依赖。这是相比 Fyne 的核心优势。

### 核心编写要点

- 事件循环：`for { switch e := w.Event().(type) { case system.DestroyEvent: ...; case system.FrameEvent: ... } }`
- 布局：`layout.Flex`（水平/垂直排列）、`layout.Stack`（层叠）、`layout.Center`（居中）、`layout.Inset`（边距）
- 组件：`material.Button`、`material.Label`、`material.Editor`（输入框）、`material.List`（滚动列表）、`material.CheckBox`、`material.Switch`、`material.Slider`
- 点击：`var clickable widget.Clickable`，在 FrameEvent 中检查 `clickable.Clicked(gtx)`
- 异步数据更新：goroutine 中完成后调用 `w.Invalidate()` 触发重绘
- 填充背景：`material.Background(th, color)` 或 `paint.FillShape()`
- 图片/资源用 `go:embed` 嵌入

## 第 4 步：编译验证

```bash
go vet ./... && go build ./...
```

沙盒 / SSH 等无图形界面环境里编译通过即算验证成功（Gio 无 CGO，编译简单），可执行文件交给用户在自己电脑上运行确认。

## 第 5 步：按目标平台打包

**Gio 无需 CGO，无需安装 gcc / mesa / xorg 等系统包，直接交叉编译即可。**

构建命令：

```bash
# Linux x64
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 \
  go build -ldflags="-s -w" -o dist/myapp-linux-x64

# Windows x64（-H windowsgui 去掉黑色控制台窗口）
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 \
  go build -ldflags="-s -w" -H windowsgui -o dist/myapp-windows-x64.exe

# Linux arm64（树莓派 / 开发板 / 信创电脑）
GOOS=linux GOARCH=arm64 CGO_ENABLED=0 \
  go build -ldflags="-s -w" -o dist/myapp-linux-arm64

# macOS (Apple Silicon)
GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 \
  go build -ldflags="-s -w" -o dist/myapp-mac-arm64
```

- 全部命令都设 `CGO_ENABLED=0`，任意平台交叉编译均无 C 依赖。
- Windows 下用 `-H windowsgui` 去掉控制台窗口。
- macOS 若需要 .app 图标/Info.plist，可用 `fyne package` 风格手工组装目录，或交付裸二进制。
- Linux 运行需要图形环境（X11 / Wayland），桌面发行版自带运行库，无需额外安装。

## 交付物

每个目标平台交付：

1. 可执行文件：Windows 为 `.exe`（双击即用）；Linux 为二进制（首次运行前 `chmod +x`）；macOS 为裸二进制或 .app。
2. 一页大白话使用说明：怎么启动、数据存哪儿、出问题找谁，不用技术术语。
3. 多平台时逐个说清哪个文件对应哪种电脑，文件名带平台名（如 `myapp-windows-x64.exe`）。

## 常见问题

| 现象 | 原因与解决 |
|---|---|
| Windows 双击 .exe 弹出黑色控制台 | 缺 `-H windowsgui` |
| Linux 上报 X11 / Wayland / GL 连接错误 | 服务器 / 无桌面环境跑不了 GUI；桌面发行版一般自带运行库 |
| 中文显示为方块或乱码 | 捆绑中文字体并设置为 `th.Shaper`，不要只依赖 `gofont` |
| `go get gioui.org` 报错 | 检查 Go 版本 ≥ 1.21，网络不通时配 GOPROXY |
| 编译通过但窗口不显示 | 无图形环境（SSH / 容器）属正常；交付给用户在桌面环境运行 |
| 界面卡顿 | 减少每帧分配；用 `op.Record` 缓存静态内容；避免在 Layout 中做 I/O |
| 点击按钮没反应 | 确认在 FrameEvent 中调用了 `clickable.Clicked(gtx)` 且 widget 在约束范围内 |
| 用户需求含糊 | 先交付最小可用窗口，再按反馈迭代 |
```