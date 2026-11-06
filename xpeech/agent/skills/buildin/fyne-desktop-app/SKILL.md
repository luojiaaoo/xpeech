---
name: fyne-desktop-app
short_description: Go 桌面软件开发
description: "Build cross-platform desktop GUI software in pure Go with Fyne. Use when the user wants to develop a desktop application or program (Windows/Linux), a GUI tool with windows and buttons, a standalone exe, or to package and publish a desktop app. Keywords: 桌面软件、桌面应用、桌面程序、图形界面、GUI、窗口程序、Fyne、Golang、exe、打包发布。Before writing any code, confirm the user's runtime environment (Windows x64 / Linux x64 / Linux arm64) in plain language, guiding non-technical users to identify their computer."
---

# 桌面软件开发说明书（Fyne + 纯 Go）

用 Fyne 框架 + 纯 Go 开发带图形界面的桌面软件（Windows / Linux），并按目标平台打包交付。
业务代码 100% Go，不写 C 或其他语言。

## 工作流程

1. **确认运行环境**（写代码前必须完成，见第 1 步）
2. 确认功能需求与界面元素
3. 搭建 Fyne 项目骨架
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
        { "label": "多种都要支持", "value": "multi_platform" },
        { "label": "不确定，帮我判断", "value": "unknown" }
      ]
    },
    {
      "type": "input",
      "name": "device_hint",
      "label": "若选“不确定”，简单描述一下这台电脑（可留空）",
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
| 骁龙笔记本 / 明确说 ARM 版 Windows | `windows/arm64` | 少见 |
| MacBook / iMac | `darwin/*` | 少见，Fyne 同样支持 |
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
go get fyne.io/fyne/v2@latest
```

最小 `main.go`：

```go
package main

import (
	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/widget"
)

func main() {
	a := app.New()
	w := a.NewWindow("我的桌面软件")
	w.SetContent(container.NewVBox(
		widget.NewLabel("你好，桌面软件已运行"),
		widget.NewButton("退出", func() { a.Quit() }),
	))
	w.Resize(fyne.NewSize(400, 300))
	w.ShowAndRun()
}
```

- 目录保持简单：`main.go` + 按功能分包（`ui/`、`core/`），不要过度设计。
- 图标等资源用 `fyne bundle icon.png > bundled.go` 生成纯 Go 资源文件。
- 界面含中文时建议捆绑中文字体（`fyne bundle font.ttf > bundled.go`）并设为应用主题字体，保证精简系统上也能正常显示中文。
- Fyne 构建需要 CGO 和 C 编译器（环境已提供），但不写任何 C 代码。`go` 命令找不到时检查 `/usr/local/go/bin` 是否在 PATH 中。

## 第 4 步：编译验证

```bash
go vet ./... && go build ./...
```

沙盒 / SSH 等无图形界面环境里 `go run .` 弹不出窗口属正常现象：用编译通过做开发期验证，可执行文件交给用户在自己电脑上运行确认。

## 第 5 步：按目标平台打包

构建期依赖（装一次即可）：

- Linux 目标：`gcc libgl1-mesa-dev xorg-dev`（Debian/Ubuntu 包名）
- Windows 交叉编译：`gcc-mingw-w64-x86-64`
- Linux arm64 交叉编译：`gcc-aarch64-linux-gnu`，另需目标架构的 GL/X 开发库（`dpkg --add-architecture arm64` 后装 `libgl1-mesa-dev:arm64 xorg-dev:arm64`）

构建命令：

```bash
# Linux x64
GOOS=linux GOARCH=amd64 CGO_ENABLED=1 \
  go build -ldflags="-s -w" -o dist/myapp-linux-x64

# Windows x64（从 Linux 交叉编译；-H windowsgui 去掉黑色控制台窗口）
GOOS=windows GOARCH=amd64 CGO_ENABLED=1 CC=x86_64-w64-mingw32-gcc \
  go build -ldflags="-s -w" -H windowsgui -o dist/myapp-windows-x64.exe

# Linux arm64（树莓派 / 开发板 / 信创电脑）
GOOS=linux GOARCH=arm64 CGO_ENABLED=1 CC=aarch64-linux-gnu-gcc \
  go build -ldflags="-s -w" -o dist/myapp-linux-arm64
```

- 在 Windows 本机开发时装好 MSYS2 / TDM-GCC 直接 `go build` 即可。
- 在树莓派等目标机上开发时直接 `go build`，无需交叉编译。
- 交叉编译 arm64 链接失败、或需要带图标 / 版本信息的发布物时，用官方工具：
  - `go install fyne.io/fyne/v2/cmd/fyne@latest`，再 `fyne package -os windows -icon icon.png`（产出带图标的 .exe）或 `fyne package -os linux -icon icon.png`（产出 .tar.gz）；
  - 或 `go install github.com/fyne-io/fyne-cross@latest`，用 Docker 统一处理各平台工具链（适合在自己的开发机上使用）。

## 交付物

每个目标平台交付：

1. 可执行文件：Windows 为 `.exe`（双击即用）；Linux 为二进制或 .tar.gz（首次运行前 `chmod +x`）。
2. 一页大白话使用说明：怎么启动、数据存哪儿、出问题找谁，不用技术术语。
3. 多平台时逐个说清哪个文件对应哪种电脑，文件名带平台名（如 `myapp-windows-x64.exe`）。

## 常见问题

| 现象 | 原因与解决 |
|---|---|
| Windows 双击 .exe 弹出黑色控制台 | 缺 `-H windowsgui`，或改用 `fyne package -os windows` |
| Linux 上报 X11 / OpenGL 编译错误 | 缺构建期开发包：装 `gcc libgl1-mesa-dev xorg-dev` |
| arm64 交叉编译链接失败 | 缺目标架构 GL/X 开发库；改用 fyne-cross 或在目标机上编译 |
| 运行报找不到 libGL / 打不开显示 | 服务器 / 无桌面环境跑不了 GUI；桌面发行版一般自带运行库 |
| 中文显示为方块 | 捆绑中文字体并设置为应用主题字体 |
| 用户需求含糊 | 先交付最小可用窗口，再按反馈迭代 |
