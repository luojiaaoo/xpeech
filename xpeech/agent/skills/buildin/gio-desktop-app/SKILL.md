---
name: gio-desktop-app
short_description: Go 桌面软件开发
description: "Build cross-platform desktop GUI software in pure Go with Gio (gioui.org). Use when the user wants to develop a desktop application or program (Windows/Linux/macOS), a GUI tool with windows and buttons, a standalone exe, or to package and publish a desktop app. Keywords: 桌面软件、桌面应用、桌面程序、图形界面、GUI、窗口程序、Gio、gioui、Golang、Go、exe、打包发布、纯Go。Before writing any code, confirm the user's runtime environment (Windows x64 / Linux x64 / Linux arm64) in plain language, guiding non-technical users to identify their computer."
---

# 桌面软件开发说明书（Gio + 纯 Go）

用 Gio 框架（gioui.org）+ 纯 Go 开发带图形界面的桌面软件（Windows / Linux / macOS），并按目标平台打包交付。
业务代码 100% Go，不写 C 或其他语言。

**选 Gio 的理由：Windows 目标零 cgo、零 C 编译器**——装了 Go 就能编译出 .exe，交叉编译也是纯 Go 链路。
（注意与 Fyne 的区别：Fyne 构建必须有 gcc/mingw；Gio 在 Windows 上用纯 Go syscall 调 D3D/Win32。）

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
| Windows 电脑 / 笔记本 / 办公电脑 | `windows/amd64`（Windows x64） | 主流默认，**纯 Go 直接交叉编译** |
| Linux 电脑 / 服务器 / Ubuntu / Debian | `linux/amd64`（Linux x64） | 构建需 cgo，见第 5 步依赖 |
| 树莓派 3/4/5、开发板、工控机、信创电脑（鲲鹏 / 飞腾 / 麒麟 / 统信） | `linux/arm64`（Linux arm64） | 树莓派 1/2/Zero 是 armv6/armv7，按 `GOARCH=arm` 单独处理 |
| 骁龙笔记本 / 明确说 ARM 版 Windows | `windows/arm64` | 纯 Go 直接交叉编译 |
| MacBook / iMac | `darwin/*` | 少见，Gio 同样支持（纯 Go） |
| 多种都要支持 | 逐个平台打包 | 文件名带平台名 |

### 用户说"不确定"时的引导方法（大白话）

1. **看外观**：巴掌大的绿色电路板是树莓派（Linux arm64）；贴着"麒麟 / 统信 / 鲲鹏 / 飞腾"标牌的主机是信创电脑（多为 Linux arm64）；普通笔记本 / 台式机继续下一步。
2. **Windows 自查**：开始菜单 → 设置 → 系统 → 系统信息（或"关于"）→ 看"系统类型"：写 `x64-based processor` 就是 Windows x64，写 `ARM-based` 就是 Windows ARM。
3. **Linux 自查**：打开终端输入 `uname -m` 回车 —— `x86_64` 是 Linux x64，`aarch64` 是 Linux arm64。
4. 仍无法判断、线索指向普通家用 / 办公电脑时，**默认按 Windows x64**（占比最大），并建议顺带给 Linux x64 也打一个包（成本很低）。

确认后先向用户复述一句"好的，软件将运行在 XXX 上"，再开始开发；多平台时列出全部构建目标。

## 第 2 步：确认需求

用一两句话复述软件要做什么，列出主要窗口和控件（按钮、表格、输入框……）。关键信息缺失时用表单补问一次即可，不要反复追问；先做出能跑的最小版本再迭代。

## 第 3 步：项目骨架（纯 Go）

```bash
mkdir myapp && cd myapp
go mod init example.com/myapp
go get gioui.org@latest
```

注意模块路径就是 `gioui.org`（不是 gioui.org/v2），且**没有 `import "C"`、不设置 `CGO_ENABLED=1`**（Windows 目标）。

最小可运行骨架 `cmd/myapp/main.go`（API 对应 Gio v0.10.x）：

```go
package main

import (
	"log"
	"os"

	"gioui.org/app"
	"gioui.org/font/gofont"
	"gioui.org/layout"
	"gioui.org/op"
	"gioui.org/text"
	"gioui.org/unit"
	"gioui.org/widget"
	"gioui.org/widget/material"
)

func main() {
	// Gio 要求 app.Main() 在主 goroutine；窗口循环放子 goroutine
	go func() {
		var w app.Window // 零值即可用，第一次 Event() 时创建系统窗口
		w.Option(app.Title("我的桌面软件"), app.Size(unit.Dp(400), unit.Dp(300)))
		th := material.NewTheme()
		th.Shaper = text.NewShaper(text.WithCollection(gofont.Collection()))
		var ops op.Ops
		var btn widget.Clickable
		for {
			switch e := w.Event().(type) {
			case app.DestroyEvent:
				if e.Err != nil {
					log.Fatal(e.Err)
				}
				os.Exit(0)
			case app.FrameEvent:
				gtx := app.NewContext(&ops, e)
				if btn.Clicked(gtx) {
					os.Exit(0)
				}
				layout.Center.Layout(gtx, func(gtx layout.Context) layout.Dimensions {
					return layout.Flex{Axis: layout.Vertical, Alignment: layout.Middle}.Layout(gtx,
						layout.Rigid(func(gtx layout.Context) layout.Dimensions {
							return material.H3(th, "你好，桌面软件已运行").Layout(gtx)
						}),
						layout.Rigid(layout.Spacer{Height: unit.Dp(16)}.Layout),
						layout.Rigid(func(gtx layout.Context) layout.Dimensions {
							return material.Button(th, &btn, "退出").Layout(gtx)
						}),
					)
				})
				e.Frame(gtx.Ops)
			}
		}
	}()
	app.Main()
}
```

要点：

- **事件循环**：`w.Event()` 阻塞等事件；`app.FrameEvent` 来一帧画一帧（立即模式：每帧重绘整个界面），`app.DestroyEvent` 表示窗口关闭。
- 目录保持简单：`cmd/myapp/main.go` + 按功能分包（`ui/`、`core/`），不要过度设计。
- 常用控件在 `widget/material`：`material.Button(th, &clickable, "文本")`、`material.Label(th, unit.Sp(14), "文本")`、`material.Editor(th, &editor, "提示")`、`material.CheckBox`、`material.Switch`、`material.Slider`、`material.ProgressBar`、`material.List`；布局用 `layout.Flex`（`layout.Rigid`/`layout.Flexed`）、`layout.Stack`（层叠）、`layout.Inset`（边距）、`layout.Spacer`。需要导航栏、抽屉等更丰富组件时加扩展模块 `go get gioui.org/x@latest`（`gioui.org/x/component`）。
- 状态存局部变量 / 结构体，UI 每帧重建；goroutine 里异步更新数据后调用 `w.Invalidate()` 触发重绘。
- **界面含中文必须捆绑中文字体**（内置 gofont 只有拉丁字形，中文会显示为方块）：

  ```go
  //go:embed NotoSansSC-Regular.ttf
  var fontCN []byte

  faces, err := opentype.ParseCollection(fontCN) // 包: gioui.org/font/opentype
  if err != nil { log.Fatal(err) }
  th.Shaper = text.NewShaper(text.WithCollection(faces))
  ```

  字体文件可用沙盒自带的 `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`，或网上下载 Noto Sans SC。
- 数据文件存 `app.DataDir()` 返回的目录（跨平台各平台惯例位置），交付说明里写给用户听。
- `go` 命令找不到时检查 `/usr/local/go/bin` 是否在 PATH 中。

## 第 4 步：编译验证

```bash
go vet ./... && go build ./...
```

沙盒 / SSH 等无图形界面环境里 `go run .` 弹不出窗口属正常现象：用编译通过做开发期验证，可执行文件交给用户在自己电脑上运行确认。

## 第 5 步：按目标平台打包

**Windows 目标（本说明书的主场景）：零额外依赖，只要 Go。**

```bash
# Windows x64（从 Linux 沙盒直接交叉编译；-H windowsgui 去掉黑色控制台窗口）
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 \
  go build -ldflags="-s -w -H windowsgui" -o dist/myapp-windows-x64.exe ./cmd/myapp

# Windows ARM64（骁龙笔记本，同样是纯 Go）
CGO_ENABLED=0 GOOS=windows GOARCH=arm64 \
  go build -ldflags="-s -w -H windowsgui" -o dist/myapp-windows-arm64.exe ./cmd/myapp

# macOS（Apple Silicon，同样是纯 Go；交付裸二进制即可，双击可运行）
CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 \
  go build -ldflags="-s -w" -o dist/myapp-mac-arm64 ./cmd/myapp
```

**Linux 目标（例外：Gio 在 Linux 上经 cgo 加载 Vulkan，需要构建期依赖）：**

- `gcc`（沙盒已预装 build-essential）+ 图形开发头文件 `libvulkan-dev libwayland-dev xorg-dev`（沙盒未预装，按需 `apt-get install`）
- 报 wayland 相关编译错误、或不需要 Wayland 时，加 `-tags nowayland` 只保留 X11 后端

```bash
# Linux x64
CGO_ENABLED=1 GOOS=linux GOARCH=amd64 \
  go build -ldflags="-s -w" -o dist/myapp-linux-x64 ./cmd/myapp
```

Linux arm64 交叉编译需要 `gcc-aarch64-linux-gnu` + arm64 版头文件，链路较繁琐，建议在目标机上直接 `go build`。

其他打包细节：

- 在 Windows 本机开发时：装好 Go 直接 `go build`，什么都不用配。
- Windows exe 加图标 / 版本信息：用纯 Go 工具 `go install github.com/tc-hib/go-winres@latest`，`go-winres make --in icon.json` 生成 `.syso` 放在包目录，`go build` 时自动嵌入。

## 交付物

每个目标平台交付：

1. 可执行文件：Windows 为 `.exe`（双击即用）；Linux 为二进制或 .tar.gz（首次运行前 `chmod +x`）。
2. 一页大白话使用说明：怎么启动、数据存哪儿（`app.DataDir()` 的位置）、出问题找谁，不用技术术语。
3. 多平台时逐个说清哪个文件对应哪种电脑，文件名带平台名（如 `myapp-windows-x64.exe`）。

## 常见问题

| 现象 | 原因与解决 |
|---|---|
| Windows 双击 .exe 弹出黑色控制台 | 缺 `-H windowsgui` |
| 中文显示为方块 | 没捆绑中文字体；见第 3 步的 `opentype.ParseCollection`，不能只用 gofont |
| Linux 编译报 vulkan.h / wayland-client 找不到 | 缺构建期头文件：`apt-get install libvulkan-dev libwayland-dev xorg-dev`；或加 `-tags nowayland` |
| `internal/vk: build constraints exclude all Go files` | 在 Linux 上把 `CGO_ENABLED=0` 用于了 Linux 目标；Linux 目标必须 `CGO_ENABLED=1`，Windows 目标才是 0 |
| 运行报打不开显示 / 找不到显示器 | 服务器 / 无桌面环境跑不了 GUI；交付给用户的桌面电脑运行 |
| 用户需求含糊 | 先交付最小可用窗口，再按反馈迭代 |
