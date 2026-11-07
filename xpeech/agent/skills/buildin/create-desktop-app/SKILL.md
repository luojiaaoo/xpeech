---
name: create-desktop-app
short_description: 创建桌面应用
description: 用户需要创建桌面软件时触发，无需用户指定技术栈。用 MyGo 框架（github.com/egoist/mygo）原生绘制（Go 包 ui）生成桌面应用；只允许生成原生绘制的软件，禁止 webview/网页前端方案。也适用于在 MyGo 项目中添加窗口、控件、界面。
---

# MyGo 原生绘制（Native UI）用法

## 硬性规则（最高优先级）

1. **只允许生成原生绘制的软件**：界面一律用 Go 包 `ui` 编写、由 MyGo 在 GPU 上自绘
   （Metal/D3D11/OpenGL），窗口内容用 `mygo.WindowOptions{Content: ui.View(...)}` 挂载。
2. **禁止 webview 与网页前端**：不得生成 HTML/JavaScript/TypeScript、Vite/Bun/npm 工程、
   `index.html`、`src/main.ts`、`mygo.config.ts`、`packages/*` 依赖，也不得使用
   webview 专用 API（`mygo.Bind`、`mygo.NewEvent`、`win.Page()`、`ui.WebView` 等）。
   一个窗口要么是原生 UI 要么是网页——本 skill 只做原生 UI。
3. 原生 UI 项目是**纯 Go 模块**：无前端、无 Bun、无代码生成；只有 `go.mod`、`mygo.json`、
   `main.go`、`main_test.go`、`resources/icon.png`。
4. 遇到用户想要网页技术（HTML/CSS/现有 Web 代码）时，提醒该需求超出本 skill 范围并拒绝
   用 webview 实现，可改用 `ui` 的绘制能力（Painter、自定义控件）实现同样的视觉效果。
5. 本 skill 与文档以用法为准；API 标识符（函数名、方法名、类型）必须与下文原型一致，
   不要臆造。需要更细的字段时查阅下载的框架文档 `mygo-main/docs/ui/`（同名 .md）。

## 项目结构与命令

```sh
mygo init -template native my-app   # 创建原生 UI 项目（或 go run github.com/egoist/mygo/cmd/mygo@latest init -template native my-app）
cd my-app
go tool mygo dev                    # 开发：构建并启动，改 .go 后自动重建重启
go test                             # 视图测试，无窗口，和单元测试一样快
go tool mygo vet                    # 检查把构建期值存进结构体/全局/协程的错误
go tool mygo build                  # 打包：macOS .app+.dmg / Windows .exe+安装器 / Linux 可执行+deb
go tool mygo build -platform darwin/universal,windows/amd64,linux/amd64   # 交叉打包（无 cgo）
```

```
my-app/
├── main.go          应用：状态、view、窗口
├── main_test.go     view 测试
├── go.mod           CLI 以 tool 方式固定版本：go tool mygo
├── mygo.json        应用名、identifier、版本（字段见 platform.md；勿写前端字段，linux 设 "webview": false）
└── resources/icon.png   1024×1024 应用图标
```

## 应用骨架（规范写法）

```go
package main

import (
	"log"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
)

// counter 是应用状态（自定义类型，字段即状态）。
type counter struct{ n int }

// view 由状态构建界面，每帧在主线程调用；c 是该窗口的 *ui.Context。
func (s *counter) view(c *ui.Context) {
	if c.Shortcut(0, ui.KeyUp) {
		s.n++
	}
	t := c.Theme()
	ui.Column(c).Fill().Center().Gap(16).Children(func() {
		ui.Textf(c, "%d", s.n).FontSize(56).Bold()
		if ui.PrimaryButton(c.Key("increment"), "+").Label("Increment").Clicked() {
			s.n++
		}
		ui.Text(c, "↑ 也能计数").FontSize(12).TextColor(t.TextMuted)
	})
}

func main() {
	s := &counter{}
	mygo.App.WhenReady(func() {
		mygo.NewWindow(mygo.WindowOptions{
			Title:   "Counter",
			Width:   360,
			Height:  260,
			Content: ui.View(s.view),
		})
	})
	if err := mygo.App.Run(); err != nil {
		log.Fatal(err)
	}
}
```

要点：
- `mygo.App.WhenReady(func(){...})` 里建窗口；`mygo.App.Run()` 阻塞到退出，出错 `log.Fatal`。
- `ui.View(方法值)` 把 `func(c *ui.Context)` 变成窗口内容；同一个 Content 可给多个窗口，
  各窗口元素状态独立。
- 示例中 `app.xxx` 字段/方法是**你自己**在状态类型上声明的（如 `counter.n`）。

## 视图模型（立即模式）

- view 是「状态 → 界面」的纯构建函数，每帧运行在**主线程**；一帧可能构建多遍。
  构建期产生的值（Element、builder）**不得**存进结构体/包变量/协程——长期持有的用
  `ui.Handle`、`ui.Services`、`ui.Local`（自定义控件的跨帧局部状态，`io.Closer` 值会在
  停止构建时关闭），数据和控件状态（如 `ListState`）放在应用状态里。
- **事件查询**（询问上次帧以来发生了什么，poll 风格，可在 if 里直接 `return`/`break`）：
  `.Clicked() bool`、`.Changed() bool`（绑定值变了）、`.Submitted() bool`（单行输入回车）。
- **回调风格**（构建完成、绑定输入应用后执行；修改正在遍历的集合时用它）：
  `.OnClick(fn)`、`.OnChange(fn)`、`.OnSubmit(fn)`。
- **改值控件拿指针**：`ui.Checkbox(c, &app.x, "Label")`、`ui.TextInput(c, &app.name)`——
  控件在构造函数返回前就已把用户输入写回指针，下一行读到的是新值（ImGui/egui 风格）。
- **键（Key）**：集合变化时让控件状态跟着条目走——构造**之前**给键：
  `ui.Button(c.Key(item.ID), "Del")`，或 `c.Key(k)` 设下一个控件的键（重复调用以最后一个为准）。
  键在兄弟间唯一；带键的容器（`ui.Column(c.Key(id))`）给子树独立身份域。
- **从其他协程改状态**（view 只在主线程跑）：
  - `win.Update(func(){ ... })` —— 主线程执行后重建并绘制（每次调用都重建）；
  - `win.Invalidate()` —— 只要求重画；
  - `mygo.RunOnMain(func(){ ... })` —— 主线程执行，不重画；
  - `c.Invalidate()` —— 要求再来一帧；`c.After(d)` —— 延时后再来一帧。
  低频轮询数据用 `RunOnMain` 比较后 `win.Invalidate()`，避免 `Update` 的无谓重建。
- 输入在重建前被消费，一次输入只触发一次动作。`Disabled`/`ReadOnly`/`Step` 之类选项
  从设置它的那帧起影响之后的输入。

## 布局（flexbox + 网格，单位 DIP）

容器（`Children(func(){...})` 放子元素）：`ui.Column(c)`（竖排、横向拉满）、`ui.Row(c)`
（横排、垂直居中）、`ui.Box(c)`（同 Column）、`ui.Spacer`（占满剩余空间）、`ui.Divider`
（分隔线）、`ui.Grid(c)`（网格）。

常用链式方法（多数元素通用）：
- 尺寸：`Width/Height(v)`、`Size(w,h)`、`WidthPercent/HeightPercent`、`MinWidth/MaxWidth`
  （及 Percent 形式）、`Fill()`/`FillWidth()`/`FillHeight()`；flex：`Grow(1)`、`Shrink`、
  `Basis`、`AspectRatio`。
- 间距：`Gap(v)`/`GapX/GapY`、`Padding(...)`/`Margin(...)`（1/2/4 个值，CSS 简写语义）、
  `PaddingX/Y`、`MarginStart/End`（RTL 感知的逻辑边）；`ui.Auto` 作 margin 值占据剩余空间
  （`Margin(0, ui.Auto)` 居中；`MarginStart(ui.Auto)` 推到行尾）。
- 对齐：`Justify(ui.Start|ui.Center|ui.End|ui.SpaceBetween|...)`（主轴）、
  `AlignItems(ui.Start|ui.Center|ui.End|ui.Stretch|ui.Baseline)`（交叉轴）、
  `AlignSelf`、`Center()`（双向居中）、`Wrap()`。
- 定位：`Absolute()` + `Top/Right/Bottom/Left(v)`（脱离流）；`Attach(at, self)` 把元素某点
  锚到父盒某点（`ui.AnchorTopRight`、`ui.AnchorCenter`、`ui.AnchorTopEnd`…）；
  `Sticky()`（滚动容器内粘滞）；`ZIndex(n)`（兄弟间叠放，高的先拿指针）。
- 其它：`Clip()`（圆角裁剪）、`Invisible()`（占位不画）、`Hidden(bool)`（隐藏并让位，
  保留内部状态）、`Dividers(w, color)`（子项间画线，`Gap` 要够宽）、`Debug()`（描边调试）。
- 元素通用事件：`.Hovered()`、`.Clicked()`、`.Focusable()`、`.Label("名")`（无文字元素
  的无障碍名/测试查找名）。

## 文本

- `ui.Text(c, "…")`、`ui.Textf(c, fmt, args…)`（自动换行）；`ui.RichText(c, ui.Span{...}, ...)`；
  `ui.Link(c, "标签", url)`。
- 样式链（可在文本或祖先上设，继承）：`FontSize(n)`、`Bold()`、`Italic()`、
  `FontWeight(w)`、`Font("monospace" | "Inter, Noto Sans JP")`、`TextColor(c)`、
  `TextAlign`、`Underline()/Strikethrough()/WavyUnderline()`、`LineHeight`、
  `LetterSpacing(n)`、`FontFeatures("tnum")`、`TextShadow(x,y,blur,color)`（仅一个）、
  `TextStroke(width, color)`、`TextBackground(c)`。
- 排版：`SingleLine()`（省略号）、`MaxLines(n)`、`Ellipsis(" →")`、`NoWrap()`、`Balance()`、
  `Pretty()`。
- 选择：`Selectable()`（文本或容器，容器内共享一个选择）、`Unselectable()`、
  `SelectionColor(c)`、`SelectedTexts()`、`CopyData(...)`。
- `ui.Span{Text, Font, Size, Weight, Color, Underline, Background, Radius, PaddingX/Y,
  BaselineShift, Shadow: ui.NewTextShadow(...)}`；RichText 的 `.Children(func(){...})`
  放内联元素（Text/Link/RichText）。
- 字体：默认系统字体；`ui.RegisterFont(data, family)`（`//go:embed` + `init()` 注册）；
  `ui.FontFamilies()` 列出家族。

## 样式与主题

- 颜色：`ui.RGB/RGBA`、`ui.Hex("#2563eb")`、`ui.Transparent`、`ui.Oklch(l,c,h)`、
  `Mix/MixOklab(a,b)`、`color.Alpha(a)`。
- 盒样式链：`Background(c)`、`Radius(r)`、`Border(w, c)`/`BorderWidth/BorderColor/
  BorderStyle(ui.BorderDashed)`、`Shadow(x,y,blur,spread,c)`、`InsetShadow(...)`、
  `NoShadow()`、`Opacity(o)`（整组淡出）、`Blur(sigma)`、`Gradient(from,to,angle)`、
  `LinearGradient/ConicGradient/RadialGradient`、`Cursor(ui.CursorPointer|...)`、
  `ClipInset(...)`、`MaskLinearGradient(...)`、`Cache()`（GPU 位图缓存重绘结果）。
- 状态驱动样式（无样式表/选择器）：`if row.Hovered() { row.Background(t.SurfaceHover) }`；
  调用返回的元素再链式，只影响这一次调用。
- 主题：`t := c.Theme()`，字段 `Background/Text/TextMuted/Surface/SurfaceHover/Border/
  Accent/AccentText/Inverse/InverseText/Spacing/FontSize/Radius/Focus...`。
  改主题：复制后 `c.SetTheme(&t)`（整个窗口）；子树用 `ui.Themed(c, t)`；
  `ui.LightTheme()`/`ui.DarkTheme()`。度量：`t.Space(n)`、`t.Rem(n)`。
- 桌面偏好：`c.Preferences()`（`Accent`、`HighContrast`、`TextScale`、`ReduceMotion`、
  `Hover()` 等）。`ReduceMotion` 下 `Animate` 直接跳到目标——做动效时要读它。

## 测试（无窗口跑 view）

```go
func TestCounter(t *testing.T) {
	s := &counter{}
	tt := ui.NewTester(s.view, 360, 260) // view、宽、高
	if err := tt.Click("Increment"); err != nil { t.Fatal(err) } // 按文字/Label 查找
	if s.n != 1 || !tt.HasText("1") {
		t.Errorf("count %d, texts %q", s.n, tt.Texts())
	}
}
```

- 查找按可见文字或 `Label`：`tt.Click("Close")`、`tt.Find("Volume")`、`tt.Focused("Name")`。
- 指针：`tt.Click/ClickAt/ClickWith(ui.Shift, "Row 3")/RightClick/Press/Move/Scroll/Wheel`。
  键盘：`tt.Key(ui.Cmd, ui.KeyS)`、`tt.Type("文本")`、`tt.TypeKey`、`tt.Compose`、
  `tt.Command("copy")`。菜单：`tt.Menu()`、`tt.ChooseMenuItem("Move to", "Archive")`。
  窗口：`tt.SetSize/SetScale/SetZoom/SetDark/SetPreferences/SetDirection`。
- **时间**：测试有独立时钟，`tt.Advance(d)` 推进时间并跑到期的帧——**绝不 sleep**；
  动画、双击、提示等都跟随时钟；外部改状态后先 `tt.Frame`。
  `tt.FramePending()` 问视图是否还在要帧；`tt.Image()` 截帧快照。
- 无障碍断言：`tt.Access(name)`、`tt.AccessByRole(ui.RoleStatus, "Reply")` 返回
  `ui.AccessInfo`（Role、Name、Value、Checked、Busy…）。

## 无障碍（写 view 时遵守）

- 无文字元素给 `Label("名")`（也用于测试查找）；表单用 field 的 label 命名控件。
- 自绘控件用 `Role(...)`（`ui.RoleSwitch`、`ui.RoleHeading`+`Level(n)`、`ui.RoleNavigation`、
  `ui.RoleRow/RoleCell/RoleColumnHeader`、`ui.RoleListBox/RoleOption`…）和状态方法
  （`Checked(on)`、`Expanded(open)`、`Value(s)`、`Range(lo,hi,value)`、`AccessSelected(on)`、
  `RoleDescription(s)`）。
- `c.Announce("3 条结果")` 让读屏播报；`AccessHidden(true)` 把元素（含子树）移出无障碍树；
  `IgnoreInput(true)` 忽略输入。

## 参考文件（按需读取）

- [references/widgets.md](references/widgets.md) —— **控件目录**：全部内置控件的构造签名、
  链式方法、事件查询与最小示例（含代码编辑器/差异/Markdown/富文本/图表等复合控件）。
- [references/guides.md](references/guides.md) —— **UI 进阶**：输入与快捷键、拖放、浮层
  （Popover/Modal/Overlay）、页面路由、自定义控件（无外观基座、ui.Local）、绘制（Painter/路径）、
  动画（Animate/弹簧/位姿/手势/关键帧/时间线）、布局过渡与共享元素、变换（2D/3D）、GPU 着色器。
- [references/platform.md](references/platform.md) —— **平台用法**：应用生命周期（mygo.App）、
  窗口（WindowOptions/方法/事件、标题栏/缩放/毛玻璃）、Flyout/面板、菜单与托盘、mygo.json 配置、
  CLI 命令。

写任何界面前先在 widgets.md 找现成控件，不要手拼；超出控件能力再用 guides.md 的基座/绘制。

## 检查清单

- [ ] 界面 100% 由 `ui` 包构建，`Content: ui.View(...)`；无 HTML/TS/Bun/webview/IPC。
- [ ] 状态放自定义结构体；构建期值不落库（Handle/Services/Local 才跨帧）。
- [ ] 集合中的控件都带 `Key`；改集合用 `OnClick` 回调。
- [ ] 协程改状态走 `win.Update`/`RunOnMain`，view 只在主线程。
- [ ] 无文字控件有 `Label`；测试用 `ui.NewTester` + `Advance`，不 sleep。
- [ ] 动效读 `c.Preferences().ReduceMotion`。
- [ ] `go test` 通过；`go tool mygo vet` 干净。
