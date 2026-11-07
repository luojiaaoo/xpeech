# 应用、窗口、菜单、配置与 CLI（原生 UI 项目）

配合 SKILL.md 使用。以下均为原生绘制应用能用的部分；webview 专属 API
（`mygo.Bind`、`mygo.NewEvent`、`win.Page()`、Cookie/Session、页面事件等）一律不用。

## 应用生命周期（`mygo.App`，包 `github.com/egoist/mygo`）

```go
func main() {
	mygo.App.WhenReady(func() {          // 就绪后建窗口
		mygo.NewWindow(mygo.WindowOptions{Title: "App", Width: 800, Height: 600,
			Content: ui.View(app.view)})
	})
	if err := mygo.App.Run(); err != nil { log.Fatal(err) }  // 主 goroutine 调用，阻塞到退出
}
```
- 所有 MyGo 方法任意协程安全（自动跳主线程）；监听器/菜单回调在主线程跑，保持短小；
  `mygo.RunOnMain(fn)` 显式上主线程。
- 事件（`On…` 返回移除函数）：`OnReady/WhenReady`、`OnActivate(func(hasVisibleWindows bool))`、
  `OnDidBecomeActive/OnDidResignActive`、`OnWindowCreated/OnWindowAllClosed`、
  `OnBeforeQuit/OnWillQuit(func(e *mygo.QuitEvent))`（`e.PreventDefault()` 取消）、`OnQuit`。
- 退出：默认关完最后窗口即退出（有 `OnWindowAllClosed` 监听则不）；macOS 惯例
  `OnWindowAllClosed(func(){ if runtime.GOOS != "darwin" { mygo.App.Quit() } })` + `OnActivate` 重开窗口。
  `App.Quit()`（可被取消）/ `App.Exit(code)`（立即）/ `App.Relaunch()`。
- 单实例：`App.RequestSingleInstanceLock() bool`（Run 前调）+ `OnSecondInstance(func(args []string, dir string))`。
- 深链/文件：`OnOpenURL(func(rawURL string))`、`OnOpenFile(func(path string))`（Run 前注册连启动时一起收）、
  `RegisterURLScheme("myapp")`；最近文档 `AddRecentDocument/RecentDocuments/ClearRecentDocuments`。
- 开机启动：`App.SetOpenAtLogin(true)`、`WasOpenedAtLogin()`（配合 `WindowOptions{Hidden: ...}`）。
- 目录：`App.Path(mygo.PathUserData|PathCache|PathLogs|PathResources|PathAppData|PathHome|PathTemp|
  PathExe|PathDesktop|PathDocuments|PathDownloads|...) (string, error)`。
- 其它：`App.Name()/Version()/Locale()/PreferredLanguages()`、`mygo.IsDev()`、`App.IsPackaged()`、
  `App.SetBadgeCount(n)`、`App.Dock.SetMenu/SetBadge/Bounce/Hide`、
  `App.SetActivationPolicy(mygo.ActivationPolicyAccessory)`（菜单栏应用，藏 Dock 图标）、
  `App.Focus()/IsActive()`。
- Run 前可用：`SetName/SetMenu/SetOpenAtLogin/RegisterURLScheme/App.Path/App.Locale`；
  剪贴板/显示器/对话框/通知/全局快捷键/托盘/`Theme.IsDark` 等在 `Run` 后（`WhenReady` 起）才可用。

## 窗口

`mygo.NewWindow(mygo.WindowOptions{...})` 返回 `*mygo.Window`；原生 UI 用 `Content: ui.View(...)`
（`URL`/`Page` 被忽略）。零值 = 800×600 居中可调。尺寸位置都是 DIP。

常用字段：`Title, Content, Width, Height, X, Y, MinWidth/MinHeight/MaxWidth/MaxHeight,
FitContent（随内容定尺寸）, StateKey（记忆窗口几何到 window-state.json）, Hidden（建而不显，
OnReadyToShow 首帧后 Show）, Maximized, FullScreen, Frameless, TitleBarStyle: mygo.TitleBarHidden,
TrafficLightPosition, TitleBarHeight, HideWindowButtons, Transparent, Vibrancy: mygo.VibrancyHUD|
VibrancyGlass|VibrancyGlassClear, VibrancyTint("rgba(0,122,255,0.3)") , CornerRadius,
BackgroundColor("#1e1e1e"), Opacity, DisableResize/Move/Minimize/Maximize/Close/FullScreen,
AlwaysOnTop, SkipTaskbar, AutoHideMenuBar, Parent, Modal, Panel（置顶小面板）, UseContentSize`。

方法（节选）：`SetSize/SetContentSize/SetPosition/SetBounds/Center()`、`Maximize/Unmaximize/
ToggleMaximize/Minimize/Restore/SetFullScreen/ToggleFullScreen`、`Hide/Show/IsVisible`、
`Focus/Blur`、`SetTitle/SetResizable/SetAlwaysOnTop/SetOpacity/SetIcon(png)/SetVibrancy(...)`、
`SetProgressBar(mygo.ProgressBar{Value: 0.4})`、`FlashFrame(true)`、`Close()`（触发 OnClose，可拦）/
`Destroy()`、`win.NativeHandle()`（配 `mygo.RunOnMain` 调原生 API）、`win.CapturePage()`（截图，原生 UI 也行）、
`win.SetMenu(menu)`。

事件：`OnClose(func(e *mygo.CloseEvent))`（`e.PreventDefault()` 拦截）、`OnClosed/OnFocus/OnBlur/
OnShow/OnHide/OnResize/OnMove/OnMaximize/.../OnReadyToShow（原生 UI=首帧已画）/OnFileDrop`。
查找：`mygo.Windows()`、`mygo.FocusedWindow()`、`mygo.WindowByID(id)`。

## 原生 UI 窗口专属（`ui/windows.md`）

- 标题栏：`TitleBarStyle: mygo.TitleBarHidden` 时 `c.TitleBar()` 返回按钮占用的房间
  （`bar.Height/Left/Right`，全屏为零）；`el.DragWindow()` 让元素拖动窗口（双击缩放/最大化）。
  `HideWindowButtons: true` 时 view 自画按钮并调 `win.Minimize/Maximize/Unmaximize/Close`。
- 缩放：`c.SetZoom(z)`（0.25–5）、`c.Zoom()`；view 内长度=内容 DIP，`WindowOptions.Width/Height`
  是窗口 DIP。
- 毛玻璃：窗口带 `Vibrancy` 时 `c.Vibrancy()` 为真——内容不画的地方透出材质：
  `c.Root().Background(ui.Transparent)`，实体部分给 `t.Surface/t.Background` 背景。
  Liquid Glass：`VibrancyGlass` + `CornerRadius`，`win.SetVibrancyTint("light-dark(...)")`。
- 激活态：内置控件在窗口失焦时淡化为 `t.InactiveAccent`；自绘用 `c.WindowActive()`。
  `c.WindowHidden()`：隐藏时暂停工作（计时器等）。
- 从 view 关窗口：先建替换窗口再 `win.Close()`（关完最后窗口应用就退出）。
- 纯原生应用不需要 webview 运行时（Linux 只要 GTK 3；Windows 不需要 WebView2）。

## Flyout 与面板

`mygo.NewFlyout(mygo.FlyoutOptions{Parent/Tray, Anchor mygo.Rectangle（ui.Element.Bounds 给出）,
Placement: mygo.PlacementBottomStart..., Gap, Width, Height, FitContent, Focusable, Popover,
Shadow, Content: ui.View(...)})`——锚在父窗口或托盘的无边框小窗（菜单栏应用面板）；
`SetAnchor/SetSize/Close`。全局快捷键召唤面板用 `mygo.GlobalShortcut.Register("Alt+Shift+Space", fn)` +
`Window.Hide`（`OnClose` 里 `e.PreventDefault(); win.Hide()`）。

## 菜单与托盘（`menus.md`）

```go
menu := mygo.NewMenu([]*mygo.MenuItem{
	{Label: "New", Accelerator: "CmdOrCtrl+N", Click: func(item *mygo.MenuItem, win *mygo.Window) { ... }},
	mygo.Separator(),
	{Label: "Quit", Role: mygo.RoleQuit},
})
mygo.App.SetMenu(menu)   // macOS=应用菜单栏；Linux/Windows=给每个无菜单窗口
win.SetMenu(menu)        // 每窗口菜单（Linux/Windows）
```
- `MenuItem`：`Label, Accelerator("CmdOrCtrl+N"), Click（主线程）, Type: mygo.MenuItemNormal|
  MenuItemCheckbox|MenuItemRadio|MenuItemSubmenu, Checked, Disabled, Hidden, ToolTip, Icon,
  Subtitle, ID, Role, Submenu`。运行时改：`item.SetLabel/SetEnabled/SetChecked/SetAccelerator`、
  `Menu.Append/Insert`。
- Role（自带标签/快捷键）：`RoleUndo/Redo/Cut/Copy/Paste/Delete/SelectAll`、`RoleBold/Italic/
  Underline/Strikethrough/AlignLeft/...`、`RoleReload/RoleToggleDevTools`（原生 UI=开检查器）、
  `RoleToggleFullScreen/Minimize/Zoom/Close`、`RoleNew/Open/Save/SaveAs/Rename/...`、`RoleQuit`
  （macOS 另有 `RoleAbout/Hide/HideOthers/...`）；整菜单 `RoleFileMenu/EditMenu/FormatMenu/ViewMenu/
  WindowMenu/RoleAppMenu`。一套模板全平台（macOS-only 的别处自动省略）。
- 图标：`mygo.NewMenuIcon(pngBytes, template)`（≥32×32 PNG）、`mygo.NewMenuIconSVG(svg, template)`。
- 快捷键语法：修饰 `CmdOrCtrl/Cmd/Super/Meta/Win/Ctrl/Alt/Shift` 用 `+` 连；键名 `F1-F24`、
  `Enter/Tab/Space/Backspace/Delete/Escape/Home/End/PageUp/PageDown/PrintScreen/...`。
  `mygo.GlobalShortcut.Register(accel, fn)` 同语法。
- 原生 UI 元素的右键菜单用 `el.ContextMenu(...)`（ui 包），不是 Page 的。
- 托盘：`mygo.NewTray(mygo.TrayOptions{Icon: pngBytes（16×16pt）, IconIsTemplate, ToolTip, Menu, ID})`；
  `tray.OnClick/OnRightClick`（无菜单时）、`SetIcon/SetMenu/SetToolTip/Destroy/PopUpMenu/Bounds`。
  **托盘不保活**——用 `App.OnWindowAllClosed(func(){})` 或隐藏代替关闭；
  菜单栏应用加 `App.SetActivationPolicy(mygo.ActivationPolicyAccessory)`。
  Linux 需 `libayatana-appindicator3`。

## mygo.json（原生项目配置，全字段可选）

```json
{
  "name": "My App", "slug": "my-app", "identifier": "com.example.myapp",
  "version": "0.1.0", "icon": "resources/icon.png",
  "resources": ["locales"], "urlSchemes": ["myapp"],
  "fileAssociations": [{"ext": ["md"], "name": "Markdown", "role": "Editor"}],
  "i18n": {"dir": "locales", "source": "en"},
  "linux": { "webview": false }
}
```
- 默认值：`name`=目录名、`identifier`=`com.mygo.<name>`、`version`=`0.1.0`、`icon`=`resources/icon.png`。
- **原生项目不要写前端字段**（`devUrl/devCommand/buildCommand/frontendDist/bindings`）。
- `linux.webview: false` 让打包依赖只剩 GTK（原生 UI 应用应设置）。
- 发布/签名：`updates`（`publicKey`、`github`/`url`/`s3`）、`macos`（`signingIdentity`、`entitlements`、
  `infoPlist` 等）、`windows`（`certificate` 等）、`linux`（`maintainer`、`categories`、`depends`）。

## CLI（原生项目用 `go tool mygo`，go.mod 固定版本）

```sh
mygo init -template native my-app   # -name/-module/-mygo <checkout> 可选
go tool mygo dev [dir]              # 热重启：r 重建、c 清屏、q 退出；-lang zh-Hant、-sign
go tool mygo build [dir]            # -platform darwin/universal,windows/amd64,linux/amd64
                                    # -debug（保留检查器）、-o、-skip-dmg、-upload
go tool mygo vet [dir]              # 检查构建期值误存
go test ./...                       # 视图测试（没有 mygo test 命令）
go tool mygo keygen                 # 自动更新密钥对
go tool mygo doctor                 # 环境体检
MYGO_INSPECTOR=1 ...                # 生产构建保留原生 UI 检查器
```
- `mygo generate` 仅对有前端的项目有用（TypeScript 客户端）+ i18n 目录生成；原生项目只用到 i18n。
- `mygo dev` 构建名 `<name> Dev`、标识 `<identifier>.dev`（数据互不干扰）。
