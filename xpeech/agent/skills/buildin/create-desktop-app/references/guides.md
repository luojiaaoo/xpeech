# MyGo 原生 UI 进阶用法

配合 SKILL.md 使用。只讲用法；标识符与框架文档一致。

## 输入：每帧询问发生了什么

元素查询（构建元素处就地询问上次帧以来的事件）：
- 指针：`Hovered()`、`Pressed()`、`Clicked()`、`DoubleClicked()`、`RightClicked()`、
  `Dragged() (dx, dy, held)`、`DragVelocity()`、`DragEnded()`、`PointerPosition() (x,y,ok)`、
  `PressedButton() (b, ok)`（0 主/1 右/2 中）、`PressedPointer()`（`ui.PointerMouse/Pen/Touch` + 压感）、
  `PressCanceled()`（系统取消的按压：不产生 Clicked/Drop，手势预览要回滚）、
  `ClickModifiers()`、`PressedInside()`。
- 文件拖入：`FileDragOver()`、`DroppedFiles() []string`；`c.Modifiers()` 当前修饰键。
- 指针穿透：`PassThrough()`（不下传）、`HitTest(func(x, y, r) bool)`（按位置放行）、
  `IgnoreInput(true)`（整棵子树穿透且不进键盘焦点）。
- 语义：最内层元素拿指针；`Clicked` = 在元素内松开、或键盘/辅助技术激活。

键盘焦点：
- `Focusable()`（可点聚焦、Tab 巡航）、`AutoFocus()`（出现即聚焦，如对话框首字段）、
  `Focus()`（保持焦点）、`Focused()`、`FocusVisible()`、`FocusWithin()`、`FocusLeft()`（焦点离开子树）、
  `TabStop(false)`（可聚焦但不进 Tab 序）、`FocusGroup(ui.Horizontal|ui.Vertical)`（整组一个 Tab 停靠，
  方向键内部移动）、`FocusBind(&field, value)` + `ui.FocusedValue(c, &field)`（焦点绑数据）。
- `c.FocusTakesText()`：焦点控件是否吃打字——单键快捷键要判它。

跨帧持久句柄 `ui.Handle`（存应用状态，如 `search ui.Handle`）：
```go
ui.TextInput(c.Key("search"), &app.query).Bind(&app.search)  // 绑定
app.search.Focus()          // 请求聚焦（隐藏时等待）
el := c.Resolve(handle)     // 本帧元素
handle.OnShortcut(c, ui.Cmd, ui.KeyK, fn)   // 控件启用时才触发
```

快捷键：
- `c.Shortcut(mods, key)`：全窗口精确修饰键（`ui.Cmd` = macOS⌘/他处 Ctrl）；
  `el.Shortcut(mods, key)` 仅焦点在元素子树内时（优先于 c 级）；
  `c.OnShortcut(mods, key, fn)` / `el.OnShortcut(...)` 回调形式；
  `c.OnWindowShortcut(...)` 全窗口（对话框在其上时优先）。
- `c.ShortcutRepeated()` 判自动重复（切换类动作用 `!c.ShortcutRepeated()` 防抖）。
- `c.KeyDown(key)` / `c.KeysDown()`：按住的键。`ui.KeyBack/KeyForward`（鼠标侧键）留给 router。
- 按钮暴露快捷键给读屏：`ui.Button(c,"Save").KeyShortcuts(ui.Cmd, ui.KeyS)`。
- 原始输入：`el.HandleInput(func(ev ui.InputEvent) bool)`——逐个键/指针事件（`Kind/Key/Code/
  PhysicalKey/Mods/Text/X/Y/Button/DX/DY/Precise/Pointer/Pressure/Repeat`；Kind：`ui.InputKeyDown/Up/
  Text/Select/PointerDown/Move/Up/Scroll/Zoom`），返回 true 消费；未消费的滚动落入外层滚动容器。
  自绘控件文字输入：`el.TextCaret(rect)`（启用输入法）/`el.InputMethod(false)`（Vim 普通模式）。
- 观察不消费：`el.ObserveInput(func(ev ui.InputEvent))`（容器看子项输入，capture 阶段）。
- 剪贴板/链接：`c.ReadClipboard/WriteClipboard/OpenURL`；异步版 `OpenURLThen/WriteClipboardThen`；
  多格式 `Services.ReadClipboardData/WriteClipboardData`。

## 拖放

窗口内 Go 值拖放：
```go
ui.Row(c.Key(id)).Drag(task).Children(...)            // 拖源
if v, over := ui.DragOver[*Task](zone); over { ... }  // 悬停高亮
if task, ok := ui.Drop[*Task](zone); ok { ... }       // 放下取值
el.Dragging() / el.DragMissed() (at ui.Point, ok)     // 拖中/落空（窗外=撕出）
```
- 列表重排：`ListState.Reorder = func(rows []int, to int)`；`GridState.Reorder` 同；
  `ListState.DragRows` 把行作为应用值拖出。

系统/跨应用（`github.com/egoist/mygo/ui/transfer`）：
```go
el.DragData(transfer.TextData("..."))                 // 或 URLData/FileData/Bytes/Lazy/New
el.DragDataFrom(func() transfer.Data { ... })         // 手势开始时快照
options := transfer.DropOptions{Formats: []transfer.Format{transfer.Text}}
if drop, ok := ui.DropData(zone, options); ok { b, _ := drop.Data.Read(transfer.Text) }
transfer.DragOptions{Operations: transfer.Copy | transfer.Move, Done: func(r transfer.Result)}
```
- `Done` 恰好一次、主线程；`transfer.Lazy` 提供器在主线程跑，别阻塞。
- `Window.CancelDataDrag()` / `StartDataDrag(...)`；Escape 取消拖放（取消不算 `DragMissed`）。

## 浮层（Popover / Modal / Overlay / AttachTo）

```go
ui.Popover(c, anchorEl, &app.open, func(){ ... })   // 锚点下方弹层
ui.Modal(c, &app.open, func(){ ... })               // 对话框语义：焦点困住、背景失效、读屏只看它
ui.Overlay(c, func(){ ... })                        // 绝对定位画在一切之上（抽屉等）
```
- `AttachTo(target, at, self)`：把元素某点贴到目标某点（`ui.AnchorBottomLeft` 等），
  越界自动翻面/内移；`AttachedAnchors() (at, self)` 问最终贴边；`AttachedOrigin()` 翻面缩放原点；
  `AttachEdgeMargin(v)` 离窗边距离。
- `PressedOutside()`：按在元素/锚点/其内嵌弹层之外（关弹层用）；`OverlayShortcut(mods, key)`
  接住 Escape 之类（最后构建的浮层最先）。
- 行为约定：Escape 关最上层浮层；对话框把焦点还给开它的元素；Popover 外点会**继续点到下面**；
  浮层取请求处主题（toast 用窗口主题）。
- 原生系统对话框（`mygo` 包）也可用：从协程调用，别让 view 等待。

## 页面与路由

```go
type app struct{ router *ui.Router }   // 建一次放状态：ui.NewRouter("/notes")

app.router.View(c, func(r *ui.Route) {
	switch {
	case r.Match("/notes"):        r.Title("Notes"); app.notes(c)
	case r.Match("/notes/{id}"):   app.note(c, r.Param("id"))
	case r.Match("/files/{path...}"): app.files(c, r.Param("path"))
	}
})
```
- 导航：`router.Push(path)/Replace/Back()/Forward()/Go(n)`；读取：`Path()/Query()/Location()`。
  相对路径同链接语义（`Push("?tab=info")`）。`ui.Link(c, label, path)` 跳路由。
- 嵌套布局：`r.View(c, func(r *ui.Route){...})` 处理剩余路径；页面外的侧栏/工具栏不重建。
- 页面切换动效：`router.Transition = ui.TransitionNone | ui.TransitionMatch`（共享元素 Magic Move）、
  `router.Animation = &ui.Snappy`。历史保留页面元素状态（各向 10 页）。
- 键盘/侧键（`ui.KeyBack/KeyForward`、⌘[⌘]、Alt+←→、触控板滑动）自动作用于持有焦点的 router。

## 自定义控件

- 每个控件都有**无外观基座**（`ButtonBase`、`CheckboxBase`、`SliderBase`、`SelectBase`、
  `MenuBase`、`DialogBase`、`TextInputBase`、`TabsBase`、`SegmentedBase`、`CollapsibleBase`、
  `TreeBase`、`ToastBase`…）：负责指针/键盘/焦点/无障碍语义，你样式化它返回的元素。
  **有状态基座必须先给键**：`ui.SegmentedBase(c.Key("period"), &app.view, 3)`。
- 元素局部跨帧状态：`ui.Local(el, "name", func() T) *T`——元素停止构建即销毁；
  `T` 或 `*T` 实现 `io.Closer` 则销毁时 `Close()`（主线程、别阻塞）。行内数据别放 Local
  （滚出视图就没了），应用共享数据更不能放。
- 自绘开关示例：
```go
func Spoiler(c *ui.Context, text string) {
	t := c.Theme()
	box := ui.Box(c).Padding(2, 6).Radius(4).Focusable().Role(ui.RoleSwitch).Label("Spoiler")
	shown := ui.Local(box, "shown", func() bool { return false })
	if box.Clicked() { *shown = !*shown }
	box.Checked(*shown)
	if *shown { box.Background(t.Surface) }
	box.Children(func() { ui.Text(c, text) })
}
```
- 编辑上报：`el.SetChanged()` 后 `el.Changed()` 为 true。禁用外观：`DimDisabled(false)` 自己画；
  不可用但仍要可聚焦/可读屏：`AccessDisabled`。
- 自定义文本控件用 `HandleTextInput` + 保留几何（text-input-client.md），别从零实现。

## 绘制（Painter）

```go
box := ui.Box(c).Height(120).Draw(func(p *ui.Painter, r ui.Rect) {
	// 在背景之后画；DrawOver 在子元素之后画。坐标是窗口 DIP。
	p.Fill(r.Inset(8), t.Accent, 8)                // Fill(rect, color, radius...)
	p.StrokePath(&path, 2, t.Text)
	p.Image(bitmap, r, ui.FillBox)
	p.ClipPath(path, func(){ ... })                 // 裁剪回调
	p.Transform(ui.Rotation(15), ox, oy, func(){ ... })
	w, h := p.MeasureText(width, ui.Span{Text: "hi"})
})
```
- Painter：`Fill/FillGradient/FillConicGradient/FillRadialGradient/Stroke/StrokeDashed/Shadow/
  Line/Image/ImageRegion/Icon/Text/RichText/Glyphs/MeasureText/Clip/ClipPath/FillPath/StrokePath/
  FillPathGradient/Transform/FocusRing/Scrollbar/Scale/Now()/AnimationFrame()/AnimationFrameRate(fps)/
  After(d)/Accelerated()`。
- `ui.Path`：`MoveTo/LineTo/Circle/Trim(from,to)/Cap/Join/Dash/FillRule(ui.EvenOdd)/Contains(x,y)`；
  SVG 语法 `ui.ParsePath(d)` / `ui.MustParsePath(d)`。
- 命中测试：`el.HitTest(func(x, y, r) bool)`（只在画出的像素上取指针）。
- 位图/画布：`ui.DecodeBitmap(bytes)`、`ui.NewCanvas(w, h)`、`canvas.Draw(p, func(cp *ui.Painter))`、
  `canvas.Bitmap()`。文本整形：`ui.Shape(text, ui.Font{Family, Size}) []Glyph`、`ui.Font.Metrics()`。
- Draw 内动画：`el.Animate(key, target, d)` / `el.AnimateTo(key, target, ui.Smooth)` / `el.Loop(key, period, ease)`。
- **注意**：Draw 可能一帧跑多次，只画不改状态；移动的绘制从 `p.Now()` 计算并调 `p.AnimationFrame()`；
  跨帧的 `ui.Path` 放 Draw 外面；窗口不可见时不画（自算运动用时间别用帧数）；
  `c.Preferences().ReduceMotion` 时 Animate 直接跳目标。

## 动画（值、弹簧、位姿、手势、时间线）

```go
card := ui.Box(c).Size(120, 80).Radius(12).Background(t.Accent)
target := float32(0); if app.open { target = 200 }
card.Translate(card.AnimateTo("x", target, ui.Snappy), 0)   // 每帧给目标，返回当前值
```
- 值动画：`Animate(key, target, d)`、`AnimateWith(key, target, d, ease)`、`Spring(key, target, opts)`、
  `AnimateColor(key, target ui.Color, a)`、`Jump(key, value)`。key 需帧间稳定（Key 或兄弟序）。
- `ui.Animation{Duration, Ease, Spring, Delay, Oklab, NonMotion}`（零值=弹簧）；
  预设 `ui.Smooth/Snappy/Bouncy`；`ui.SpringOptions{Duration, Bounce | Stiffness/Damping/Mass, Velocity}`；
  缓动 `ui.Linear/EaseIn/EaseOut/EaseInOut/Bounce(e)/CubicBezier(...)` 或任意 `func(t float32) float32`。
- 完成判定：`el.Settled(key)`（到目标）、`el.Arrived(key)`（到达那一帧，如淡出后销毁）。
- 位姿（纯绘制动，view 不重跑）：`el.AnimatePose(ui.Pose{X, Y, Scale, Rotate, Fade}, anim)`；
  入场惯用法：一帧摆位姿 + `c.Invalidate()`，下一帧归位。
- 拖拽甩动：`el.Draggable(&offset, ui.DraggableOptions{Axis, Bounds, Snap: ui.Nearest(pts...), Animation})`；
  查询 `Dragging()/DragOffset()/Dragged()/DragEnded() (vx,vy,ok)`；`ui.Project(v, vel)` 预测落点。
- 关键帧：`el.Keyframes(key, []ui.Keyframe{{At: 0, Value: ...}, {At: 300*time.Millisecond, ...}}, opts)`。
- 时间线（多补间一个播放头）：`tl := el.Timeline(key, ui.TimelineOptions{Progress: &p})`；
  `tl.Tween(ui.Tween{From, To, Duration, Ease}, ui.AfterPrevious(d)|ui.WithPrevious(d)|ui.AtTime(d))`。
- 滚动感知：`el.InView()`、`InViewMargin(m...)`、`InViewAmount(0to1)`；内容过渡 `el.ContentTransition(
  ui.ContentTransition{Numeric, Morph})`。
- 无动画：`c.DisableAnimations()`（在变化前调用）、`c.Instant()`。测试用 `tt.Advance(d)` 推进时钟。

## 布局过渡与共享元素

```go
var rowMotion = ui.ElementTransition{Enter: &ui.Motion{Collapse: true}, Exit: &ui.Motion{Collapse: true}}
ui.Row(c.Key(it.ID)).Transition(rowMotion).Children(...)
```
- `el.Transition(ui.ElementTransition{Duration, Ease, Animation, Position, Size, Colors, Shadows,
  Enter/Exit *ui.Motion, ExitDuration, AfterExits, Match, Fly, ScaleContent})`；
  元素需稳定身份（Key/兄弟序）；每个被移动的元素都要挂过渡（同参数=同步移动）。
- `ui.Motion{X, Y, XPercent, YPercent, Opacity, Scale, Rotate, Blur, Collapse, Flip}`——进出场位姿。
- 共享元素：`Match: myType{...}` 同值元素在新旧帧间飞行（Magic Move）；`Fly: true` 内容逐尺寸重排
  飞行（Hero）。`el.ChildExit(motion)` 在父上给离场子元素指定去向。
- `el.Transitioning()` 仍在动。窗口尺寸变化不过渡；List 行不能滑入滑出。

## 变换（2D/3D）

`Translate/TranslatePercent/Scale/Rotate(deg)/Skew/RotateX/RotateY/TranslateZ/Perspective(d)/
TransformOrigin(fx, fy)/Transform(ui.Scaling(x,y).Then(ui.Rotation(15)))/BackfaceHidden()`。
- 指针、输入法、无障碍、裁剪都跟随所画内容；`Draw` 收到未变换坐标。
- `el.Bounds()`（上一帧变换后包围盒）、`LayoutBounds()`、`LocalToWindow/WindowToLocal`。
- 整窗缩放用 `c.SetZoom`，别用变换。

## GPU 着色器（`ui/shader`，Go 语法 = Kage）

```go
var waves = shader.MustCompile(`package main
var Time float
func Fragment(dst vec4, pos vec2) vec4 {
	v := 0.5 + 0.5*sin(pos.x/20+Time)
	return vec4(v*0.2, v*0.4, v, 1)
}`)
// 包级变量编译一次；绘制：
waves.Draw(p, r, shader.Options{Uniforms: map[string]any{"Time": t}})
// 还有 FillPath/StrokePath/Text/Fill(p, r, opts, fn)、waves.Effect(opts) → Element.Effect(...)
```
- `shader.Options{Uniforms, Values, Images [4]*ui.Bitmap, Radius, BackdropBlur, MaxSampleOffset}`；
  快速设值 `u := waves.Uniform("Time"); v := waves.NewValues(); v.Float(u, f)`。
- 内建：`rectSize()/pixelScale()/image0At/Sample/Size/layerAt/backdropAt/discard()/dfdx...`。
  入口 `Fragment(dst, pos) vec4`（返回预乘色）；效果 `ColorEffect/DistortionEffect/LayerEffect`。
- CPU 回退渲染很慢：大着色器用 `if !p.Accelerated() { return }` 守卫。颜色 uniform 非预乘。
