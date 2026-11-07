# MyGo 原生 UI 控件目录（速查）

全部控件都在 `github.com/egoist/mygo/ui`（标注包名的除外）。`c` 是 `*ui.Context`。
改值控件拿指针（构造函数返回前就写回）；事件方法 `.Clicked()/.Changed()/.Submitted()`
询问上次帧以来发生了什么。通用链式样式（`Padding/Radius/Background/Width/Label/Tooltip/
Disabled` 等）对所有控件返回的元素都有效，此处不再重复。

## 动作

### Button / Link
```go
ui.Button(c, label)          // 按钮；ui.PrimaryButton(c, label) 强调色主按钮
  .Clicked() / .OnClick(fn)  // 点击/Enter/Space；回调在构建完成后执行
  .Children(func(){...})     // label 传 "" 时放图标等内容
  .Label("名")               // 无障碍/测试查找名
```
- `ui.Link(c, label, url)`：打开 URL；无 scheme 的路径（`/notes/42`）在 router 里当路由跳转；
  url 传 `""` 就是应用内动作文本，`.Clicked()` 判断。
- `ui.ButtonBase`：无外观按钮基座（焦点行，报告 Clicked）。
- 聚焦的按钮独占 Enter/Space，所以默认按钮的 `c.Shortcut(0, ui.KeyEnter)` 只在焦点在别处时触发。

### Menu button 与 Context menu
```go
ui.MenuButton(c, "Sort by", func(m *ui.Menu) {
	if m.Item(by).Checked(app.sort == by).Chosen() { app.sort = by }
})
```
- `ui.MenuButton(c, label, build)`：按钮打开**系统菜单**；`ui.PopoverMenuButton(...)` 同参，
  菜单画在窗口内（主题外观）。`ui.PopoverMenuButtonOptions{At, Self ui.Anchor, Open *bool}` 控制位置/开合。
- `*ui.Menu` 菜单项 API：`m.Item(label)` + `.Checked(on)/.Radio(on)/.Disabled(d)/.Shortcut(mods,key)/
  .ID(id)/.Tooltip(s)/.Icon(icon)/.Subtitle(s)/.Danger()/.Highlighted()/.KeepOpen()`，`.Chosen()` 报告选择
  （在选择后的那一帧报告）；`m.Separator()`、`m.Submenu(label, build)`、`m.EditItems()`（补 Copy/Select All）。
- 元素右键菜单：`el.ContextMenu(build)`（系统）/ `el.PopoverMenu(build)`（窗口内）。
- `c.SetMenuPresentation(ui.PopoverMenus | ui.SystemMenus)` 统一全窗口菜单画法。
- 无外观：`ui.MenuBase`/`ui.ContextMenuBase`（`Trigger`、`Popup(func(panel ui.Element))`、`Item/CheckItem/
  RadioItem/SubmenuItem`）。
- 测试：`tt.RightClick`、`tt.Menu`、`tt.ChooseMenuItem("Move to", "Archive")`。

### Toggle / Segmented / Toolbar
```go
ui.Toggle(c, &app.bold, "B")                 // 按下为 true 的开关钮
ui.ToggleGroup(c, func(){ ... }).Label("Style")  // 成组，一个 Tab 停靠点
ui.Segmented(c, &app.view, "List", "Grid").Label("View")  // *int 选中下标
ui.Toolbar(c, func(){ ... }).Label("Format") // 工具栏：悬停显形、溢出收进菜单
```
- `Changed()` 报告切换；无外观基座 `ToggleBase`、`SegmentedBase(c, &idx, n)`（`Track`/`Segment(i)`）。
- Toolbar 里 `ui.Spacer(c)` 推右侧；`ui.BackButton(c, router)`/`ui.ForwardButton(c, router)`。

## 选择

```go
ui.Checkbox(c, &app.mail, "Mail")            // 勾选框（点 label 也切换）
ui.CheckboxGroup(c, "Notifications", func(){ ... })   // 父勾选 + 缩进子项（三态）
ui.CheckboxState(c, &state, label)           // 三态：ui.CheckOff/CheckOn/CheckMixed
ui.Switch(c, &app.wifi).Label("Wi-Fi")       // 开关（立即生效的设置）
ui.RadioGroup(c, func(){ ui.Radio(c, &app.size, v, label) ... }).Label("Size")  // *T 任意可比类型
ui.Select(c, &app.plan, []string{"Free","Pro"}).Label("Plan")  // 下拉（字符串）
```
- 全部 `.Changed()` 报告变化。`ui.SelectBase(c.Key("k"), &val)` 支持任意 `T`：`Trigger`、
  `Popup(func(panel ui.Element))`、`sel.Item(v)`（`.Highlighted()`）、`SetOpen(bool)`；
  `SelectBaseOpen` 自带 `open *bool`（可做动画弹层）。

## 数值

```go
ui.Slider(c, &app.volume, 0, 100).Label("Volume")       // *float64
ui.StepSlider(c, &app.quality, 0, 100, 25)              // 带刻度吸附
ui.RangeSlider(c, &app.lo, &app.hi, 0, 1000, 10)        // 双滑钮（step 0 = 自由）
ui.Stepper(c, &app.copies, 1, 99, 1)                    // 加减箭头
ui.Rating(c, &app.stars, 5).Label("Rating")             // 星级
ui.NumberInput(c, &app.copies, 1, 99, 1)                // 可输入数字 + 步进
ui.DecimalInput(c, &app.amount, "0", "", "0.01")        // 金额精确十进制（字符串，永不浮点舍入）
```
- `.Changed()` 报告；`SliderBase`（`.Step/.Vertical/.Draw`）无外观。
- 金额用 `DecimalInput`（`"1.005"` 在 step `0.01` 下标错不静默改写）；RTL 下滑条低值在右。

## 文本输入

```go
ui.TextInput(c, &app.name)        // 单行；ui.TextArea(c, &app.draft) 多行
  .Placeholder("...").Label("...").Password().AutoFocus().ReadOnly(true)
  .MaxLength(n).Disabled(true).Lines(1, 8)   // TextArea 自适应行数
  .Changed() / .Submitted()                   // Enter（单行）/ Composing()（IME）
```
- `ui.SearchField(c, &app.query)`：搜索框（放大镜 + 清除 + Escape 清空），`.Changed()` 边输边筛。
- `ui.Combobox(c, &app.font, options)`：输入过滤的下拉（选中即定值）；`ComboboxBase` 无外观。
- `ui.Autocomplete(c, &app.city, cities)`：保留自由文本的建议。
- `ui.TokenField(c, &app.tags, allTags)`：chips 标签输入（Enter/逗号添加，空 Backspace 删末尾）。
- `ui.EditableText(c, &app.title)`：原地重命名（双击/Enter 编辑，Escape 取消）。
- `ui.FindBar(c, &app.finding, &app.query, len(matches), &app.match)`：查找条（计数、上一个/下一个）。
- 选择 API：`.TextSelection() (start, end)`、`.SetTextSelection(a,b)`、`.TextAnchorCaret()`。
- 换 key（`c.Key(...)`）= 新的编辑会话（清空选择/撤销历史）。

## 代码与文档（独立包）

```go
// 代码编辑器 ui/editor
buf := editor.NewBuffer(source); buf.SetLanguage(editor.LanguageFor("main.go"))
app.code = &editor.State{Buffer: buf, WrapGuides: []int{100}}
editor.View(c, app.code).Grow(1).FillWidth()
app.code.Run(editor.SortLines)   // 命令；Bindings 绑键；OnCompletion/OnHover/SetDiagnostics

// 差异查看器 ui/editor
app.diff = &editor.DiffState{Old: before, New: after, Editable: true}
editor.DiffView(c, app.diff).Grow(1)
app.diff.Revert(i) / Accept(i) / Hunks() / Stats()

// Markdown ui/markdown（CommonMark+GFM、表格、任务、脚注、数学）
app.md := markdown.Options{}   // 跨帧共享一份
markdown.View(c, src, &app.md)         // 成品文档
markdown.Stream(c, src, &app.md)       // 流式追加中的文本
markdown.Revealed(el)                  // 刚构建的元素文字是否已全部显现

// 富文本编辑器 ui/richtext
app.doc = richtext.NewDocument(richtext.NewText("Hello"))
richtext.Editor(c, app.doc).Fill().Changed()   // 用户编辑才报 Changed
app.doc.ToggleBold(); app.doc.SetLink(url); app.doc.Text().Markdown()

// 图表 ui/chart（无外观：坐标轴 + 元素标记）
x := chart.NewBand(months...).Padding(0.3, 0.15)
y := chart.NewLinear(0, hi).Nice(4)
chart.Plot(c, chart.Axes{Left: y.Ticks(4), Bottom: x.Ticks()}, func(area ui.Element) {
	chart.Rect(c, x.Band(i), y.Span(0, v)).Radius(4).Background(t.Accent)
}).Label("Sales").Height(200)
```
- 线/面积在 `area.Draw` 里用 `chart.Line/Area(...)` 生成路径后 `p.StrokePath/p.FillPath`；
  饼图 `chart.Pie/Sector`；选点 `chart.Select(c, area, &idx, xs, describe)`。
- 编辑器/文档状态（Buffer、State、Document、Options）都放在应用状态里跨帧持有；
  协程改 buffer 必须经 `Window.Update`。

## 日期与颜色

```go
ui.Calendar(c, &app.day).Label("Due date")    // *time.Time 月历；.Browse() 浏览不选
ui.DateInput(c, &app.birthday)                // 日期字段（点开日历）
ui.TimeInput(c, &app.alarm).Label("Alarm")    // 时刻字段（分段步进/输入）
ui.ColorWell(c, &app.tint)                    // 色井 + 取色器弹层
ui.ColorPicker(c, &app.tint)                  // 取色器本体（自定义面板用）
```
- 全部 `.Changed()`；Date/Time 输入与同一个 `*time.Time` 配对保留时/日。

## 集合（虚拟化，行只在可见时构建）

```go
ui.List(c, &app.listState, n, func(i int) { ... }).Grow(1)
```
- `ui.ListState`（建前设置）：`Key func(i) any`、`Label func(i) string`、`Selected *T`、
  `Selection *ui.Selection[K]`（多选）、`FollowEnd`（聊天贴底）、`FollowAnimation`、
  `Header func(i) string`（分组钉住标题）、`Reorder func(rows []int, to int)`、`DragRows`、`KeepOffset`、`Handle`。
- 事件：`Changed()`、`Submitted()`（双击/Enter）、`ScrollTo(row, align)`、`ScrollIntoView(row)`、
  `Visible() (first,last)`；行元素 `.RowDoubleClicked()`。
- `ui.Table(c, &state, []ui.TableColumn{{Title:"Name"},{Title:"Size",Width:90,Align:ui.End}}, n, cell)`：
  可排序/缩放/拖列（`Sort *ui.SortOrder`、`Columns` 持久化、`HeaderMenu`/`CellMenu`）；**排序自己做**。
- `ui.Tree(c, build)` + `ui.TreeItem(c, label, open *bool, children)`（小树）；
  `ui.Outline(c, &state, roots, children, row)`（大树，`OutlineState` 含 `Open` 展开集）、
  `ui.OutlineTable`（树表）。树项 `.Clicked()/.Selected(bool)`。
- `ui.GridView(c, &app.photos, n, itemW, itemH, func(i int){...})`：网格（相册）。

## 容器

```go
ui.Scroll(c).Grow(1).Children(func(){ ... })      // 还有 ScrollHorizontal/ScrollBoth
ui.Grid(c).ColumnTracks(ui.Fr(1).Min(240), ui.Fixed(320)).GapX(16).Children(func(){ ... })
ui.Split(c, &app.sidebarW, first, second).SplitReset(220)   // SplitVertical / SplitPercent
ui.Collapsible(c, "Advanced", &app.open, content)
ui.Accordion(c, func(){ ui.AccordionItem(c, "General", &app.g, content) ... })
ui.Field(c, "Email", func(){ ui.TextInput(c, &app.email) }).Description("...").Error(errStr)
ui.Form(c, build)     // macOS 风格标签右对齐列
ui.Fieldset(c, "组", build).Disabled(false)
```
- 滚动：`ScrollState`（`TrackScroll(&st)`：X/Y/MaxX/MaxY、`ScrollTo(x, y, ui.Smooth)`）、
  `el.ScrollIntoView()`、`.Sticky().Top(0)`、`.ScrollSnap(ui.SnapMandatory)` + `.SnapAlign(...)`。
- 网格轨道：`ui.Fr(n)`、`ui.Fixed(w)`、`ui.FitContent()`；子项 `.ColumnSpan(n)/.RowSpan(n)`。
  很多项用 GridView，Grid 构建全部子项。
- `Field` 命名第一个可聚焦控件；控件侧 `.LabelledBy(el)` 自定义 caption。

## 导航

```go
ui.Tabs(c, &app.tab, "General", "Appearance")   // 页内容自己在下方按 *int 构建
ui.Sidebar(c, &app.mailbox, func(){
	ui.SidebarSection(c, "Mailboxes", &app.shown, func(){
		ui.SidebarItem(c, "inbox", inboxSVG, "Inbox").Children(func(){ ui.Badge(c, "3") })
	})
}).Width(220)
ui.Breadcrumbs(c, path, &app.chosen).Changed()
ui.BackButton(c, app.router); ui.ForwardButton(c, app.router)
```
- Sidebar 的 `chosen *string` 是条目 ID；配 router 时 ID 即路径，`.OnChange` 里 `router.Push`。

## 对话框与消息

```go
ui.Modal(c, &app.open, func(){ ... })      // 压暗对话框；外点/Escape 关；.Dismissed() 报告关闭
switch ui.AlertDialog(c, &app.asking, "删除？", "不可撤销", "Cancel", "Delete") {
case 1: app.delete()                        // 返回按钮下标；ui.AlertDismissed = Escape
}
ui.Popover(c, anchorEl, &app.menu, func(){ ... })   // 锚点下方弹层
el.Tooltip("说明文字")                       // 悬停提示（兼无障碍描述）
c.Toast("Saved")                            // 底部短消息（4 秒）
c.ToastAction("已删除", "Undo", func(){ ... })     // 带动作
id := c.AddToast(ui.Toast{Title: ..., Type: "error", Timeout: -1, Action: ..., OnAction: ...})
```
- `ui.PopoverBase/DialogBase/TooltipBase/ToastViewportBase+ToastBase` 为无外观版本。
- 弹层定位细节（AttachTo、Escape 层级、PressedOutside）见 guides.md「浮层」。

## 状态

```go
ui.Progress(c, v0to1).Label("Uploading")   // 负值 = 不定长滑块
ui.Spinner(c).Label("Loading").Size(32, 32)
ui.Meter(c, value, min, max, &ui.MeterLevels{Warning: 0.8*max, Critical: 0.95*max})
ui.Badge(c, "New").Background(t.Accent).TextColor(t.AccentText)
```

## 图像

```go
var logo, _ = ui.DecodeBitmap(logoPNG)   // //go:embed；建一次，勿在 view 里
ui.Image(c, logo).Size(64, 64).Fit(ui.Contain).Radius(12)   // 也可传 SVG（ui.MustParseSVG）
ui.Icon(c, saveSVG)                      // 文字色、字号高的 SVG 图标
ui.Avatar(c, user.Name, user.Photo)      // 圆头像或首字母
```
- `ui.DecodeBitmap(png/jpeg/gif/webp/bmp)`、`ui.NewBitmap(image.Image)`、`bmp.Update(*image.RGBA)`、
  `bmp.Pixelated()`；`ui.DecodeImage(data)` 通吃、`ui.ImageSize(pic)`。
- `Fit(ui.Contain|ui.Cover|ui.FillBox|ui.ScaleDown|ui.NaturalSize)`、`Grayscale()`、`ObjectPosition(x,y)`。
- 视频：`ui.NewVideoPlayer(ui.NewGIFDecoder(data))` + `ui.Video(c, player)`（无声；可见才播；
  ReduceMotion 下停在首帧）。`player.Play/Pause/SetLoop/Close`（主线程）。
- 图标不拉伸；纯图标按钮用 `.Label(...)` 命名按钮本身。
