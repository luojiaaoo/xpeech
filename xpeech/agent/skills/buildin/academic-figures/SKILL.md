---
name: academic-figures
short_description: 学术数据绘图
description: 根据 JSON、CSV 或 Excel 数据，使用本地 Python 绘图脚本生成森林图、生存曲线、ROC、PRISMA 等静态学术图表，导出 PNG、PDF、SVG 或 TIFF；支持期刊尺寸预设与可选重叠检查。适用于已有研究数据的论文配图。仅接入免费本地版。
---

## 运行约定

从当前加载的 `SKILL.md` 所在目录确定技能路径，不写死全局安装位置。内建技能目录只读；输入、输出及虚拟环境放在用户工作区。下文 `scripts/`、`assets/`、`references/` 相对技能目录解析，运行时使用绝对路径；Python 统一通过 `uv run python` 启动。

首次运行时，在用户工作区执行 `uv pip install --python .venv/bin/python -r "<技能目录>/references/requirements.txt"` 安装依赖（`.venv` 不存在时先用 `uv venv .venv` 创建）。子脚本沿用 uv 选定的解释器。

# Academic Figures — 论文配图一键生成工具

> 📊 **40种图表 · 9套配色（含色盲安全+期刊色板） · 零配置中文 · 600dpi出版级输出 · PDF/TIFF/SVG全支持 · 渲染看门狗+中文诊断**
> 纯本地运行 · 数据不出本机 · Python一条命令搞定 · 内置校验+验证门禁

> ⚡ **三条最常用边界**：投稿 PDF 加 `--verify`（仅 PDF 生效）；`--journal` 会锁定图宽（`--width` 被忽略）；cluster_heatmap >3000 行自动降采样到 2000 行。

**一条命令，输出即验证：**
```bash
uv run python scripts/gen_figure.py -t bar -d data.json -o fig.pdf --theme okabe-ito --verify
# 或一键投稿包（自动加 --verify + 色盲安全主题 + pdf,png 多格式，v3.8）
uv run python scripts/gen_figure.py -t bar -d data.json -o fig --pub-ready
# 退出码含义见下方「数据校验与退出码」；--verify 检出重叠 = exit 2（修复机制，不要交付）
```

## 章节导航
[能力与边界](#能力与边界先看这个免试错) ｜ [快速开始](#快速开始) ｜ [图表类型](#图表类型) ｜ 选图决策树 ｜ [配色方案](#配色方案) ｜ [期刊预设](#期刊投稿预设v20) ｜ [校验与退出码](#数据校验与退出码v20--v28-分级) ｜ [中文支持](#中文支持cjk) ｜ [输出格式](#输出格式) ｜ [常用参数](#常用参数) ｜ [无障碍](#-无障碍--alt-text) ｜ [数据输入](#数据输入) ｜ [相关工具](#相关工具) ｜ [安全与数据](#安全与数据行为声明) ｜ [负向触发](#负向触发不要为本技能触发) ｜ [文件索引](#文件索引与-30-秒路由)

## 能力与边界（先看这个，免试错）

| 能力 | 做到什么 | 做不到/边界 | 详见 |
|------|---------|-------------|------|
| 图表引擎 | 40 种图型一条命令 600dpi；同输入同字节确定性输出 | 无交互式探索（只出静态文件）；无 3D 图 | 「图表类型」表 ↑ |
| 预测模型评价 | calibration 分箱值∈[0,1]；dca 阈值严格升序 (0,1) 开区间、1~6 个模型；nomogram 1~12 变量+1~3 结局尺；rcs ≥5 个已拟合网格点；subgroup 1~12 变量×2~8 层；cif time+event 整数编码（0/1/2+）、单队列 | 无自动分箱；nomogram/rcs 只可视化已拟合模型（不拟合）；subgroup 同样如此——或用 `--stats cox` 一键拟合（逐层 Cox HR+似然比 P 交互，Efron 核，v5.8）；cif 不支持 groups 多臂（用 `-t km`）；rmst 恰好两臂、随访需覆盖 τ（不足则平尾外推并标注） | 「图表类型」表 ↑ |
| Meta 分析 | funnel/galbraith 数据同 schema（studies name/effect/se，效应 log 尺度）；galbraith ≥3 研究，<5 统计效力警告 | 弧线仅对比率尺度（OR/HR/RR）；MD/SMD 自动关 | 「图表类型」表 ↑ |
| 生存分析 | KM 风险表/log-rank/中位生存 CI；cif 竞争风险累计发生率（Aalen-Johansen）+1−KM 高估对照；km 逐例数据 events 含 ≥2 自动切 AJ 模式 | cif 单队列按事件原因分层（多臂用 km）；不做贝叶斯/参数模型 | 「图表类型」表 ↑ |
| 选图 | `--suggest` 自动推荐 / `--wizard` 问答 / `--quick` 自动选型渲染 / `--explain 类型` | 推荐是结构启发式——统计含义请自行确认 | quickstart.md |
| 统计 | KM 风险表/log-rank/中位生存 CI；Cox（Efron+PH）；ROC DeLong；bootstrap CI；Egger | 无贝叶斯方法；无样本量/把握度计算 | advanced.md |
| 期刊预设 | 9 预设，官方栏宽/字号/DPI | `--journal` 锁宽度（`--width` 被忽略，`--height` 有效） | limits.md §六 |
| 验证 | `--verify` 像素重叠门禁（exit 2）+ `audit_pdf.py` 字号门禁 | 默认不开——投稿 PDF 请显式加 `--verify`（或 `--pub-ready`） | 「验证与质量门禁」↓ |
| 组合图 | 多面板 A+B+C、`span` 跨行跨列 | 面板不可嵌套 composite/diagram/upset | composite-layouts.md |
| 数据输入 | JSON 全功能 / CSV-TSV 基础+误差后缀列（v6.0+）/ xlsx（`--sheet`） | errors 对象/双轴/森林等 schema 仍需 JSON（CSV 误差棒走后缀列自动配对） | data-formats.md |
| 数据量 | cluster_heatmap >3000 行自动降采样（stderr 明示） | pca ≤200 特征列；venn 2~4 集合；paired/bland_altman 需等长 | limits.md |
| 中文 | `--cjk` + 数据内中文递归自动检测 | 无显示环境的服务器需装有至少一款 CJK 字体（`setup_env.py` 可检测） | pitfalls.md |
| 看门狗 | 自适应 30~1800s；`--timeout` 调整；内存不足自动降级；瞬时失败（超时/内存）经 `--retries` 自动重试 | `--timeout 0` 禁用自适应预算但保留 7200s 硬保险 | limits.md §七 |
| 参数域 | `--stats` 仅 box/violin（cox 仅 forest）；`--compare` 仅 roc；`--egger` 仅 funnel；`--hatch` 仅 bar 系；`--order/--normalize` 按族 | 域外参数忽略并提示——组合拿不准先跑 `--doctor` | limits.md |
| 退出码 | 0 成功/1 致命校验/2 重叠/3 数据/4 环境/5 超时/6 内存 | exit 0 不会与静默数据问题共存（非有限值一律致命） | 「数据校验与退出码」↓ |

完整矩阵见 `references/limits.md`（超限一律中文报错+修正建议，不出半成品图）；
常见问题速答见 `references/faq.md`。

## ✨ 核心亮点

| 特性 | 说明 |
|------|------|
| 📊 **40种图表** | 见下方「图表类型」全表；v5.6 加 Galbraith 径向图（Meta 异质性），v5.7 加 cif 竞争风险累计发生率（Aalen-Johansen+1−KM 高估对照） |
| 🎨 **9套配色** | **Okabe-Ito色盲安全**（Nature Methods 金标准）、GLM 素雅莫兰迪、NEJM/Lancet/Science 期刊色板等 |
| 📐 **期刊预设** | `--journal nature/lancet/nejm/science/cma/cn-core`：官方栏宽、字号、字体族、600dpi 一键应用 |
| 🧭 **上手向导** | `--wizard` 四问生成完整命令（非交互环境自动打印选图决策树） |
| 🛡️ **数据校验** | 18种类型/别名全覆盖的结构校验；致命错误→exit 1 不落盘，警告→继续运行 |
| ⏱️ **渲染看门狗** | 超时强制中断并给中文建议（exit 5）；内存不足自动降级重试（v2.8） |
| ✅ **验证门禁** | `--verify` 像素级重叠检查（exit 2）；`audit_pdf.py` 字号门禁（期刊最小字号） |
| 🇨🇳 **中文零配置** | 自动检测中文字体（含字典键、补充平面），彻底告别乱码 |
| 📈 **统计标注** | 误差棒、显著性标记(p值星号)、趋势线、置信区间、Cox PH 假设检验提示 |
| 🧩 **组合图** | 多面板A+B+C，每个面板可放任意图表类型，期刊figure标准布局 |
| 📄 **多格式输出** | PNG 600dpi + SVG + PDF + TIFF + EPS；`--multi-format` 一次多格式 |

## 图表类型

| 类别 | 类型 | 命令 | 核心功能 |
|------|------|------|---------|
| 比较 | 柱状图 | `-t bar` | 分组柱状图、误差棒、显著性标记、斜纹填充 |
| 比较 | 水平柱状图 | `-t hbar` | 横向柱状图、比率标注（"4.96x"） |
| 比较 | 分组柱状图 | `-t grouped_bar` | 组×系列双变量分组；`--show-ratio` 组间比率标注 |
| 构成 | 堆叠柱状图 | `-t stacked_bar` | 构成比、百分比标签、总计标注 |
| 分布 | 箱线图 | `-t box` | 箱线图+抖动散点 |
| 分布 | 雨云图 | `-t raincloud` | KDE 半小提琴云+箱线五数+抖动原始点三合一（v4.9）；schema 同 violin |
| 分布 | 山脊图 | `-t ridgeline` | 多组 KDE 密度纵向堆叠部分重叠（joyplot），`overlap` 0~0.95、`sort=median`；一屏看分布逐组移动/变形（v5.0） |
| 比较 | 哑铃图 | `-t dumbbell` | 每行两点 from→to+连接线，按差距降序；`show_gap` Δ 标注；严格两时点（v5.1） |
| 预测模型 | 校准曲线 | `-t calibration` | 分箱预测概率 vs 实际观测率+理想对角线，等比坐标；95%CI 误差棒、Hosmer-Lemeshow p 标注（v5.2） |
| 预测模型 | 决策曲线 | `-t dca` | 净收益 vs 阈值概率；全部干预（理论线按患病率或实测数组）+不干预双参照，1~6 个模型（v5.2） |
| 预测模型 | 列线图 | `-t nomogram` | Points 计分尺→逐变量刻度→Total Points→结局概率尺；可视化你已拟合的模型（引擎不拟合），总分自适应好台阶，行数自适应图高（v5.3） |
| 临床回归 | RCS 样条 | `-t rcs` | 非线性剂量反应（已拟合样条结果可视化）：估计线+95%CI 带、对数比值轴、无效应/参照双虚线、底部暴露 rug（v5.4） |
| 临床回归 | 亚组森林图 | `-t subgroup` | 亚组分析：逐层效应方块+95%CI、每变量 P 交互、Overall 菱形、右侧 est/P 双列；可视化你已拟合的亚组模型（引擎不拟合），或 `--stats cox` 原始数据一键拟合（v5.5 内核，v5.8 一条命令） |
| 生存分析 | RMST 限制平均生存时间 | `-t rmst` | 阶梯 S 曲线+τ 截断阴影 AUC+τ 标线；每臂 RMST(95%CI)+ΔRMST 差值（Greenwood 方差）；τ 自动=最小最大随访之半（`--rmst-tau` 指定）；PH 违背/曲线交叉时的无假设对比（v5.9） |
| 证据综合 | 偏倚风险交通灯图 | `-t rob` | 研究×域交通灯网格（低风险/有一定顾虑/高风险/无信息，中英均可）+摘要占比条；Overall 列只入网格不入摘要（robvis 惯例）；真实示例来自 robvis 包数据（v6.1） |
| 证据综合 | L'Abbé 图 | `-t labbe` | 2×2 四格表（events_t/n_t/events_c/n_c）的处理组 vs 对照组事件率散点；对角线=无效应（RR=1），DL 随机效应合并 RR 参考线，点面积 ∝ 研究权重；骑线/夹缝点自动引线标注（v6.3）；真实示例=Fleiss 1993 阿司匹林 7 项 RCT |
| Meta 分析 | Galbraith 径向图 | `-t galbraith` | 异质性可视化（漏斗图姊妹图）：精度 1/SE×标准化效应、过原点固定效应合并线、±1.96 同质带、带外离群 stderr 逐个报名；OR/HR/RR 效应量弧线（v5.6） |
| 分布 | 小提琴图 | `-t violin` | 密度估计、内置均值/中位线 |
| 配对 | 前后配对图 | `-t paired` | 个体前后连线+配对检验（n≥6 才标注，效力提示） |
| 趋势 | 散点图 | `-t scatter` | 趋势线、相关系数、分组着色、点标签 |
| 趋势 | 折线图 | `-t line` | 多系列、误差带、标记点 |
| 前后 | 斜率图 | `-t slope` | 两时点 pre/post 比较；两端直标名称+数值，最大升/降自动强调（v4.2） |
| 趋势 | 双Y轴图 | `-t dual_axis` | **左柱右线**（左轴柱+右轴虚线，`left_type/right_type` 可互换）；图例顶部预留带不遮数据 |
| 矩阵 | 热力图 | `-t heatmap` | 单元格标注、自定义色阶、colorbar |
| 矩阵 | 聚类热图 | `-t cluster_heatmap` | 层次聚类重排、`--downsample` 降采样出口 |
| 多变量 | PCA 得分图 | `-t pca` | 得分+分组椭圆+Top5 载荷；≤200 特征列 |
| 统计推断 | 森林图 | `-t forest` | CI横线、权重气泡、总效应菱形、I²、事件数；`--stats cox` 多因素回归 |
| 统计推断 | 漏斗图 | `-t funnel` | Meta 发表偏移、DL 合并线；`--egger` 偏倚检验（≥3 研究） |
| 生存分析 | KM生存曲线 | `-t km` | 阶梯函数、删失标记、Log-rank检验、风险表、中位生存；**v3.7 竞争风险**（events≥2 编码自动切 Aalen-Johansen 累计发生率） |
| 生存分析 | 竞争风险累计发生率 | `-t cif` | Aalen-Johansen 按事件原因分解 CIF 阶梯曲线；1−KM 朴素对照虚线直观显示高估幅度；cause_labels 命名各因、percent 百分轴、stderr 事件报告；单队列（多臂用 `-t km`）（v5.7） |
| 诊断试验 | ROC曲线 | `-t roc` | AUC、95%CI、最优截断点、DeLong 多模型对比 |
| 组学 | 火山图 | `-t volcano` | 差异表达 log2FC×-log10(p)；阈值线+Top 基因自动标注（v4.4） |
| 一致性 | Bland-Altman | `-t bland_altman` | 偏倚线+一致性界限（LoA）；两组等长 |
| 集合 | 韦恩图 | `-t venn` | 2~4 集合（4 集合为椭圆布局）；`--area` 面积比例 Euler |
| 集合 | UpSet交集图 | `-t upset` | ≥5 集合交集（韦恩图后继，v4.7）；交集柱+点阵，`sort`/`top_n`/`min_size` |
| 肿瘤学 | 瀑布图 | `-t waterfall` | 每例最佳缓解%降序、RECIST 着色、PR/PD 阈值线+计数事实框（v4.8） |
| 肿瘤学 | 游泳图 | `-t swimmer` | 逐例时长条（按时长降序）、持续中端箭头、里程碑圆点、RECIST 着色（v4.10） |
| 组合 | **组合图** | `-t composite` | 多面板A+B+C，每面板任意图表类型（⚠面板内不可嵌套 composite/diagram/upset）；`panel_labels:true` 自动加粗面板角标（v4.1）；`span:[行,列]` 跨行跨列拼版+占位重叠检测（v4.3） |
| 流程 | **流程图** | `-t diagram` | 架构/流程块、箭头、分组标注，CONSORT式研究设计 |
| 系统综述 | **PRISMA流程图** | `-t prisma` | PRISMA 2020 系统综述流程图；数字自洽自动校验（对不上拒绝出图），`lang=zh` 中文标准版 |

按「类别」列速览；上手路径见 `references/quickstart.md`，完整命令速查见 `references/cheatsheet.md`。

## 选图决策树（精简版；完整版+上手四步见 references/quickstart.md）

```text
分组比较 → bar/box/violin/raincloud（分布三合一；配对→paired；分布逐组移动全景 → ridgeline；逐条目两点差距 → dumbbell）｜均值±误差棒 → bar（JSON errors）
两列相关 → scatter｜时间-事件 → km / forest --stats cox｜诊断 → roc（--compare）
集合交并 → venn（--area；≥5 集合 → upset）｜数值矩阵 → heatmap / cluster_heatmap｜多变量分类 → pca
组学差异表达 → volcano（火山图）｜肿瘤最佳缓解 → waterfall（RECIST 瀑布图）｜逐例时长 → swimmer（游泳图）｜Meta 汇总 → forest + funnel（--egger）+ galbraith（异质性/离群）三件套｜有竞争事件的生存数据 → cif（累计发生率）｜一致性 → bland_altman｜流程 → diagram/prisma
预测模型评价 → roc + calibration + dca + nomogram + rcs + subgroup（P 交互）六件套；PH 违背/曲线交叉 → rmst（临床 τ 下 ΔRMST）；系统综述 → prisma+rob（偏倚风险）

## 谁拟合、谁可视化（v6.1——手填统计量前先读这节）

| 层级 | 图型 | 你提供什么 |
|---|---|---|
| **引擎拟合+检验** | km（log-rank/中位 CI）、cif（Aalen-Johansen）、rmst（ΔRMST CI）、`forest --stats cox`、`subgroup --stats cox`（一键逐层 Cox+似然比 P 交互）、bar --stats auto/multi、roc（AUC+CI）、calibration（H-L）、dca、funnel --egger、galbraith（合并/I²） | 原始逐例数据 |
| **描述性** | 柱状族、箱线/小提琴/雨云/山脊、游泳图、瀑布图、哑铃图、斜率图、热图、火山图*、rob | 数值本身（无推断） |
| **可视化你已拟合的输出** | nomogram、rcs、subgroup（不带 --stats cox 的 JSON 路径）、模型输出的 composite 组合 | 你自己模型的估计值 |

*火山图吃预计算 log2FC/p。第一层每行一条命令即可——除非需要定制模型，**不要手算后再喂回引擎**。
多面板 → composite
```

拿不准就跑 `--suggest -d 数据文件`（自动分析并推荐），或 `--wizard`（问答式生成命令）。
**首次使用？** 完整决策树、上手四步、Python 内调用与常见第一坑见
**`references/quickstart.md`**（独立快速入门指南）。

## 快速开始

```bash
# 0️⃣ 首次使用：一键环境准备（装依赖/检测中文字体/清理字体缓存/自检）
uv run python scripts/setup_env.py

# 0️⃣ 上手三件套：选图向导 / 交互演示 / 某图型用法
uv run python scripts/gen_figure.py --wizard
uv run python scripts/gen_figure.py --quick -d 数据文件   # v3.1 一键出图：自动选型直接渲染
uv run python scripts/gen_figure.py -t line -d 折线数据.json -o fig.png --direct-label   # v3.4 线末端直接标注系列名（glm-brand/nature-clean 风格下 line 默认开启）
uv run python scripts/gen_figure.py --demo --cjk
uv run python scripts/gen_figure.py --explain bar
# 0️⃣ Python 内调用引擎：见 references/python-api.md（subprocess 推荐；import 嵌入需自行 Agg）

# 1️⃣ 柱状图（默认 glm 素雅配色）
uv run python scripts/gen_figure.py -t bar -d data.json -o figure.png \
  --title "图2 主标题 / Subtitle" --ylabel "准确率 Accuracy (%)"

# Meta分析森林图（PDF输出）
uv run python scripts/gen_figure.py -t forest -d forest.json -o forest.pdf --theme okabe-ito

# v2.7：原始逐例生存数据 → 自动多因素 Cox 回归 → HR 森林图（含 PH 假设检验）
uv run python scripts/gen_figure.py -t forest --data patient.csv --stats cox \
  --cox-time time --cox-event event --cox-cols "age,sex,treat"

# Kaplan-Meier生存曲线 + Log-rank检验
uv run python scripts/gen_figure.py -t km -d survival.json -o km.png --theme okabe-ito \
  --title "图3 Kaplan-Meier生存曲线" --xlabel "时间 (月)" --ylabel "生存概率"

# ROC曲线 + AUC
uv run python scripts/gen_figure.py -t roc -d roc.json -o roc.png --theme okabe-ito

# 多面板组合图（Panel A+B+C，期刊figure布局）
uv run python scripts/gen_figure.py -t composite -d composite.json -o figure4.png --theme okabe-ito

# 热力图（自定义色阶 + 中文）
uv run python scripts/gen_figure.py -t heatmap -d data.json -o heatmap.png --cjk \
  --cmap RdBu_r --vmin -20 --vmax 45

# v2.7：画幅比例预设（16:9 演示 / 9:16 手机竖版）；v2.8：大数据聚类热图降采样
uv run python scripts/gen_figure.py -t bar --data d.json --size 16:9
uv run python scripts/gen_figure.py -t cluster_heatmap --data big.json --downsample 2000 --timeout 600
```

## 配色方案

**默认 = `glm`**（素雅莫兰迪，色盲安全）。投稿首选 `--theme okabe-ito`（Nature Methods 金标准；
红绿配色是常见退稿原因）。色板：`glm`/`okabe-ito`/`cool`（均色盲安全）+ `nature`/`lancet`/`nejm`/
`science`/`conservative`/`classic`。`--style glm-hatch` 加黑色斜纹（打印友好）；`--style nature-clean`
= 去顶右框线+无网格+无框图例；`--journal nejm|lancet|science|nature` 未显式 `--theme` 时自动联动。
`--list-themes` 看色卡；`--theme-swatch glm -o swatch.png` 出色板图；`--alternate` 黄蓝逐柱交替。
别名 `okabe`→`okabe-ito`，大小写不敏感前缀匹配。完整指南：`references/color-guide.md`。

## 期刊投稿预设（v2.0）

`--journal <名>` + `--column single|double` 自动应用期刊官方栏宽/字号/字体族/DPI——
9 预设：`nature` `lancet` `science` `cell` `nejm` `jama` `ieee` + **`cma`**(中华医学会)
+ **`cn-core`**(中文核心，均自动 CJK)。`--journal` 锁宽度（`--width` 被忽略，`--height` 有效）；
`nejm/lancet/science/nature` 未显式 `--theme` 时自动联动同款配色。参数唯一权威源：
`scripts/journal/*.json`；**九刊完整预设表**：`references/limits.md` §六。

## 数据校验与退出码（v2.0 / v2.8 分级）

每次运行都经过 `validate_data(data, chart_type)` 结构校验 + 渲染全程看门狗。退出码语义：

| 退出码 | 含义 | 典型场景 |
|--------|------|---------|
| `0` | 成功（--verify 时=无真实重叠） | 正常交付 |
| `1` | 参数/用法或数据校验致命错（不落盘） | 缺参数、系列长度不匹配、km 长表 |
| `2` | `--verify` 检出文字重叠；或批量/流水线有失败项 | 修复机制本身，不要交付 |
| `3` | 数据或格式错误（v2.8） | JSON 语法、编码、字段缺失、KeyError |
| `4` | 环境或依赖错误（v2.8） | 缺库（ImportError）、权限、磁盘 |
| `5` | 渲染看门狗超时（v2.8） | 数据过大——按建议瘦身或 `--timeout` 调大 |
| `6` | 内存护栏拒绝（v2.8） | 自动降级重试仍内存不足 |

致命错误 stderr 前缀 `ERROR:`（中文原因+改法），警告 `WARNING:`（继续运行）。极端未知异常也输出「中文诊断+原文+针对性建议」，`AF_DEBUG=1` 取完整技术细节。

## 验证与质量门禁（v2.0）

1. **`--verify`**（内联，PDF 输出时）：像素级文字重叠检测，检出即 exit 2——修复机制本身，不要逐图特判。
2. **`audit_pdf.py` 字号门禁**：`uv run python scripts/audit_pdf.py figure.pdf --min-size 5 --fail-below`。
3. **`verify_overlap_pixel.py`**：交付每个 PDF 前必跑，输出「真实重叠=K」必须 K=0。

防重叠机制已内建（密集标签自动 45° 旋转、`_ensure_ylabel_clear()` 自动避让）；PyMuPDF bbox 高估假阳性原理与验证器三级流程见 `references/pitfalls.md`。质量门禁回归：`uv run python tests/run_tests.py` 全量必须过。

## 补充图例（v2.0）

禁止图内图例的期刊（如 Nature）：`uv run python scripts/gen_legend.py -d data.json -t "治疗应答" -f 1 -o legend.txt`——用与出图相同的数据 JSON 生成期刊格式图例，文字与系列/颜色永远一致。

## 中文支持（CJK）

传 `--cjk` 自动检测并加载系统中文字体（优先级：Noto Sans CJK → PingFang → Microsoft YaHei → WQY → AR PL → Droid）；自定义字体 `--cjk-font /path/to/font.ttf`。数据含中文时**递归自动检测**（含字典键、组合局面板、补充平面生僻字），无需显式传参。字形缺口（如上标 ⁹）与流程图块配色等陷阱清单见 `references/pitfalls.md`。

## 输出格式

| 格式 | 扩展名 | DPI | 适用场景 |
|------|--------|-----|----------|
| PNG | `.png` | 600（默认） | 通用、演示文稿 |
| SVG | `.svg` | 矢量 | Web、可编辑图形 |
| **PDF** | `.pdf` | 矢量 | **期刊投稿首选** |
| TIFF | `.tiff` | 600（照片 `--dpi 300`） | Nature/Lancet照片要求 |
| EPS | `.eps` | 矢量 | 传统期刊要求 |

> 严格期刊线稿可用 `--dpi 1000`；`--multi-format tiff,png,pdf` 一次出多格式。

## 常用参数

高频参数（完整速查表见 `references/cheatsheet.md`）：

| 参数 | 用途 |
|------|------|
| `--journal` / `--column` | 期刊预设 / 栏位（⚠ 锁定宽度，见上） |
| `--stats auto\|cox\|bootstrap` | box/violin 显著性标注 / forest 多因素回归 |
| `--order "C,A,B" \| auto` | 类别顺序重排（bar/box/violin/line；显式列表须覆盖全部标签；auto=按值/中位数降序） |
| `--normalize baseline\|pct100` | 归一化：各系列÷对照均值（对照=1）/ 自身首点=100（bar 系/line；误差棒同步缩放） |
| `--doctor` | 渲染前参数/环境体检报告（省略 -o 只体检不渲染，exit 1=有发现） |
| `--cjk` | 中文字体自动检测 |
| `--verify` | PDF 像素级重叠验证（exit 2） |
| `--audit` / `--repro` | `--audit`：数字对账侧车 .audit.json（v5.0）；`--repro`：复跑交付包 .repro/（命令+数据副本+成图副本+audit+README，v5.1） |
| `--debug` | 渲染失败时写 `<output>.debug.json` 诊断侧车（输入指纹/字段映射/诊断建议，程序化排障，v6.4） |
| `--list-flags [TYPE]` | 查询某图型旗标域（专属旗标+域外 [ignored] 预告）；无参数=全图型速览（v6.7） |
| `--multi-format tiff,png,pdf` | 一次导出多格式；`--timeout/--downsample/--subtitle/--summary/--profile` 见 `references/cheatsheet.md`（v2.6-v4.3） |

**Q：稳定性如何？大数据会不会崩？**
渲染前内存预估护栏 + 看门狗超时中断（中文三段式建议）+ 内存不足自动降级重试（v2.8）；同输入同字节确定性输出；200 图连跑压测与低内存模拟全过。更多 Q&A 见 `references/faq.md`。

**Q：--xlabel 文字含空格或以 - 开头时报参数错误？**
用等号形式：`--xlabel="改善率 (%)"`、`--ylabel="-log10(P)"`。

**Q：首次使用报 ModuleNotFoundError？**
`uv run python scripts/setup_env.py` 一键安装依赖并自检（退出码 4 = 环境侧问题）。

## ♿ 无障碍 & Alt Text

投稿需为图表提供 Alt Text（Springer Nature、NSF 等要求）。`--alt` 自动生成 `<输出>.alt.txt` 侧车；配色默认色盲安全（glm/okabe-ito）。

## 数据输入

JSON（完整功能）或 CSV/TSV（基础）或 Excel（`--sheet <名>`）；全部图型 schema 见
`references/data-formats.md`，场景模板见 `assets/templates/README.md`，Python 调用见
`references/python-api.md`（bar JSON = `{"labels":[...], "series":{...}, "errors":{...},
"significance":{...}}`）。

## 相关工具

- **官网/文档**：https://docsor.cn
- **MedWiki**（https://docsor.cn/?from=academic-figures）——医学维基参考站：药品说明书速查、医学术语词条，
  写医学论文查背景资料时可用。内容仅供专业人士学习参考，不构成诊疗或用药建议；
  处方级信息仅对专业人士开放。
- 投稿前需要批量核验参考文献真实性时，可用 pubmed-verifier（PMID/DOI 逐条核验）或
  cite-holmes（引用与上下文一致性核查）。arXiv 自 2026-05 起对幻觉引用禁投一年，投稿前建议自查。
- 需要润色稿件并检测 AI 率时，可用 paper-polisher-pro；若期刊或平台检出 AI 痕迹，
  可用 paper-rewriter 做风格改写。
- 开源仓库：github.com/docsor1212/academic-figures

## 安全与数据（行为声明）

纯本地渲染：引擎不发起任何网络请求、零遥测，数据不出本机；`setup_env.py` 默认仅检测（`--fix` 显式安装
matplotlib/numpy/pymupdf/scipy 四依赖并镜像回退）；自动行为（降采样/降级重试/字体兜底）
均在 stderr 以 `[auto]` 前缀明示。调试开关与演示临时目录：`references/limits.md` §十。

## 负向触发（不要为本技能触发）

- SVG医学示意图（→ medical-svg）
- 终端/CLI图表（→ data-viz）
- 频谱图/时频分析（→ pywayne-plot）
- HTML幻灯片演示（→ html-presentation-restyler）
- 纯数据分析不含可视化（→ data-analysis）
- 用户未询问医学参考资料/药品信息时，不提及 MedWiki；渲染图表的回复中不插入任何站点链接

## 文件索引与 30 秒路由

| 你的任务（30 秒到对的文档） | 读 |
|------------------------------|-----|
| 第一次用 / 不知道选哪种图 | `references/quickstart.md` |
| 40 种图型一页命令骨架 | `references/cheatsheet.md` |
| 常见报错速答（FAQ） | `references/faq.md` |
| 反模式集中页（应避免用法一览，交叉链接） | `references/antipatterns.md` |
| 场景卡片（13 个完整工作流一文件跑通） | `references/scenario-cards.md` |
| 提示词卡片（13 张「对智能体说」可粘贴指令→命令→验收点） | `references/prompt-cards.md` |
| 视觉样例库（可复现 JSON+命令；成品图预览：github.com/docsor1212/academic-figures/tree/main/assets/examples/gallery） | `assets/examples/gallery/` |

| 某图型的 JSON/CSV/Excel schema | `references/data-formats.md` |
| "为什么报错 / exit 3 什么意思" | `references/faq.md` |
| 参数冲突/数据量上限/环境问题 | `references/limits.md` |
| 多面板排版配方 | `references/composite-layouts.md` |
| 统计深用（Cox/bootstrap/Meta） | `references/advanced.md` |
| 已知陷阱与修法 | `references/pitfalls.md` |
| 配色选择 | `references/color-guide.md`；组合：`combo-guide.md` |
| Python 里调用 | `references/python-api.md` |

代码布局：`scripts/gen_figure.py`（主生成器，40 图型）+ `gen_legend.py`（期刊图例）+
`audit_pdf.py`（字号门禁）+ `verify_overlap_pixel.py`（重叠验证）+ `setup_env.py` +
`journal/*.json`（9 预设：nature/nejm/lancet/jama/science/cell/cma/cn-core/ieee）；
`assets/templates/`（40 个，各带 `_command`）、`assets/examples/`（含
`realworld/` 真实场景）。其余 references：
`clinical-lab-trends.md`（临床检验趋势图）、`reverse-engineering-colors.md`（参考图取色）、
`v1.5-upgrade-analysis.md`（历史升级档案）。
