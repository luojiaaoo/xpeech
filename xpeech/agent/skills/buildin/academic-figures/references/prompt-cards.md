# 提示词卡片（Prompt Cards）——对智能体念的完整指令

> **何时读我**：你通过智能体（Claude/其它 coding agent）使用本 skill——把下面的提示词直接粘贴给智能体，或照着改你的数据描述。
> **与 `scenario-cards.md` 的分工**：场景卡=人肉命令卡（你已经知道命令怎么写）；提示词卡=对智能体说的话（让智能体替你写命令、造数据、跑验证）。
> **最短路径**：Ctrl-F 你的场景关键词；每张卡 = 对智能体说 + 应产出 + 验收点。
> **回 INDEX**：SKILL.md「文件索引与 30 秒路由」。

> Prompt cards for agent-driven use. Paste the quoted text to your coding agent as-is,
> replacing the data description with your own. Each card = prompt + expected command + acceptance checks.

## 卡1 三组基线表对比（组学/临床通用）

**对智能体说**：

> 我有一份三组患者基线数据（组别：安慰剂/低剂量/高剂量，各 n≈30，指标：年龄、BMI、收缩压），数据文件 `baseline.csv`（首列组别）。请用 academic-figures 生成期刊投稿级基线特征对比柱状图：自动做组间显著性检验并标星号，黑白云纹区分系列，输出 600dpi PNG + 投稿 PDF，并跑重叠门禁。

**应产出**：`uv run python scripts/gen_figure.py -t bar -d baseline.json -o fig1.png --hatch --pub-ready`
**验收点**：柱高与组间差异如实呈现；显著性星号——bar 是聚合数据不能自动检验，在 baseline.json 里带 `comparisons` 字段并加 `--sig-bracket`（p 值来自你的检验），或原始逐例数据直接改 `-t box --sig-auto`；pub-ready 包含多格式+复跑包。

## 卡2 KM 生存曲线（含风险表）

**对智能体说**：

> 我有逐例生存数据（JSON：`[{time, event, group}…]`，两组，time 单位月）。请画 KM 生存曲线：自动风险表、log-rank P 值、中位生存 95%CI；若有第二事件类型（竞争风险）请自动切 Aalen-Johansen。输出 PDF 并 `--verify`。

**应产出**：`-t km -d survival.json -o fig2.pdf --verify`
**验收点**：风险表时间点合理（默认 5 个）；log-rank P 出现在 stderr 事实行；竞争风险自动告知。

## 卡3 诊断试验 ROC 多模型对比

**对智能体说**：

> 两个标志物（markers A/B）+金标准诊断结果，各 200 例，scores 在 `roc.json`。请画多模型 ROC 对比：DeLong 配对检验 AUC 差异，AUC+95%CI 标注，语义轴标题（1-特异度/灵敏度）。

**应产出**：`-t roc -d roc.json -o fig3.pdf --compare`
**验收点**：每条曲线 AUC 可见；DeLong P 在 stderr；轴标题为诊断语义（非 x/y）。

## 卡4 临床预测模型四件套（ROC+校准+DCA+列线图）

**对智能体说**：

> 我拟合了一个预测 3 期慢性肾病进展的 logistic 模型（训练集 n=500），现在要投稿图：请依次生成 ROC、校准曲线（含 H-L 检验）、决策曲线（含 all/none 参照）、列线图（变量计分→总积分→概率尺）。模型预测概率在 `pred.json`。列线图只可视化我已拟合的模型，不要引擎自己拟合。

**应产出**：四条命令 `-t roc / -t calibration / -t dca / -t nomogram`，均带 `--verify`
**验收点**：校准曲线分箱在 [0,1]；DCA 阈值 (0,1) 开区间；列线图总分好台阶；六件套完整版（+RCS/亚组）见 `scenario-cards.md` 卡4。

## 卡5 Meta 分析三件套（森林+漏斗+径向）

**对智能体说**：

> 我在做 Meta 分析（干预 vs 对照，二分类结局），已提取各研究 effect（logOR）与 se，共 12 项，数据在 `meta.json`。请生成三件套：森林图（合并效应+异质性 I²）、漏斗图（`--egger` 不对称检验）、Galbraith 径向图（离群研究检出）。效应是 log 尺度，轴按 OR 读数。

**应产出**：`-t forest / -t funnel --egger / -t galbraith` 三条命令
**验收点**：森林图 Overall 菱形与 stderr 的 DL 合并一致；漏斗竖线=DL 合并；径向图 stderr 报带外离群名单（与森林图权重交叉核对）。

## 卡6 组学两件套（火山+聚类热图）

**对智能体说**：

> RNA-seq 差异表达结果在 `degs.csv`（列：gene, log2fc, pvalue，约 2000 行）。请画火山图：log2FC 截断 ±1、P<0.05，标注 top 10 基因名；另取前 50 个差异基因在 20 个样本中的表达矩阵画聚类热图（行列均聚类）。

**应产出**：`-t volcano -d degs.csv --top 10` 与 `-t cluster_heatmap -d matrix.json`
**验收点**：火山图三带形态（不粘顶）；热图行列标签自适应不糊死（>1500 行会警告，需先筛）。

## 卡7 多面板期刊组合图（A/B/C 面板）

**对智能体说**：

> 期刊要求 Figure 1 = 三面板组合图（A 柱状基线、B KM 生存、C ROC），每个面板一张子图带加粗 A/B/C 角标。三个面板的 y 轴都是「应答率(%)」，但数值范围差很大，请统一 y 尺度以免审稿人误读。输出 PDF+TIFF。

**应产出**：`-t composite -d fig1.json --pub-ready`，JSON 里 `"panel_labels": true, "share_y": true`
**验收点**：A/B/C 角标按行序自动；同 ylabel 面板 y 轴范围一致（stderr 报 share_y 统一行）；`--doctor` 无共享尺度提示。

## 卡8 期刊主题与配色（投目标期刊）

**对智能体说**：

> 目标期刊是 Lancet。请用 Lancet 配色主题重画我现有的全部图（命令在 `pipeline.yaml`），黑白打印也要能区分系列（斜纹），图型语义轴标题保留。YAML pipeline 一键批量重渲。

**应产出**：`uv run python scripts/gen_figure.py -p pipeline.yaml`（YAML 内 theme: lancet）
**验收点**：主题名从 `--list-themes` 取；同输入同字节确定性；批量逐张 stderr 无 WARNING 升级为 ERROR。

## 卡9 中文标签与 CJK 字体

**对智能体说**：

> 图里轴标签/图例是中文（如「无进展生存期」「接受治疗」）。请用 `--cjk` 正确渲染中文字体，粗体亚组标题要用真粗体（不要描边假粗）。输出 PNG 预览后放大目检确认无豆腐块。

**应产出**：命令带 `--cjk`；引擎自动挂中文字体兜底
**验收点**：stderr 出现「CJK 字体兜底：N 处」=正常；无 Glyph missing 警告；放大看笔画实心。

## 卡10 一键投稿包（审稿人材料）

**对智能体说**：

> 图已定稿，请生成完整投稿材料包：像素级重叠检查+色盲安全模拟+TIFF/PNG/PDF 三格式+复跑交付包（命令+数据副本+审计 JSON）。目标 600dpi。

**应产出**：原命令加 `--pub-ready`（600dpi 输出）
**验收点**：包内 `audit.json` 数字对账与图一致；复跑包 README 可逐字节复现；`--verify` 门禁 0 退出。

## 卡11 批量重渲（数据更新后）

**对智能体说**：

> 数据更新了（同结构新版本 `v2.csv`），请只改数据文件重跑 `pipeline.yaml` 里全部 6 张图，保持版式/主题/输出格式不动；跑完后用 `--audit` 核对每张图的描述统计与数据一致。

**应产出**：`-p pipeline.yaml`（替换数据源路径）+逐张 `--audit`
**验收点**：audit 侧车 JSON 的输入指纹变化、统计重算一致；无图静默漏跑（对照 yaml 清单数）。

## 卡12 选型拿不准（让智能体先建议）

**对智能体说**：

> 我不确定该用什么图：数据是各研究效应值+标准误（12 项 Meta）。请先用 `--suggest` 看推荐，再用 `--explain <类型>` 给我讲字段格式，最后用 `--wizard` 交互式生成完整命令。

**应产出**：`--suggest -d meta.json` → 漏斗(90)+Galbraith(85)；`--explain funnel`；`--wizard`
**验收点**：推荐理由里说明数据形态匹配；explain 给出必填/可选字段；wizard 产出可跑命令。

## 卡13 竞争风险生存数据（审稿人挑「KM 高估」时）

**对智能体说**：

> 我的生存数据里有竞争事件（一部分患者还没等目标事件发生就先死亡了）。数据是 `assets/examples/realworld/hematology_mgus_cif.json`（time=随访月数，event：0=删失 1=进展 2=死亡）。请用 `-t cif` 画竞争风险累计发生率：实线按事件原因分解，叠加 1−KM 虚线对照显示高估幅度；跑完把 stderr 里的末点 CIF 和高估百分点报给我，并提醒我在论文里改用 CIF 口径。

**应产出**：`uv run python scripts/gen_figure.py -t cif -d assets/examples/realworld/hematology_mgus_cif.json -o cif.png --cjk`（投稿 PDF 版改 -o cif.pdf 并加 `--verify`——重叠门禁仅对 PDF 生效）
**验收点**：stderr 报 n/目标事件/竞争事件/删失计数与「高估 +X.X 个百分点」；图例为各原因名（cause_labels）；多臂分组数据智能体会指路 `-t km`。
