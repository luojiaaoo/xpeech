# Changelog / 版本历史

> **何时读我**：查某版本改了什么。**版本索引一表总览；最近三版展开详情；完整历史见 `changelog-full.md`**（v5.6.0 拆分：降低单文件篇幅，内容零删减）。

> **回 INDEX**：SKILL.md「版本历史」。


## 版本索引 Version Index


| 版本 | 日期 | 首条要点 |
|---|---|---|
| v6.9.0 | 2026-11-02 | 新功能 `--sig-bracket` 组间显著性括号（comparisons 任意比较对+分层防重叠+bar/grouped_bar/box/violin 四图型）+文档准确性四处收口（MCP 工具清单/README 语言/uvx 本地路线/版本历史当前标记） |
| **v7.2.1（当前）** | 2026-11-05 | 发布文案清净化重发（评测导向表述未过平台安全审核→中性化+机械锁；代码零差异） |
| v7.2.0 | 2026-11-05 | 一致性扫描器 v2：引擎源码运行时文案入扫描面（CHART_NOTES 等 --explain 输出字典——accuracy 4.5 抓到 bar 文案 CSV 旧口径）+MCP 安装本地优先反转+SKILL 双语/README 视觉样例库 GitHub 直链 |
| v7.1.0 | 2026-11-04 | 新工具 `check_consistency.py`（文档与能力注册表自动一致性检查：全部命令载体×旗标域+--verify/PDF 规则+CSV 口径残留——118 条载体对齐，15 处历史混用清零）+域表真值化（--stats 补 raincloud/subgroup）+ADR-0001 状态如实化 |
| v7.0.0 | 2026-11-03 | major：新功能 `--sig-auto` 全自动组间显著性（两两自动检验 Shapiro→Welch t / Mann-Whitney U+分层 bracket）+文档可信度大扫除（changelog 补 6.9.0/7.0.0 双节+CSV 误差表述全面清零含 quickstart/SKILL 决策树+README 分发形态如实化+openpyxl 入正式依赖+MCP 包声明如实化） |
| v6.8.0 | 2026-10-31 | MCP 新工具 `query_capability`（图型+旗标域动态查询，能力面与引擎注册表零漂移）+CHARTS 内置清单快照锁+CSV 误差能力文档真相化（FAQ/data-formats/引擎 hint 旧表述纠正）+守卫扩容（mcp/pyproject 入扫描+英文修饰容忍）+glm-brand 单系列斜纹补齐+show-values 精度自适应 |
| v6.7.0 | 2026-10-31 | 新功能 `--list-flags [TYPE]` 旗标域查询（专属+域外 [ignored] 预告；注册表与 argparse/图型注册表互锁防漂移）+ 官方点名收口（37 残留×5 含锚链接/gallery JSON 随包索引指针存活/Python 下限统一 3.8/changelog 表格解挤压）+ README 测试徽章动态互锁 |
| v6.6.0 | 2026-10-30 | 一致性清剿第二轮（评测 accuracy/usability 4.6 点名全修：MCP server.py 头注释与提交文档 23→40、SKILL 索引与代码布局 37→40、quickstart 36→40；limits「静默忽略」纠为实际 [ignored] 行为+机械锁）+计数守卫扩至 14 份活文档含裸 types 口径 |
| v6.5.0 | 2026-10-30 | 文档一致性大扫除（评测 accuracy 4.5 点名三硬伤全修：图型数 30/37/40 多口径统一 40、MCP 23 vs 27 自相矛盾消除、引擎头注释 v2.0.0→随版）+计数守卫扩展（mcp/README+cheatsheet 入扫描+引擎头注释版本锁）+视觉样例库 assets/examples/gallery（3 例真实数据成品图+可复现 JSON） |
| v6.4.1 | 2026-10-29 | KM 统计框标签动态降级（CI 不可估不标 95%CI）+log-rank p/显著性分隔 + CI 镜像测试自适应补漏（v6.4.0 新类漏挂 skip 致 CI 首红） |
| v6.4.0 | 2026-10-29 | `--debug` 结构化失败诊断侧车（<output>.debug.json：输入指纹/字段映射/诊断建议）+ SKILL 文档密度守卫锁（体积≤33000B+近期版本行≤700 字符，膨胀即测试拦截） |
| v6.3.0 | 2026-10-27 | 新图型#40：`labbe`L'Abbé 图（Meta 四件套收官：2×2 四格表事件率散点+对角线无效应线+DL 合并 RR 参考线+骑线/夹缝点自动引线）+ --suggest 识别四格表 + Fleiss93 金标 |
| v6.2.0 | 2026-10-27 | 渲染稳定性双修：瞬时失败自动重试（`--retries`，exit 5/6 自动重跑、每代预算 ×1.5、确定性错误不重试）+ `--timeout 0` 保留 7200s 硬保险 + 看门狗小预算精度修复 |
| v6.1.1 | 2026-10-26 | rob 判定扩展：Very high 极高档（ROBINS-I）+Unclear 归顾虑档（ROB1）；三官方工具真实示例；图例在场扫描修复|
| v6.1.0 | 2026-10-26 | 新图型#39：`rob`偏倚风险交通灯图（robvis 同款；判定中英双语；Overall 不入摘要）+「谁拟合谁可视化」能力矩阵+--suggest 识别原始生存长表|
| v6.0.0 | 2026-10-24 | CSV 误差列约定：`X_se/X_sem/X_sd/X_err` 后缀列自动配对主列画误差棒（`X_sd+X_n`→SEM），`--csv-no-errors` 逃生门|
| v5.9.2 | 2026-10-24 | 发布卫生：移除误入 5.9.1 包的内部发布清单文档（无代码变化，5.9.0/5.9.1 功能不变）|
| v5.9.1 | 2026-10-24 | cif `cause_labels` 键一致性守卫（引用不存在的码=exit 3 中文报错；缺标软提示）——标签对调类错误全谱封堵|
| v5.9.0 | 2026-10-23 | 新图型#38：`rmst`限制平均生存时间（每臂 RMST(95%CI)+ΔRMST 差值 CI；τ 截断回归锁+colon 619 例金标）+生存触发词扩充|
| v5.8.0 | 2026-10-23 | `subgroup --stats cox` 一键亚组拟合（逐层 Cox HR+似然比 P 交互；veteran 真实队列金标全表锁）+行预算 ADR-0001+CI 工作流 |
| v5.7.1 | 2026-10-21 | 修复 galbraith 弧线读数原子性（不可判读弧线整条省略）+弧线/读数一一配对测试锁 |
| v5.7.0 | 2026-10-21 | 新图型#37：`cif`竞争风险累计发生率（Aalen-Johansen+1−KM 高估对照）+反模式集中页 antipatterns.md+Galbraith 弧线读数密度放宽 |
| v5.6.1 | 2026-10-20 | 修复 galbraith 深分支崩溃（>12 研究+离群标注相邻）+研究名避让算法升级（纵向 8 档+横向镜像，密集区零叠印） |
| v5.6.0 | 2026-10-20 | 新图型#36：`galbraith`Galbraith 径向图（Meta 异质性可视化）+composite `share_y` 共享尺度自动统一+提示词卡片文档 |
| v5.5.0 | 2026-10-19 | 新图型#35：`subgroup`亚组分析森林图 |
| v5.4.0 | 2026-10-18 | 新图型#34：`rcs`限制性立方样条/非线性剂量反应曲线 |
| v5.3.0 | 2026-10-17 | 新图型#33：`nomogram`列线图（临床预测模型第四件） |
| v5.2.0 | 2026-10-16 | 新图型#31：`calibration`校准曲线（临床预测模型三件套补齐） |
| v5.1.0 | 2026-10-15 | 新图型#30：`dumbbell`哑铃图（差距优先） |
| v5.0.0 | 2026-10-14 | 新图型#29：`ridgeline`山脊图（joyplot形态） |
| v4.10.0 | 2026-10-13 | 新图型#28：`swimmer`游泳图（临床试验逐例时长） |
| v4.9.0 | 2026-10-11 | 新图型#27：`raincloud`雨云图（分布三合一：半小提琴+箱线+雨点抖动） |
| v4.8.0 | 2026-10-10 | 新图型#26：`waterfall`肿瘤缓解瀑布图（RECIST 着色+阈值线） |
| v4.7.0 | 2026-10-09 | 新图型#25：`upset`集合交集图（≥5 集合，韦恩图后继）+场景指南 |
| v4.6.0 | 2026-10-08 | ROC默认语义轴标题 |
| v4.5.0 | 2026-10-07 | 两轮独立第三方严格测试（约180用例、24/24图型、教科书数据集）的12项修复 |
| v4.4.0 | 2026-10-06 | 新图型：`volcano`火山图（第24种） |
| v4.0.0 | 2026-10-02 | 引擎模块化第四刀 |
| v3.10.0 | 2026-10-01 | `dual_axis`重构为经典"左柱右线"形态 |
| v3.9.1 | 2026-09-30 | 文档准确性热修 |
| v3.9.0 | 2026-09-30 | `--order` |

## 最近版本详情 Recent Details


## 7.2.1 — 2026-11-05

发布文案清净化重发（代码与 7.2.0 零差异）：

- 7.2.0 版本详情文案含评测导向表述，未过平台安全审核——中性化改写重发；
  并新增机械锁：changelog 主文件禁评测导向表述（历史存档豁免不回改），
  纳入 check_consistency 扫描面。

## 7.2.0 — 2026-11-05

一致性扫描器 v2 + 安装/预览体验收口：

- **扫描面扩到引擎源码**：`check_consistency.py` 自此扫描 `scripts/*.py` 的运行时
  文案（CHART_NOTES 等 `--explain` 输出字典；注释/docstring 属历史描述豁免）——
  准确性审查发现的 `af_shared.py` bar 文案旧口径即此类（文档已更新而引擎
  --explain 输出仍沿用旧表述，一句话与实现脱节）。文案已更新为双通道真相
  （JSON errors 或 CSV 后缀列 v6.0+）。
- **MCP 安装本地优先**：mcp/README 安装顺序反转——`uv pip install -e <包目录>/mcp`
  （清华/阿里镜像）升为默认推荐，uvx git 直跑降为国际网络备选。
- **视觉样例库直链**：SKILL 双语能力表与 README 的 gallery 行直挂 GitHub 仓库
  预览链接（包内为可复现 JSON+命令，成品图在仓库一键可看）。


## 7.1.0 — 2026-11-04

命令一致性与文档可信度继续收口（自动一致性检查机制化）：

- **新工具 `scripts/check_consistency.py`**：全部命令载体（模板/真实场景 `_command` +
  文档代码块共 118 条）× 能力注册表（旗标域表 + `--verify` 仅 PDF 规则）逐条校验，
  另扫 CSV 误差旧口径残留——有发现 exit 1，发车闸直接可挂。本版首跑清出 15 处
  历史混用：9 处模板/真实场景命令带 `--verify` 却输出 PNG（该门禁仅对 PDF 生效）、
  提示词卡柱状图示例误用 `--stats auto`（聚合数据不可自动检验——正确通道=
  `comparisons` + `--sig-bracket`，或原始数据 `-t box --sig-auto`）、反模式页与
  SKILL 双语能力表的 CSV 表述过时。
- **域表真值化**：`--stats` 的旗标域补齐 raincloud（auto|multi 合法域，此前域表漏列）
  与 subgroup（cox 合法域）——`--list-flags` 输出自此与参数门禁完全一致。
- **ADR-0001 状态如实化**：Proposed→Partially adopted（af_resilience/af_shared/
  wrapper 通路已落地；af_draw 拆分继续推进）。


## 7.0.0 — 2026-11-03

major：全自动组间显著性 + 文档可信度大扫除：

- **新功能 `--sig-auto`**（box/violin/raincloud）：不传 comparisons 也行——引擎自动
  两两检验（Shapiro 判常态：双常态组 Welch t，否则 Mann-Whitney U）并画分层
  bracket+星号。检验家族与 `--stats auto` 一致；聚合图（bar 系）不适用并显式告知
  （用 comparisons + --sig-bracket 传自己的 p 值）。
- **文档可信度大扫除**（准确性审查逐条收口）：`quickstart` CSV 误差旧限制残留与
  SKILL 决策树 JSON-only 提示纠正为 v6.0 后缀列真相；README 分发形态表述如实化
  （中文版=SkillHub 包 SKILL.md，无独立 SKILL_ZH.md 文件随包）；`openpyxl` 入
  正式依赖（--sheet Excel 直读自此声明完备）；MCP 子目录 wheel 包声明如实化
  （引擎随包根 scripts/ 提供，不再虚列 af_engine）。
- **版本历史补全**：6.9.0 节补写（该版 --sig-bracket 与文档收口，此前发版漏记）。

## 6.9.0 — 2026-11-02

新功能 `--sig-bracket` 组间显著性括号 + 文档准确性收口：

- **`--sig-bracket`**：数据携带 `comparisons=[{a,b,p}]` 任意比较对（p 值来自用户
  自己的检验，引擎只可视化）即自动画分层 bracket+星号（<0.05/<0.01/<0.001，
  NS 显式标注）；分层布局按区间重叠最低层分配（span 长的在底层，共享端点判撞）；
  bar/grouped_bar/box/violin 四图型可用。
- 文档准确性收口：MCP 工具清单补 `query_capability`、README 中文文档描述纠正、
  MCP 安装新增不依赖 GitHub 的本地路线、版本历史标注当前版本。


## 6.8.0 — 2026-10-31

MCP 能力透明化 + CSV 误差能力文档真相化（准确性审查逐条收口）：

- **MCP 新工具 `query_capability`**：一命令查图型能力面——chart 留空=全图型
  清单+每旗标域适用图型速览；指定图型名=该图型专属/域外旗标（[ignored] 预告）。
  动态真值=引擎 --list-types / --list-flags，MCP 能力面自此与引擎注册表零漂移。
- **CHARTS 内置回退清单补齐 + 快照锁**：v5.7 时代快照缺 labbe/rmst/rob 三图型
  （引擎可达时被运行时刷新掩盖，静态读源码即见缺口）——补齐至 43 键，并新增
  AST 快照锁（内置清单必须与引擎注册表逐键一致，回退清单从此不可能漂移）。
- **CSV 误差能力文档真相化**：v6.0 已实现 CSV 误差列（`X_se`/`X_sd`/`X_sem`/
  `X_err` 后缀列自动配对），但 FAQ 两处、data-formats 三处仍写「CSV 不支持
  误差棒」，引擎 stderr 两处 hint 同样与自身行为矛盾（同屏打出旧文案+成功
  识别误差列）——全部改写为准确表述，测试锁防回潮。
- **守卫扩容**：计数守卫清单纳入 mcp/pyproject.toml（英文描述曾漏网），正则
  容忍英文修饰语结构（「N publication-grade academic chart types」类）；文档
  内「第 N 种图型」历史序号改为「第 N 号图型」（编号保留，不再与总量口径混淆）。
- **品牌斜纹补齐**：glm-brand 单系列柱状图不再豁免斜纹（斜纹是品牌组成部分
  非可选项）；其他主题显式 --hatch 单系列豁免时改为 [ignored] 显式告知
  （不再静默失效）。
- **show-values 精度自适应**：数值标签随系列数据小数位显示（cap 3 位）——
  固定 1 位小数曾把 5.032 显示成 5.0（视觉门机械断言抓获，交付链回归锁）。


## 6.7.0 — 2026-10-31

`--list-flags` 旗标域查询（实质新功能）+ 官方点名四项收口。

- **`--list-flags [TYPE]`**：一命令查某图型的专属旗标与域外旗标（[ignored] 预告），
  无参数=全图型速览；别名可达（survival→km）。旗标域注册表与 argparse 及图型
  注册表**测试互锁**（域内图型必须是真实注册图型、旗标必须存在于 argparse）——
  自此注册表不可能与引擎漂移。
- **收口**：SKILL 双语「37」残留×5（含 Section Nav 锚链接 `37-chart-types`——
  守卫正则自此覆盖连字符变体）；gallery 改为 JSON+README 随包（PNG 仅 GitHub/
  CH），File Index 指针不再死链，打包器新增死链断言；Python 下限统一 3.8
  （FAQ 3.9→3.8，与 frontmatter/README 对齐）；changelog 版本索引表格解挤压
  （一版本一行）；README 测试徽章与套件数**动态互锁**（徽章数==def test_ 计数）。

## 6.6.0 — 2026-10-30

一致性清剿第二轮（accuracy/usability 4.6 官方点名逐条收口）：

- **MCP**：server.py 头注释旧计数与 render_chart 文档、提交文档
  MCP_SUBMISSION.md 旧计数 → 全部 40（随引擎注册表）。
- **SKILL**：文件索引与代码布局两处旧计数 → 40；quickstart 旧计数 → 40。
- **limits.md**：四组旗标的「静默忽略」描述纠为实际的 `[ignored]` 显式告知行为
  （v3.6 起的真实行为，文档描述自此与代码一致）；并新增活文档禁「静默忽略」
  机械锁（历史存档豁免）。
- **守卫扩展**：计数扫描扩至 14 份活文档，正则覆盖裸 `types` 与「种学术图型/
  种投稿级学术图表」口径——上一轮漏网的全部口径自此入闸。
