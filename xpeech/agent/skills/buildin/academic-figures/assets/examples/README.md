# Examples — 示例数据与预览

每个 `.json` 都是可直接运行的示例数据，配合 `scripts/gen_figure.py` 使用。

## 快速体验（推荐）

```bash
# 一键环境准备（首次使用）
uv run python scripts/setup_env.py

# 交互演示：选择图表类型，自动用内置数据出图
uv run python scripts/gen_figure.py --demo --style glm-hatch --cjk

# 查看全部配色（终端内色块预览）
uv run python scripts/gen_figure.py --list-themes
```

## 逐图示例

| 文件 | 命令 | 说明 |
|------|------|------|
| example_bar.json | `uv run python scripts/gen_figure.py -t bar -d assets/examples/example_bar.json -o fig.png --cjk` | 分组柱状图（药物疗效） |
| example_forest.json | `uv run python scripts/gen_figure.py -t forest -d assets/examples/example_forest.json -o fig.pdf --cjk` | Meta 分析森林图 |
| example_heatmap.json | `uv run python scripts/gen_figure.py -t heatmap -d assets/examples/example_heatmap.json -o fig.png --cjk` | 免疫指标相关热力图 |
| example_line.json | `uv run python scripts/gen_figure.py -t line -d assets/examples/example_line.json -o fig.png --cjk` | 多组折线图（临床评分） |
| example_stacked.json | `uv run python scripts/gen_figure.py -t stacked_bar -d assets/examples/example_stacked.json -o fig.png --cjk` | 构成比堆叠柱状图 |

## 真实科研场景示例（realworld/）

`assets/examples/realworld/`：贴近论文真实形态的场景数据（含样本量、真实基因名/终点命名惯例），
可直接作为你论文图 1/2/3 的起点：

| 文件 | 命令要点 | 场景 |
|------|------|------|
| realworld_rct_response.json | `-t grouped_bar --cjk` | RCT 三臂 48 周应答率（n=240） |
| realworld_omics_volcano.json | `-t volcano` | RNA-seq 差异表达（干扰素通路） |
| realworld_survival_km.json | `-t km --cjk` | IgA 肾病三臂生存曲线（36.5 月随访） |
| realworld_oncology_waterfall.json | `-t waterfall --cjk --show-values` | NSCLC II 期最佳缓解瀑布（RECIST n=26） |

## GLM 黄蓝斜线风格（招牌风格）

```bash
uv run python scripts/gen_figure.py -t bar -d assets/examples/example_bar.json -o fig.png --style glm-hatch --cjk
```

`--style glm-hatch` = `--theme glm --hatch`：素雅莫兰迪配色 + 黑色斜纹填充，
打印/黑白场景同样清晰，色盲安全。

## 配色预览

`previews/swatch_<theme>.png`：7 套配色的色板预览图（glm/classic/okabe-ito/nature/lancet/conservative/cool）。
也可随时用 `--theme-swatch <theme> -o out.png` 重新生成。
