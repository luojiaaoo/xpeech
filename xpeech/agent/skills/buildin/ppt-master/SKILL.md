---
name: ppt-master
short_description: 模板式 PPTX 生成
description: 根据 JSON 内容配置，使用预设页面版式和配色模板生成可编辑的 PPTX 文件。支持封面、目录、章节、正文、图表、表格、图片和简单流程图；适用于接受模板布局的汇报与培训课件。脚本负责填充版式，内容由模型整理；不提供自由排版、动画、视频嵌入或已有 PPT 模板编辑。
---

# 模板式 PPTX 生成

将模型整理的内容写入 JSON，再由 `python-pptx` 填入脚本内置的固定版式，生成 16:9 可编辑 PPTX。适合接受预设版式的汇报、培训课件和产品介绍。

## 运行方式

以当前加载的 `SKILL.md` 所在目录作为 `<技能目录>`。内建目录只读，输入、输出与 `.venv` 保存在用户工作区；命令从工作区执行，使用技能资源的绝对路径。

首次使用时在工作区准备依赖（已有 `.venv` 时省略创建）：

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r "<技能目录>/references/requirements.txt"
```

## 工作流程

1. 根据用户提供的主题、受众和材料整理内容；只追问影响内容的缺失信息。用户已给出配色或大纲时直接沿用。
2. 参考 [配置示例](references/example-config.json) 和 [暖色配置示例](references/example-config-warm.json)，写出工作区内的 JSON。必需字段为 `meta` 和非空 `slides`。
3. 运行生成命令，回读 PPTX 核对页数、主要文字和图表数据后发送文件。

```bash
uv run python "<技能目录>/scripts/generate_ppt.py" --config ./ppt_config.json --output ./report.pptx
```

## 可用配置与边界

- `slides[].type` 支持 `cover`、`toc`、`section`、`content`、`chart`、`table`、`image`、`flowchart`、`ending`。正文 `content` 是含 `text`、可选 `level`/`bold`/`font_size` 的对象数组。
- `meta.category` 选择脚本内置的类别配色；`style.color_primary`、`color_secondary`、`color_accent`、`color_background`、`color_text` 覆盖颜色，`font_title`、`font_body` 和 `palette` 覆盖字体与图表配色。
- 图表 `categories` 与各 `series[].values` 数量必须相等。散点图的 `categories` 是数值 X 坐标。支持类型以 `scripts/generate_ppt.py` 的 `CHART_TYPE_MAP` 为准；上游图表参考中的 `bar_3d`、`line_3d`、`pie_3d` 不受当前导出器支持。
- 图片仅插入本地 `image_path`，缺失会显示占位框；流程图是横向节点和箭头。字体由查看文档的机器提供，未嵌入字体。
- 不支持导入现有 `.pptx`/`.potx` 模板、自由设计版式、自动适配任意长文本、艺术字、动画或视频。上游示例中的 `wordart`、`animation`、全局字号和 `bullet_style` 字段未实现，不据此承诺效果。
- 需要自由排版的浏览器演示页面时选用 `presentation-generation`；需要套版且可编辑的 PPTX 时选用本技能。
