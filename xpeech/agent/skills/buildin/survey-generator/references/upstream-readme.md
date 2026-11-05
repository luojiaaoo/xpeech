# Survey Format Converter / 问卷格式转换工具

将 AI 生成的调查问卷文本自动转换为目标问卷平台可识别的导入格式，无需手动逐字输入。

AI-generated survey text → **One command** → Platform-ready import format (问卷星 / 腾讯问卷 / 金数据)

## 功能特性

- ✅ **多平台支持** — 问卷星、腾讯问卷、金数据三大主流问卷平台
- ✅ **自动题型识别** — 单选题、多选题、填空题、矩阵题、排序题、比重题、量表题
- ✅ **题型标记自适应** — 各平台格式自动匹配（问卷星 `【多选题】` / 腾讯 `[单选题]` / 金数据 `［多选题］`）
- ✅ **智能解析** — 兼容多种 AI 输出风格（标准结构化、Markdown 列表、行内选项、嵌入填空等）
- ✅ **金数据矩阵展开** — 矩阵题自动展开为独立单选题
- ✅ **三种输入方式** — 粘贴文本 / 文件输入 / 管道输入

## 快速开始

```bash
# 从文件转换到问卷星
uv run python survey_format_converter.py --platform 问卷星 --input survey.txt

# 从文件转换到腾讯问卷
uv run python survey_format_converter.py --platform 腾讯问卷 --input survey.txt

# 从文件转换到金数据
uv run python survey_format_converter.py --platform 金数据 --input survey.txt

# 管道输入（Linux/macOS）
cat survey.txt | uv run python survey_format_converter.py --platform 问卷星

# 输出到文件
uv run python survey_format_converter.py --platform 问卷星 --input survey.txt --output result.txt

# 查看帮助
uv run python survey_format_converter.py --help

# 查看支持的平台
uv run python survey_format_converter.py --list-platforms
```

## 支持的输入格式

工具自动识别以下 AI 输出风格：

### 标准结构化（最常见）
```
1. 您的性别是？（单选题）
A. 男
B. 女
```

### 含可多选标注
```
1. 您喜欢哪些运动？（可多选）
A. 跑步
B. 游泳
C. 篮球
```

### Markdown 列表
```
### 1. 您的性别是？ [单选题]
- A. 男
- B. 女
```

### 行内选项
```
1. 您对公司前景如何看待？
□非常看好 □比较看好 □一般 □不太看好
```

### 含嵌入填空
```
2. 您部门主要可提供哪些招商要素保障支持？（可多选）
□用地指标 □能耗指标 □其他____
```

### 矩阵 / 量表题
```
请为以下方面打分：
| 1分 | 2分 | 3分 | 4分 | 5分
工作环境
团队协作
```

### 带章节标题
```
模块A：基本信息

1. 您的性别是？
A. 男
B. 女
```

## 题型标记映射

| 题型 | 问卷星 | 腾讯问卷 | 金数据 |
|------|--------|---------|-------|
| 单选题 | — | `[单选题]` | `［单选题］` |
| 多选题 | `【多选题】` | `[多选题]` | `［多选题］` |
| 填空题 | `____` | `[单行文本题]` | `［单行文字］` |
| 排序题 | `【排序题】` | `[排序题]` | `［排序］` |
| 量表题 | 矩阵格式 | `[量表题]` + `1 ~ 5` | `［评分］` |
| 矩阵题 | `\|` 分隔 | `[矩阵单选题]` | 展开为独立题 |
| 说明文本 | 原样输出 | 段落保留 | `［描述］` |

## 安装到 AI 助手（SKILL 模式）

项目包含一个 OpenClaw 兼容的 AI Skill，让 AI 助手能够根据需求直接生成问卷。

### 方式一：通过 ClawHub 安装（推荐）

```bash
# 先发布到 ClawHub（需要 npx clawhub）
cd skill
npx clawhub publish

# 其他用户安装
npx clawhub install survey-generator
```

### 方式二：手动安装（通用）

将 `skill/` 文件夹复制到 AI 助手的 skills 目录：

| AI 平台 | 目标路径 |
|---------|---------|
| WorkBuddy | `~/.workbuddy/skills/survey-generator/` |
| CodeBuddy | `~/.codebuddy/skills/survey-generator/` |
| OpenClaw | `~/.openclaw/skills/survey-generator/` |

### 方式三：复制提示词（任何 AI 通用）

如果你使用的是 ChatGPT、Claude 等通用 AI，打开 `skill/SKILL.md`，从 `步骤 1` 到 `步骤 3` 的完整指令直接复制给 AI，它就能按规范生成问卷。

## 项目结构

```
survey-format-converter/
├── README.md                       ← 本文件
├── survey_format_converter.py      ← Python 转换脚本（零依赖）
├── examples/                       ← 输入输出示例
│   ├── input_ai_survey.txt
│   ├── output_wjx.txt
│   ├── output_tencent.txt
│   └── output_jinshuju.txt
└── skill/
    └── SKILL.md                    ← AI Skill（OpenClaw 兼容）
```

## 许可

```
usage: survey_format_converter.py [-h] [--platform {问卷星,腾讯问卷,金数据}]
                                  [--text TEXT] [--input INPUT_FILE]
                                  [--output OUTPUT] [--list-platforms]

将 AI 生成的问卷文本转换为目标平台的导入格式

optional arguments:
  -h, --help            show this help message and exit
  --platform, -p {问卷星,腾讯问卷,金数据}
                        目标问卷平台
  --text, -t TEXT       直接传入问卷文本（适用于短问卷）
  --input, -i INPUT_FILE
                        输入文件路径
  --output, -o OUTPUT   输出文件路径（默认输出到终端）
  --list-platforms      列出支持的平台
```

## 示例文件

`examples/` 目录包含：
- `input_ai_survey.txt` — AI 生成的问卷输入示例
- `output_wjx.txt` — 转换后的问卷星格式
- `output_tencent.txt` — 转换后的腾讯问卷格式
- `output_jinshuju.txt` — 转换后的金数据格式

## 许可

MIT License
