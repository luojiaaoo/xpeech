---
name: gongwenformat-pro
short_description: 公文 Word 排版
description: 使用本地脚本将 Markdown、TXT 或 DOCX 正文套入 GB/T 9704-2012 公文版式，生成 Word 文件，支持标题、层级、红头、发文字号及版记。适用于已有内容的公文排版；字体需本机可用，PDF 导出另需 LibreOffice。包名含 Pro，但脚本没有付费接口或激活检查。
---

## 运行约定

从当前加载的 `SKILL.md` 所在目录确定技能路径，不写死全局安装位置。内建技能目录只读；输入、输出及虚拟环境放在用户工作区。下文 `scripts/`、`assets/`、`references/` 相对技能目录解析，运行时使用绝对路径；Python 统一通过 `uv run python` 启动。

首次运行时，在用户工作区执行 `uv pip install --python .venv/bin/python -r "<技能目录>/references/requirements.txt"` 安装依赖（`.venv` 不存在时先用 `uv venv .venv` 创建）。

# 党政机关公文标准排版 Pro Skill

依据《党政机关公文格式》国家标准(GB/T 9704-2012),将任意格式文本自动排版为标准公文格式的 Word 文档。

## 何时使用

- 用户写好了研讨材料、发言稿、汇报材料、通知、请示等,需要生成正式Word文件
- 用户要求"排版"、"标准格式"、"公文格式"、"红头文件"
- 用户说"发给我Word"、"生成docx"、"帮我排个版"
- 用户发来 TXT、不标准的Word文件,要求排成标准格式

## 使用方法

### 第一步:准备输入文件

接收用户发来的文件,或将用户聊天中发的文字保存为文件。

**输入格式支持:**
- **Markdown (.md)**:自动识别 `##` 标题和 `**加粗**` 标记
- **纯文本 (.txt)**:按空行分段,智能识别层级序号
- **Word (.docx)**:提取纯文本后智能识别层级
- **直接文字**:用户在聊天中发的文字,保存为 .txt 后处理

### 第二步:生成Word文件

```bash
uv run python <skill-path>/scripts/gongwen_format.py \
  --title "公文标题" \
  --input /tmp/content.md \
  --output /tmp/output.docx
```

基础参数:
- `--title`:公文标题(二号小标宋,居中)
- `--input`:输入文件路径(支持 .md / .txt / .docx)
- `--output`:输出文件路径(支持 .docx 或 .pdf)
- `--format`:输出格式 `docx`(默认)或 `pdf`(需要 LibreOffice)

查看完整参数列表:`uv run python <skill-path>/scripts/gongwen_format.py --help`

### 第三步:发送文件

将生成的 `.docx` 文件发送给用户。

## 排版标准与高级参数

详细的排版标准参数(页面设置、字体字号、段落格式、页码、层次序号)、高级参数(红头、发文字号、版记、抄送等)以及注意事项,请查阅:

👉 **`references/gb9704-2012.md`** - GB/T 9704-2012 排版标准参数速查

### 常用高级参数速览

| 参数 | 说明 | 示例 |
|------|------|------|
| `--redhead` | 红头机关名称 | `--redhead "XX省人民政府"` |
| `--doc-number` | 发文字号 | `--doc-number "X政发〔2026〕12号"` |
| `--author` | 发文机关(落款) | `--author "XX镇人民政府"` |
| `--date` | 成文日期 | `--date "2026年4月16日"` |
| `--print-author` | 印发机关(版记) | `--print-author "XX镇人民政府办公室"` |
| `--print-date` | 印发日期(版记) | `--print-date "2026-04-17"` |
| `--cc` | 抄送机关(版记) | `--cc "县委办公室"` |
| `--copies` | 份号(6位数字) | `--copies "000001"` |
| `--secret-level` | 密级和保密期限 | `--secret-level "机密★20年"` |
| `--urgency` | 紧急程度 | `--urgency "特急"` |
| `--signer` | 签发人(上行文) | `--signer "张三"` |
| `--recipient` | 主送机关 | `--recipient "县委办公室"` |
| `--notes` | 附注(可多个) | `--notes "此件公开发布"` |

## 快速示例

用户说：“帮我把这篇研讨材料排成标准格式发给我”

操作步骤：
1. 将用户发来的内容保存为文件（md/txt/docx），或接收用户发送的文件
2. 运行 gongwen_format.py 生成 Word
3. 发送 .docx 文件给用户

### 完整示例（包含所有新要素）

用户说：“帮我排一份涉密公文，要包含份号、密级、签发人、主送机关、附注”

```bash
uv run python <skill-path>/scripts/gongwen_format.py \
  --title “关于进一步加强党建工作的通知” \
  --input /tmp/content.md \
  --output /tmp/output.docx \
  --redhead “XX省人民政府” \
  --doc-number “X政发〔2026〕12号” \
  --copies “000001” \
  --secret-level “机密★20年” \
  --urgency “特急” \
  --signer “张三” \
  --recipient “各市、县人民政府，省直各部门：” \
  --author “XX省人民政府” \
  --date “2026年4月26日” \
  --notes “此件公开发布” “联系人：李四，电话：0571-88888888” \
  --print-author “XX省人民政府办公室” \
  --print-date “2026-04-27” \
  --cc “省委办公厅，省人大常委会办公厅，省政协办公厅”
```

### PDF 导出示例

用户说:"帮我排版,要 PDF 格式"

```bash
uv run python <skill-path>/scripts/gongwen_format.py \
  --title "关于XXX的实施意见" \
  --input /tmp/content.md \
  --output /tmp/output.pdf
```

也可以用 `--format pdf` 参数:

```bash
uv run python <skill-path>/scripts/gongwen_format.py \
  --title "关于XXX的实施意见" \
  --input /tmp/content.md \
  --output /tmp/output.docx \
  --format pdf
```

> ⚠️ PDF 导出需要安装 LibreOffice:`sudo apt install libreoffice-writer`
