---
name: summarize
short_description: 规则式文本摘要
description: 使用本地规则脚本，从文本、TXT、Markdown、DOCX、PDF 或网页正文中优先选取含数字或提示词的句子，生成长度受限的摘录并统计关键词。适用于快速预览；不调用摘要模型，不保证语义覆盖或中文分词准确，扫描版 PDF 需先 OCR。
---

# 规则式文本摘要

脚本以句子中的数字和提示词排优先级，选取若干句并按字符长度截断；关键词基于简单正则匹配和词频。适合快速预览，不是调用模型的语义摘要服务。用户要求深度归纳或改写时，读取原文后由模型完成，并说明所读范围。

## 运行方式

以当前加载的 `SKILL.md` 所在目录作为 `<技能目录>`。内建目录只读，输入、输出与 `.venv` 保存在用户工作区；命令从工作区执行，使用技能资源的绝对路径。

首次使用时在工作区准备依赖（已有 `.venv` 时省略创建）：

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r "<技能目录>/references/requirements.txt"
```

## 使用

```bash
uv run python "<技能目录>/scripts/summarize.py" "这里是需要摘录的文本。主要结论为增长 12%。" --output json
uv run python "<技能目录>/scripts/summarize.py" --file ./report.md --length long
uv run python "<技能目录>/scripts/summarize.py" --url https://example.com/article --output markdown
uv run python "<技能目录>/scripts/summarize.py" --keywords "文本内容" --keywords-count 5
```

- `--length`：`short` / `medium` / `long`，分别最多 300 / 600 / 1500 字符，截断时追加省略号。
- `--output`：`text` / `markdown` / `json`；`--include-meta` 添加统计；`--keywords-count` 默认 8。
- 文本文件、DOCX 段落和带文字层 PDF 可读取；不提取 DOCX 表格，不含 OCR，无法承诺处理扫描版。
- 网页模式通过 HTTP 抓取静态 HTML 并去除导航等标签，不执行 JavaScript，也不能读取登录后内容。
- 中文连续文本按整段汉字匹配，关键词可能不准确；摘录也可能遗漏背景与结论，交付前核对原文。
