#!/usr/bin/env -S uv run python
"""
问卷格式转换工具 - Survey Format Converter (v1.0.0)

将 AI 生成的调查问卷文本自动转换为目标问卷平台可识别的导入格式。

支持平台:
  1. 问卷星 (wjx.cn)         — 文本导入格式
  2. 腾讯问卷 (wj.qq.com)     — 文本编辑格式
  3. 金数据 (jinshuju.net)    — 粘贴文本创建表单格式

用法:
  uv run python survey_format_converter.py --platform 问卷星 --text "问卷内容..."
  uv run python survey_format_converter.py --platform 腾讯问卷 --input survey.txt
  cat survey.txt | uv run python survey_format_converter.py --platform 金数据
  uv run python survey_format_converter.py --platform 问卷星 --input survey.txt --output result.txt
  uv run python survey_format_converter.py --list-platforms
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TextIO


# ═══════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════

class QuestionType(Enum):
    """问卷题目类型"""
    SINGLE = auto()        # 单选题
    MULTIPLE = auto()      # 多选题
    FILL = auto()          # 填空题
    MATRIX_SINGLE = auto() # 矩阵单选题
    SORT = auto()          # 排序题
    WEIGHT = auto()        # 比重题
    SCALE = auto()         # 量表题/评分题
    PARAGRAPH = auto()     # 段落说明/纯文本
    DROPDOWN = auto()      # 下拉题
    SECTION = auto()       # 章节标题（如"模块A：..."、"一、..."）


# 各平台支持的题型标记
PLATFORM_TYPE_MARKS: dict[str, dict[QuestionType, str]] = {
    "问卷星": {
        QuestionType.SINGLE: "",
        QuestionType.MULTIPLE: "【多选题】",
        QuestionType.SORT: "【排序题】",
        QuestionType.WEIGHT: "【比重题】",
    },
    "腾讯问卷": {
        QuestionType.SINGLE: "[单选题]",
        QuestionType.MULTIPLE: "[多选题]",
        QuestionType.FILL: "[单行文本题]",
        QuestionType.SCALE: "[量表题]",
        QuestionType.MATRIX_SINGLE: "[矩阵单选题]",
        QuestionType.SORT: "[排序题]",
    },
    "金数据": {
        QuestionType.SINGLE: "［单选题］",
        QuestionType.MULTIPLE: "［多选题］",
        QuestionType.FILL: "［单行文字］",
        QuestionType.SCALE: "［评分］",
        QuestionType.SORT: "［排序］",
        QuestionType.DROPDOWN: "［下拉框］",
        QuestionType.PARAGRAPH: "［描述］",
    },
}


@dataclass
class OptionItem:
    """选项项"""
    text: str = ""           # 选项文本
    has_fill: bool = False   # 是否包含填空（如"其他____"）
    fill_text: str = ""      # 填空位置的提示文本


@dataclass
class Question:
    """题目"""
    type: QuestionType
    title: str               # 题目内容
    number: int = 0          # 题号（0表示无编号）
    options: list[OptionItem] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)  # 矩阵行标题
    columns: list[str] = field(default_factory=list)        # 矩阵列标题
    description: str = ""    # 题目说明（可选）
    scale_min: int = 1       # 量表最小值
    scale_max: int = 5       # 量表最大值

    def is_choice_type(self) -> bool:
        """是否为选择题型"""
        return self.type in (QuestionType.SINGLE, QuestionType.MULTIPLE,
                             QuestionType.DROPDOWN)


@dataclass
class Survey:
    """问卷"""
    title: str = ""
    description: str = ""    # 问卷说明/欢迎语
    questions: list[Question] = field(default_factory=list)


# ═══════════════════════════════════════════════════
# 输入解析器
# ═══════════════════════════════════════════════════

# ─── 正则模式 ─────

# 匹配 "一、" "二、" "模块A：" "模块B：" 等章节标题
RE_SECTION = re.compile(r'^[（(]?[一二三四五六七八九十模块ABCDEFG]+[）).、：:\s]|^模块[ABCDEFG]?[：:]')

# 匹配题号开头： "1." "1、" "1)" "1. " "1、"
RE_QUESTION_NUMBER = re.compile(r'^\s*(?:#+\s*)?\*{0,2}\s*(\d+)\s*[.、)）]\s*')

# 匹配标题中嵌入的题型标记: （单选题） [单选题] 【单选题】 （可多选）
RE_TYPE_MARK = re.compile(r'[（(【\[]\s*(单选题|多选题|排序题|比重题|填空题|量表题|评分题|矩阵单选题|矩阵多选题|单行文本题|多行文本题|下拉题|可多选|填空)\s*[）)】\]]')

# 匹配填空占位符
RE_FILL_PLACEHOLDER = re.compile(r'_+|[（(][\s_]*[）)]')

# 匹配选项前缀: "A." "A、" "A)" "a." "1." 等
RE_OPTION_PREFIX = re.compile(r'^([A-Za-z]|[0-9]+)\s*[.、)）]\s*')

# 匹配行内选项 (□选项 □选项)
RE_INLINE_OPTION = re.compile(r'□([^□]+?)(?=□|$)')

# 行内选项中的嵌入填空: "其他____" → 分离为文本+填空
RE_OPTION_FILL = re.compile(r'(.+?)([_（）(]{2,}|[（(][\s_]*[）)])\s*$')


def detect_question_type(title: str, options_raw: list[str]) -> QuestionType:
    """从题目文本和选项中推断题型"""
    title_clean = title.strip()

    # 1. 检查题目标记
    for mark in ["【多选题】", "【排序题】", "【比重题】", "[多选题]", "［多选题］"]:
        if mark in title_clean:
            return {
                "【多选题】": QuestionType.MULTIPLE,
                "【排序题】": QuestionType.SORT,
                "【比重题】": QuestionType.WEIGHT,
                "[多选题]": QuestionType.MULTIPLE,
                "［多选题］": QuestionType.MULTIPLE,
            }[mark]

    # 2. 检查标题末尾的 (可多选) 标记
    if re.search(r'[（(]可多选[）)]', title_clean):
        return QuestionType.MULTIPLE

    # 3. 检查 (多选题) (单选题) 等
    type_match = RE_TYPE_MARK.search(title_clean)
    if type_match:
        label = type_match.group(1)
        mapping = {
            "单选题": QuestionType.SINGLE,
            "多选题": QuestionType.MULTIPLE,
            "排序题": QuestionType.SORT,
            "比重题": QuestionType.WEIGHT,
            "填空题": QuestionType.FILL,
            "量表题": QuestionType.SCALE,
            "评分题": QuestionType.SCALE,
            "矩阵单选题": QuestionType.MATRIX_SINGLE,
            "矩阵多选题": QuestionType.MATRIX_SINGLE,
            "单行文本题": QuestionType.FILL,
            "多行文本题": QuestionType.FILL,
            "下拉题": QuestionType.DROPDOWN,
        }
        if label in mapping:
            return mapping[label]

    # 4. 检查填空题标记
    if RE_FILL_PLACEHOLDER.search(title_clean):
        return QuestionType.FILL

    # 5. 检查是否有明确的简短填空提示
    if "简短填空" in title_clean:
        return QuestionType.FILL

    # 6. 分析选项
    options_text = " ".join(options_raw)
    has_box = "□" in options_text
    has_choice_prefix = bool(RE_OPTION_PREFIX.search(options_text))

    if has_box:
        # 如果有□标记，但标题没说是可多选，默认为单选题
        # 但若选项是"□是 □否"这种典型的判断对，也是单选
        return QuestionType.SINGLE

    # 7. 矩阵题检测（用|分隔的）
    has_pipe = "|" in options_text and len(options_raw) >= 2
    if has_pipe:
        return QuestionType.MATRIX_SINGLE

    # 8. 默认：如果有选项，按单选题处理
    if options_raw:
        return QuestionType.SINGLE

    # 没有选项的纯文本 → 段落说明
    return QuestionType.PARAGRAPH


def parse_inline_options(line: str) -> list[OptionItem]:
    """解析行内排列的选项，如 '□是 □否' """
    items: list[OptionItem] = []
    # 先按□分割
    parts = RE_INLINE_OPTION.findall(line)
    if parts:
        for p in parts:
            p = p.strip()
            item = OptionItem(text=p)
            # 检查是否有内嵌填空
            fill_match = RE_OPTION_FILL.match(p)
            if fill_match:
                item.text = fill_match.group(1).strip()
                item.has_fill = True
                item.fill_text = ""
            items.append(item)
        return items

    # 没有□标记，尝试按空格/制表符分隔的纯文本选项
    # 但如果选项数量很少（2-5个），且每个都不带题号/序号
    segments = [s.strip() for s in re.split(r'\s{2,}|\t', line) if s.strip()]
    if len(segments) >= 2:
        for s in segments:
            items.append(OptionItem(text=s))
    return items


def parse_option_line(line: str) -> OptionItem | None:
    """解析单行选项，返回 OptionItem 或 None"""
    line = line.strip()
    if not line:
        return None

    # 去掉 Markdown 列表标记
    line = re.sub(r'^[-*+]\s*', '', line).strip()
    # 去掉 □ ○ 标记
    line = re.sub(r'^[□○]\s*', '', line).strip()

    if not line:
        return None

    # 去掉选项前缀 A. A、A) 等
    opt_text = RE_OPTION_PREFIX.sub('', line, count=1).strip()
    if opt_text:
        text = opt_text
    else:
        text = line

    item = OptionItem(text=text)

    # 检查内嵌填空: "其他____" "其他______"
    fill_match = RE_OPTION_FILL.match(text)
    if fill_match:
        item.text = fill_match.group(1).strip()
        item.has_fill = True
        item.fill_text = ""

    return item


def is_section_header(line: str) -> bool:
    """判断是否为章节标题"""
    line = line.strip()
    if not line:
        return False
    # 匹配 "一、" "二、" "模块A：" 等
    if re.match(r'^[一二三四五六七八九十]+[、.）)]', line):
        return True
    if re.match(r'^模块[ABCDEFG]?[：:]', line):
        return True
    if re.match(r'^（?[一二三四五六七八九十]+[）).]', line):
        return True
    return False


def parse_survey(text: str) -> Survey:
    """将 AI 生成的问卷文本解析为 Survey 数据模型

    支持多种 AI 输出格式:
      - 标准结构化: "1. 题目（单选题）" + "A. 选项"
      - Markdown: "### 1. 题目" + "- 选项"
      - 含特殊标记: "【多选题】" "（可多选）"
      - 行内选项: "□是 □否"
      - 选项含嵌入填空: "其他____"
    """
    survey = Survey()
    lines = text.split('\n')

    # ── 第一遍: 识别结构 ──
    current_question: Question | None = None
    pending_title: str | None = None  # 等待确定是否为题目的文本
    found_first_question = False
    options_buffer: list[str] = []  # 暂存当前题目的选项行
    in_matrix = False
    matrix_columns: list[str] = []

    def flush_question():
        """将缓冲区的题目和选项处理为一个 Question"""
        nonlocal current_question, pending_title, options_buffer
        nonlocal in_matrix, matrix_columns

        if pending_title is None:
            return

        title = pending_title
        # 从标题中移除题号前的章节前缀（如 "一、" 等）
        # 但保留题号本身

        # 提取题号
        num_match = RE_QUESTION_NUMBER.match(title)
        q_number = 0
        if num_match:
            q_number = int(num_match.group(1))
            title = title[num_match.end():].strip()

        # 从标题中移除 Markdown 语法（两侧 **）
        title = re.sub(r'^\*{1,2}\s*', '', title)
        title = re.sub(r'\s*\*{1,2}$', '', title)
        title = re.sub(r'^#+\s*', '', title).strip()

        # 从标题中移除题型标记（已用于判断类型）
        title = RE_TYPE_MARK.sub('', title).strip()
        # 也移除 (可多选)
        title = re.sub(r'[（(]可多选[）)]', '', title).strip()
        # 移除填空题的提示文字: （简短填空）
        title = re.sub(r'[（(]简短填空[）)]', '', title).strip()
        # 移除末尾的标点冗余
        title = title.rstrip('，,；;')
        # 清理多余空格
        title = re.sub(r'\s+', ' ', title).strip()

        # 解析选项
        parsed_options: list[OptionItem] = []
        is_inline_options = False

        # 检查是否行内选项（所有选项在一行上）
        if len(options_buffer) == 1 and '□' in options_buffer[0]:
            inline_items = parse_inline_options(options_buffer[0])
            if len(inline_items) >= 2:
                parsed_options = inline_items
                is_inline_options = True

        if not is_inline_options:
            for opt_line in options_buffer:
                # 如果是矩阵题的分隔行
                if '|' in opt_line and not parsed_options:
                    cols = [c.strip() for c in opt_line.split('|') if c.strip()]
                    if len(cols) >= 2:
                        matrix_columns = cols
                        in_matrix = True
                        continue
                # 如果是矩阵题的行标题
                if in_matrix:
                    row = re.sub(r'^[-*+]\s*', '', opt_line).strip()
                    if row and '|' not in row:
                        # 这是矩阵的一行
                        if not hasattr(current_question, '_matrix_rows_detected'):
                            pass
                        survey.questions[-1].sub_questions.append(row)
                        continue
                    # 如果出现 |，表示新的矩阵列定义，重置
                    if '|' in row:
                        cols = [c.strip() for c in row.split('|') if c.strip()]
                        matrix_columns = cols
                        continue

                item = parse_option_line(opt_line)
                if item:
                    parsed_options.append(item)

        # 确定题型
        q_type = detect_question_type(pending_title, options_buffer)

        # 如果是矩阵题的特殊处理
        if in_matrix and matrix_columns:
            q_type = QuestionType.MATRIX_SINGLE
            # 从最后一题获取 sub_questions
            if survey.questions and survey.questions[-1].type == QuestionType.MATRIX_SINGLE:
                pass  # sub_questions 已经在上面添加了

        # 创建 Question 对象
        q = Question(
            type=q_type,
            title=title,
            number=q_number,
            options=parsed_options,
            columns=matrix_columns if in_matrix else [],
            sub_questions=survey.questions[-1].sub_questions if (in_matrix and survey.questions) else [],
        )

        # 检查量表题
        if q_type == QuestionType.SCALE:
            for opt_line in options_buffer:
                scale_match = re.search(r'(\d+)\s*[~～]\s*(\d+)', opt_line)
                if scale_match:
                    q.scale_min = int(scale_match.group(1))
                    q.scale_max = int(scale_match.group(2))
                    break

        survey.questions.append(q)

        # 清理矩阵状态
        if not in_matrix:
            matrix_columns = []
        in_matrix = False

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()

        # ── 空行：分隔题目 ──
        if not line:
            if pending_title and found_first_question:
                flush_question()
                pending_title = None
                options_buffer = []
            continue

        # ── Markdown 分隔线 (---, ***) ──
        if re.match(r'^[-*]{3,}$', line):
            if pending_title:
                flush_question()
                pending_title = None
                options_buffer = []
            continue

        # ── 章节标题：作为段落说明保存 ──
        if is_section_header(line):
            if pending_title:
                flush_question()
            pending_title = None
            options_buffer = []
            # 添加为段落
            survey.questions.append(Question(
                type=QuestionType.SECTION,
                title=line,
                description="章节标题",
            ))
            continue

        # ── 长段落（超过 120 字且不含选项标记）→ 段落说明 ──
        if len(line) > 120 and '□' not in line and 'A.' not in line and not RE_QUESTION_NUMBER.match(line):
            if pending_title:
                flush_question()
            pending_title = None
            options_buffer = []
            survey.questions.append(Question(
                type=QuestionType.PARAGRAPH,
                title=line,
            ))
            continue

        # ── 检测是否为题号行 ──
        if RE_QUESTION_NUMBER.match(line):
            if pending_title:
                flush_question()
            pending_title = line
            options_buffer = []
            found_first_question = True

        # ── 检测是否为选项行 ──
        elif found_first_question and pending_title:
            # 判断是否为选项
            is_option = False

            # 含 □ 或 ○
            if '□' in line or '○' in line:
                is_option = True
            # 有选项前缀 A. A、A) 1. 等
            elif RE_OPTION_PREFIX.match(line):
                is_option = True
            # Markdown 列表项
            elif re.match(r'^[-*+]\s+', line):
                is_option = True
            # 行内选项（多个短文本用空格分隔，且与当前题目相关）
            elif '|' in line:
                is_option = True  # 矩阵题列标题
            # 数字范围（量表题）
            elif re.match(r'^\d+\s*[~～]\s*\d+', line):
                is_option = True
            # 看起来像纯文本选项（没有前缀、没有□，但跟在题目后面的文本）
            elif not re.match(r'^[A-Za-z一二三四五六七八九十模块]', line) and not is_section_header(line):
                # 如果这一行看起来像是普通文本且上一行是题目
                # 但需要确保它确实是个选项（不长，不是独立句子）
                if len(line) < 60:
                    is_option = True

            if is_option:
                options_buffer.append(line)
            else:
                # 不是选项也不是新题目 → 段落说明（可能是欢迎语等）
                if pending_title and not found_first_question:
                    survey.description = line
                else:
                    survey.questions.append(Question(
                        type=QuestionType.PARAGRAPH,
                        title=line,
                    ))

        # ── 问卷标题/欢迎语（在第一个题目之前） ──
        elif not found_first_question:
            clean_line = re.sub(r'^\*{1,2}\s*', '', line)
            clean_line = re.sub(r'\s*\*{1,2}$', '', clean_line)
            if not survey.title:
                survey.title = clean_line
            else:
                # 第二行可能作为描述/欢迎语
                if not survey.description:
                    survey.description = clean_line
                else:
                    survey.description += "\n" + clean_line

    # 处理最后一个题目
    if pending_title:
        flush_question()

    # 后处理：清理空 options
    survey.questions = [q for q in survey.questions if q.title.strip()]

    return survey


# ═══════════════════════════════════════════════════
# 平台格式化器
# ═══════════════════════════════════════════════════

def format_wjx(survey: Survey) -> str:
    """格式化为问卷星文本导入格式"""
    lines: list[str] = []

    if survey.title:
        lines.append(survey.title)
        lines.append("")

    for q in survey.questions:
        if q.type == QuestionType.PARAGRAPH or q.type == QuestionType.SECTION:
            if q.title.strip():
                lines.append(q.title.strip())
                lines.append("")
            continue

        # 题号 + 题目
        title_text = q.title
        if q.number > 0:
            title_text = f"{q.number}. {title_text}"

        # 添加题型标记
        type_mark = ""
        if q.type in (QuestionType.MULTIPLE, QuestionType.SORT, QuestionType.WEIGHT):
            type_mark = PLATFORM_TYPE_MARKS["问卷星"].get(q.type, "")
        if type_mark:
            title_text = f"{title_text}{type_mark}"

        # 填空题
        if q.type == QuestionType.FILL:
            if '____' not in title_text:
                title_text = f"{title_text}：____"
            lines.append(title_text)
            lines.append("")
            continue

        # 矩阵题
        if q.type == QuestionType.MATRIX_SINGLE:
            lines.append(title_text)
            if q.columns:
                lines.append("| " + " | ".join(q.columns) + " |")
            for sub in q.sub_questions:
                lines.append(sub.strip() if sub.strip() else "（行）")
            lines.append("")
            continue

        # 选择题（单选/多选/排序/比重）
        if q.is_choice_type() or q.type in (QuestionType.SORT, QuestionType.WEIGHT):
            lines.append(title_text)
            for opt in q.options:
                opt_text = opt.text
                if opt.has_fill:
                    opt_text = f"{opt_text}____"
                lines.append(opt_text)
            lines.append("")
            continue

        # 量表题
        if q.type == QuestionType.SCALE:
            lines.append(title_text)
            # 问卷星没有量表的特殊文本标记，用 | 模拟
            scale_labels = [str(i) for i in range(q.scale_min, q.scale_max + 1)]
            lines.append("| " + " | ".join(scale_labels) + " |")
            lines.append(q.title)
            lines.append("")

        # 兜底
        lines.append(title_text)
        lines.append("")

    return "\n".join(lines).strip()


def format_tencent(survey: Survey) -> str:
    """格式化为腾讯问卷文本编辑格式"""
    lines: list[str] = []

    if survey.title:
        lines.append(survey.title)

    # 腾讯问卷的欢迎语格式
    if survey.description:
        lines.append("")
        lines.append("欢迎语")
        lines.append(survey.description)

    for q in survey.questions:
        if q.type == QuestionType.PARAGRAPH or q.type == QuestionType.SECTION:
            if q.title.strip():
                lines.append("")
                lines.append(q.title.strip())
            continue

        lines.append("")

        # 题目内容
        title_text = q.title

        # 添加题型标记（腾讯问卷用半角方括号 [题型]）
        type_mark = PLATFORM_TYPE_MARKS["腾讯问卷"].get(q.type, "")
        if not type_mark:
            # 未定义的题型用默认
            if q.is_choice_type():
                type_mark = "[单选题]"

        line = f"{title_text}{type_mark}"
        lines.append(line)

        # 填空题—没有额外选项
        if q.type == QuestionType.FILL:
            continue

        # 量表题—输出范围
        if q.type == QuestionType.SCALE:
            lines.append(f"{q.scale_min} ~ {q.scale_max}")
            continue

        # 矩阵题
        if q.type == QuestionType.MATRIX_SINGLE:
            if q.columns:
                lines.append("| " + " | ".join(q.columns) + " |")
            for sub in q.sub_questions:
                sub_text = sub.strip()
                if sub_text:
                    lines.append(sub_text)
            continue

        # 选择题—选项一行一个，无 A/B/C 前缀
        if q.options:
            for opt in q.options:
                opt_text = opt.text
                if opt.has_fill:
                    opt_text = f"{opt_text}____"
                lines.append(opt_text)

    return "\n".join(lines).strip()


def format_jinshuju(survey: Survey) -> str:
    """格式化为金数据粘贴文本格式"""
    lines: list[str] = []

    # 第一行: 《标题》
    if survey.title:
        lines.append(f"《{survey.title}》")
    else:
        lines.append("《问卷》")

    # 第二行: 描述
    if survey.description:
        lines.append(survey.description)
    else:
        lines.append("")

    # 第三行: 空行（必须）
    lines.append("")

    for q in survey.questions:
        if q.type == QuestionType.PARAGRAPH or q.type == QuestionType.SECTION:
            if q.title.strip():
                lines.append(f"［描述］{q.title.strip()}")
                lines.append("")
            continue

        # 获取金数据的题型标记（全角符号）
        type_mark = PLATFORM_TYPE_MARKS["金数据"].get(q.type)
        if not type_mark:
            # 兜底映射
            if q.is_choice_type():
                type_mark = "［单选题］"
            elif q.type == QuestionType.FILL:
                type_mark = "［单行文字］"
            elif q.type == QuestionType.SCALE:
                type_mark = "［评分］"
            else:
                type_mark = "［描述］"

        # 构造题干
        title_text = q.title
        # 金数据格式: [题型]题干(描述)
        hint = ""
        if q.type == QuestionType.FILL:
            hint = "请填写"
        elif q.type == QuestionType.MULTIPLE:
            hint = "可多选"
        elif q.type == QuestionType.SINGLE:
            hint = "请选择"

        hint_str = f"({hint})" if hint else ""
        if hint_str:
            line = f"{type_mark}{title_text}{hint_str}"
        else:
            line = f"{type_mark}{title_text}"
        lines.append(line)

        # 填空题/评分题/描述 — 没有选项行
        if q.type in (QuestionType.FILL, QuestionType.SCALE, QuestionType.PARAGRAPH):
            lines.append("")
            continue

        # 矩阵题—金数据不支持矩阵，转为一组独立单选题
        if q.type == QuestionType.MATRIX_SINGLE:
            for sub in q.sub_questions:
                sub_text = sub.strip()
                if sub_text:
                    # 每个子行作为一个独立的单选题
                    lines.append(f"［单选题］{sub_text}(请选择)")
                    for col in (q.columns or []):
                        col_text = col.strip()
                        if col_text:
                            lines.append(col_text)
                    lines.append("")
            continue

        # 选择题—选项每行一个
        if q.options:
            for opt in q.options:
                opt_text = opt.text
                if opt.has_fill:
                    opt_text = f"{opt_text}____"
                lines.append(opt_text)
        lines.append("")

    return "\n".join(lines).strip()


# 平台格式化器映射
FORMATTERS: dict[str, callable] = {
    "问卷星": format_wjx,
    "腾讯问卷": format_tencent,
    "金数据": format_jinshuju,
}

# 合法平台名
VALID_PLATFORMS = list(FORMATTERS.keys())


# ═══════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════

def read_input(source: str | None, file_path: str | None, stdin: TextIO) -> str:
    """从命令行文本或文件或 stdin 读取输入"""
    if source:
        return source
    if file_path:
        path = Path(file_path)
        if not path.exists():
            print(f"[ERR] 文件不存在: {file_path}", file=sys.stderr)
            sys.exit(1)
        return path.read_text(encoding="utf-8")
    # 从 stdin 读取
    return stdin.read()


def main():
    parser = argparse.ArgumentParser(
        description="问卷格式转换工具 - 将 AI 生成的问卷转换为目标平台的导入格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --platform 问卷星 --text "1. 您的性别是?\\nA. 男\\nB. 女"
  %(prog)s --platform 腾讯问卷 --input survey.txt
  cat survey.txt | %(prog)s --platform 金数据
  %(prog)s --platform 问卷星 --input survey.txt --output result.txt
  %(prog)s --list-platforms
        """,
    )

    parser.add_argument(
        "--platform", "-p",
        choices=VALID_PLATFORMS,
        help=f"目标问卷平台: {'/'.join(VALID_PLATFORMS)}",
    )
    parser.add_argument(
        "--text", "-t",
        help="直接传入问卷文本（适用于短问卷）",
    )
    parser.add_argument(
        "--input", "-i",
        dest="input_file",
        help="输入文件路径",
    )
    parser.add_argument(
        "--output", "-o",
        help="输出文件路径（默认输出到终端）",
    )
    parser.add_argument(
        "--list-platforms",
        action="store_true",
        help="列出支持的平台",
    )

    args = parser.parse_args()

    # 列出平台
    if args.list_platforms:
        print("支持的问卷平台:")
        for p in VALID_PLATFORMS:
            print(f"  - {p}")
        return

    # 检查必要参数
    if not args.platform:
        print("[ERR] 请指定目标平台 (--platform 或 --list-platforms 查看)", file=sys.stderr)
        sys.exit(1)

    # 读取输入
    try:
        raw_text = read_input(args.text, args.input_file, sys.stdin)
    except Exception as e:
        print(f"[ERR] 读取输入失败: {e}", file=sys.stderr)
        sys.exit(1)

    if not raw_text.strip():
        print("[ERR] 输入内容为空", file=sys.stderr)
        sys.exit(1)

    # 解析
    try:
        survey = parse_survey(raw_text)
    except Exception as e:
        print(f"[ERR] 问卷解析失败: {e}", file=sys.stderr)
        if __debug__:
            import traceback
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    if not survey.questions:
        print("[WARN] 未识别到任何题目，请检查输入格式", file=sys.stderr)
        print("[INFO] 原始输入:")
        print(raw_text)
        sys.exit(1)

    # 格式化
    formatter = FORMATTERS[args.platform]
    try:
        result = formatter(survey)
    except Exception as e:
        print(f"[ERR] 格式转换失败: {e}", file=sys.stderr)
        if __debug__:
            import traceback
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # 输出
    if args.output:
        try:
            Path(args.output).write_text(result, encoding="utf-8")
            print(f"[OK] 转换完成，已输出到: {args.output}", file=sys.stderr)
        except Exception as e:
            print(f"[ERR] 写入输出文件失败: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        sys.stdout.write(result)
        sys.stdout.write("\n")
        print(file=sys.stderr)  # 空行
        print(f"[OK] 转换完成 (共 {len(survey.questions)} 题)", file=sys.stderr)


if __name__ == "__main__":
    main()
