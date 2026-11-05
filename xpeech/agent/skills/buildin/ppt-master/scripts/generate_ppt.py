#!/usr/bin/env -S uv run python
"""
PPT Master — JSON 填充预设版式与类别配色
Usage:
    uv run python generate_ppt.py --config /tmp/ppt_config.json --output /workspace/output.pptx
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.chart import XL_CHART_TYPE
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.enum.shapes import MSO_SHAPE


# ════════════════════════════════════════════════════════════════════
# 一、颜色工具
# ════════════════════════════════════════════════════════════════════

def hex_to_rgb(hex_color: str) -> RGBColor:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"无效的十六进制颜色: {hex_color}")
    return RGBColor(
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def lighten(hex_color: str, factor: float = 0.7) -> RGBColor:
    c = hex_to_rgb(hex_color)
    return RGBColor(
        int(c[0] + (255 - c[0]) * factor),
        int(c[1] + (255 - c[1]) * factor),
        int(c[2] + (255 - c[2]) * factor),
    )


def hex_to_rgb_tuple(h: str):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# ════════════════════════════════════════════════════════════════════
# 二、8大类别温和色系调色板（参考WPS Office风格）
# ════════════════════════════════════════════════════════════════════

CATEGORY_PALETTES = {
    "营销推广": {
        "primary": "#E8634A",
        "secondary": "#F5A623",
        "accent": "#4A90D9",
        "background": "#FDF8F5",
        "text": "#2C3E50",
        "palette": ["#E8634A", "#F5A623", "#4A90D9", "#7ED321", "#D0021B", "#F8E71C", "#50E3C2", "#B8E986"],
        "font_title": "微软雅黑",
        "font_body": "微软雅黑",
    },
    "教育培训": {
        "primary": "#4A90D9",
        "secondary": "#7ED321",
        "accent": "#F5A623",
        "background": "#F5F9FF",
        "text": "#2C3E50",
        "palette": ["#4A90D9", "#7ED321", "#F5A623", "#50E3C2", "#9013FE", "#D0021B", "#4AE3B0", "#F8E71C"],
        "font_title": "微软雅黑",
        "font_body": "微软雅黑",
    },
    "汇报总结": {
        "primary": "#2C3E50",
        "secondary": "#3498DB",
        "accent": "#E74C3C",
        "background": "#FFFFFF",
        "text": "#2C3E50",
        "palette": ["#2C3E50", "#3498DB", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C", "#E67E22"],
        "font_title": "微软雅黑",
        "font_body": "微软雅黑",
    },
    "企业宣讲": {
        "primary": "#1A237E",
        "secondary": "#5C6BC0",
        "accent": "#FF6F00",
        "background": "#F5F5F5",
        "text": "#1A237E",
        "palette": ["#1A237E", "#5C6BC0", "#FF6F00", "#00BCD4", "#8BC34A", "#FF5722", "#9E9E9E", "#CDDC39"],
        "font_title": "微软雅黑",
        "font_body": "微软雅黑",
    },
    "党政民生": {
        "primary": "#B71C1C",
        "secondary": "#D4A017",
        "accent": "#F5F5F5",
        "background": "#FFFDF5",
        "text": "#1A1A1A",
        "palette": ["#B71C1C", "#D4A017", "#F5F5F5", "#8D6E63", "#D32F2F", "#FBC02D", "#795548", "#E53935"],
        "font_title": "宋体",
        "font_body": "微软雅黑",
    },
    "人资行政": {
        "primary": "#455A64",
        "secondary": "#78909C",
        "accent": "#8D6E63",
        "background": "#F8F9FA",
        "text": "#263238",
        "palette": ["#455A64", "#78909C", "#8D6E63", "#A1887F", "#90A4AE", "#BCAAA4", "#607D8B", "#D7CCC8"],
        "font_title": "微软雅黑",
        "font_body": "微软雅黑",
    },
    "医疗健康": {
        "primary": "#00695C",
        "secondary": "#26A69A",
        "accent": "#4DB6AC",
        "background": "#F0F7F4",
        "text": "#004D40",
        "palette": ["#00695C", "#26A69A", "#4DB6AC", "#80CBC4", "#A5D6A7", "#C8E6C9", "#B2DFDB", "#E0F2F1"],
        "font_title": "微软雅黑",
        "font_body": "微软雅黑",
    },
    "产品研发": {
        "primary": "#6A1B9A",
        "secondary": "#AB47BC",
        "accent": "#CE93D8",
        "background": "#F8F0FC",
        "text": "#4A148C",
        "palette": ["#6A1B9A", "#AB47BC", "#CE93D8", "#7B1FA2", "#BA68C8", "#E1BEE7", "#9C27B0", "#F3E5F5"],
        "font_title": "微软雅黑",
        "font_body": "微软雅黑",
    },
}

# 通用温和色系调色板（用户未指定类别时使用）
GENTLE_PALETTES = {
    "莫兰迪": ["#C4B7A6", "#D4C5A9", "#E8D5C4", "#B8C4C9", "#D1B8B8", "#C9D4C5", "#D4C9C9", "#C5C9C4"],
    "北欧蓝": ["#A8D8EA", "#AAE3E2", "#D4E2D4", "#F7E5D4", "#F5D4D4", "#D4C5A9", "#B8D4D4", "#E8D5C4"],
    "暖阳": ["#FADBD8", "#F5CBA7", "#F9E79F", "#A9DFBF", "#AED6F1", "#F2D7D5", "#D5F5E3", "#FCF3CF"],
    "清新": ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#FAC858", "#EE6666", "#73C0DE", "#3BA272"],
}


def get_category_style(category: str) -> dict:
    """根据类别获取配色方案"""
    if category in CATEGORY_PALETTES:
        return CATEGORY_PALETTES[category]
    # 默认使用汇报总结风格
    return CATEGORY_PALETTES["汇报总结"]


# ════════════════════════════════════════════════════════════════════
# 三、当前导出器支持的图表类型
# ════════════════════════════════════════════════════════════════════

CHART_TYPE_MAP = {
    # 柱状图（4种）
    "bar_clustered": XL_CHART_TYPE.COLUMN_CLUSTERED,
    "bar_stacked": XL_CHART_TYPE.COLUMN_STACKED,
    "bar_100": XL_CHART_TYPE.COLUMN_STACKED_100,
    # 条形图（3种）
    "bar_horizontal": XL_CHART_TYPE.BAR_CLUSTERED,
    "bar_h_stacked": XL_CHART_TYPE.BAR_STACKED,
    "bar_h_100": XL_CHART_TYPE.BAR_STACKED_100,
    # 折线图（4种）
    "line_markers": XL_CHART_TYPE.LINE_MARKERS,
    "line_stacked": XL_CHART_TYPE.LINE_STACKED,
    "line_100": XL_CHART_TYPE.LINE_STACKED_100,
    # 饼图（2种）
    "pie": XL_CHART_TYPE.PIE,
    # 环形图
    "doughnut": XL_CHART_TYPE.DOUGHNUT,
    # 面积图（3种）
    "area": XL_CHART_TYPE.AREA,
    "area_stacked": XL_CHART_TYPE.AREA_STACKED,
    "area_100": XL_CHART_TYPE.AREA_STACKED_100,
    # 雷达图（2种）
    "radar": XL_CHART_TYPE.RADAR,
    "radar_filled": XL_CHART_TYPE.RADAR_FILLED,
    # 散点图
    "scatter": XL_CHART_TYPE.XY_SCATTER,
}

# 图表类型中文名称
CHART_NAMES = {
    "bar_clustered": "柱状图（簇状）",
    "bar_stacked": "柱状图（堆积）",
    "bar_100": "柱状图（百分比堆积）",
    "bar_3d": "柱状图（3D簇状）",
    "bar_horizontal": "条形图（簇状）",
    "bar_h_stacked": "条形图（堆积）",
    "bar_h_100": "条形图（百分比堆积）",
    "line_markers": "折线图（带数据标记）",
    "line_stacked": "折线图（堆积）",
    "line_100": "折线图（百分比堆积）",
    "line_3d": "折线图（3D）",
    "pie": "饼图",
    "pie_3d": "饼图（3D）",
    "doughnut": "环形图",
    "area": "面积图",
    "area_stacked": "面积图（堆积）",
    "area_100": "面积图（百分比堆积）",
    "radar": "雷达图",
    "radar_filled": "雷达图（填充）",
    "scatter": "散点图",
}


# ════════════════════════════════════════════════════════════════════
# 四、配置验证
# ════════════════════════════════════════════════════════════════════

def validate_config(config: dict) -> list:
    errors = []
    if "meta" not in config:
        errors.append("缺少 'meta' 字段")
        return errors
    if "slides" not in config or not config["slides"]:
        errors.append("缺少 'slides' 字段或为空")
        return errors

    for i, slide in enumerate(config["slides"]):
        slide_type = slide.get("type", "")
        valid_types = ("cover", "toc", "content", "chart", "table", "image", "flowchart", "ending", "section")
        if slide_type not in valid_types:
            errors.append(f"幻灯片 {i+1}: 未知类型 '{slide_type}'")

        if slide_type == "chart":
            chart_type = slide.get("chart_type", "")
            if chart_type not in CHART_TYPE_MAP:
                errors.append(f"幻灯片 {i+1}: 未知图表类型 '{chart_type}'，支持: {', '.join(CHART_TYPE_MAP.keys())}")
            if "categories" not in slide or "series" not in slide:
                errors.append(f"幻灯片 {i+1}: chart 缺少 categories 或 series")
            else:
                cat_count = len(slide.get("categories", []))
                for s_idx, s in enumerate(slide["series"]):
                    if len(s.get("values", [])) != cat_count:
                        errors.append(
                            f"幻灯片 {i+1}: series[{s_idx}] values 数量 ({len(s.get('values',[]))}) "
                            f"与 categories 数量 ({cat_count}) 不匹配"
                        )
    return errors


# ════════════════════════════════════════════════════════════════════
# 五、幻灯片生成器
# ════════════════════════════════════════════════════════════════════

def add_cover_slide(prs, slide_data, style):
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)

    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = hex_to_rgb(style["primary"])

    # 装饰条
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(5.5), prs.slide_width, Inches(2))
    shape.fill.solid()
    shape.fill.fore_color.rgb = hex_to_rgb(style["secondary"])
    shape.line.fill.background()

    # 标题
    txBox = slide.shapes.add_textbox(Inches(1), Inches(1.5), prs.slide_width - Inches(2), Inches(1.5))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = slide_data.get("title", "标题")
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    p.font.name = style.get("font_title", "微软雅黑")
    p.alignment = PP_ALIGN.LEFT

    # 副标题
    subtitle = slide_data.get("subtitle", "")
    if subtitle:
        txBox2 = slide.shapes.add_textbox(Inches(1), Inches(3.2), prs.slide_width - Inches(2), Inches(0.8))
        tf2 = txBox2.text_frame
        p2 = tf2.paragraphs[0]
        p2.text = subtitle
        p2.font.size = Pt(20)
        p2.font.color.rgb = lighten(style["primary"], 0.8)
        p2.font.name = style.get("font_body", "微软雅黑")

    # 页脚
    footer = slide_data.get("date", datetime.now().strftime("%Y-%m-%d"))
    author = slide_data.get("author", "")
    if author:
        footer += f"  |  {author}"
    txBox3 = slide.shapes.add_textbox(Inches(1), Inches(6.5), prs.slide_width - Inches(2), Inches(0.5))
    tf3 = txBox3.text_frame
    p3 = tf3.paragraphs[0]
    p3.text = footer
    p3.font.size = Pt(12)
    p3.font.color.rgb = RGBColor(255, 255, 255)
    p3.font.name = style.get("font_body", "微软雅黑")


def add_toc_slide(prs, slide_data, style):
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)

    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = hex_to_rgb(style["background"])

    # 标题
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(8), Inches(0.8))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = "目  录"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = hex_to_rgb(style["primary"])
    p.font.name = style.get("font_title", "微软雅黑")

    # 装饰线
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.3), Inches(1.5), Pt(4))
    line.fill.solid()
    line.fill.fore_color.rgb = hex_to_rgb(style["accent"])
    line.line.fill.background()

    items = slide_data.get("items", [])
    for i, item in enumerate(items):
        y = Inches(1.8 + i * 0.7)
        # 序号圆圈
        circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(1.0), y, Inches(0.4), Inches(0.4))
        circle.fill.solid()
        circle.fill.fore_color.rgb = hex_to_rgb(style["primary"])
        circle.line.fill.background()
        ctf = circle.text_frame
        cp = ctf.paragraphs[0]
        cp.text = str(i + 1)
        cp.font.size = Pt(14)
        cp.font.bold = True
        cp.font.color.rgb = RGBColor(255, 255, 255)
        cp.font.name = style.get("font_title", "微软雅黑")
        cp.alignment = PP_ALIGN.CENTER

        txBox2 = slide.shapes.add_textbox(Inches(1.7), y, Inches(7), Inches(0.4))
        tf2 = txBox2.text_frame
        p2 = tf2.paragraphs[0]
        p2.text = item
        p2.font.size = Pt(18)
        p2.font.color.rgb = hex_to_rgb(style["text"])
        p2.font.name = style.get("font_body", "微软雅黑")


def add_section_slide(prs, slide_data, style):
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)

    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = hex_to_rgb(style["primary"])

    num = slide_data.get("section_number", 1)
    txBox = slide.shapes.add_textbox(Inches(1), Inches(1.5), Inches(1), Inches(1))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = f"0{num}" if num < 10 else str(num)
    p.font.size = Pt(60)
    p.font.bold = True
    p.font.color.rgb = hex_to_rgb(style["accent"])
    p.font.name = style.get("font_title", "微软雅黑")

    txBox2 = slide.shapes.add_textbox(Inches(1), Inches(3.0), prs.slide_width - Inches(2), Inches(1.5))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = slide_data.get("title", "")
    p2.font.size = Pt(36)
    p2.font.bold = True
    p2.font.color.rgb = RGBColor(255, 255, 255)
    p2.font.name = style.get("font_title", "微软雅黑")


def add_content_slide(prs, slide_data, style):
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)

    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = hex_to_rgb(style["background"])

    # 标题栏
    title_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), prs.slide_width, Inches(1.1))
    title_bar.fill.solid()
    title_bar.fill.fore_color.rgb = hex_to_rgb(style["primary"])
    title_bar.line.fill.background()

    # 章节号标签
    num = slide_data.get("section_number", 1)
    tag = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.3), Inches(0.25), Inches(0.6), Inches(0.6))
    tag.fill.solid()
    tag.fill.fore_color.rgb = hex_to_rgb(style["accent"])
    tag.line.fill.background()
    ttf = tag.text_frame
    tp = ttf.paragraphs[0]
    tp.text = f"0{num}" if num < 10 else str(num)
    tp.font.size = Pt(20)
    tp.font.bold = True
    tp.font.color.rgb = RGBColor(255, 255, 255)
    tp.font.name = style.get("font_title", "微软雅黑")
    tp.alignment = PP_ALIGN.CENTER

    txBox = slide.shapes.add_textbox(Inches(1.2), Inches(0.15), prs.slide_width - Inches(1.8), Inches(0.8))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = slide_data.get("title", "")
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    p.font.name = style.get("font_title", "微软雅黑")

    # 内容
    items = slide_data.get("content", [])
    if not items:
        return
    txBox2 = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), prs.slide_width - Inches(1.6), prs.slide_height - Inches(2.2))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True

    for i, item in enumerate(items):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.text = item.get("text", "")
        p.level = item.get("level", 0)
        p.font.size = Pt(item.get("font_size", 16))
        p.font.bold = item.get("bold", False)
        p.font.color.rgb = hex_to_rgb(item.get("color", style["text"]))
        p.font.name = style.get("font_body", "微软雅黑")
        p.space_after = Pt(8 if p.level == 0 else 4)


def add_chart_slide(prs, slide_data, style):
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)

    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = hex_to_rgb(style["background"])

    # 标题
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.3), prs.slide_width - Inches(1.6), Inches(0.7))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = slide_data.get("title", "图表")
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = hex_to_rgb(style["primary"])
    p.font.name = style.get("font_title", "微软雅黑")

    chart_type_key = slide_data.get("chart_type", "bar_clustered")
    categories = slide_data.get("categories", [])
    series_list = slide_data.get("series", [])

    if chart_type_key == "scatter":
        chart_data = XyChartData()
        for s in series_list:
            series = chart_data.add_series(s.get("name", ""))
            for x, y in zip(categories, s.get("values", [])):
                series.add_data_point(float(x), float(y))
    else:
        chart_data = CategoryChartData()
        chart_data.categories = categories
        for s in series_list:
            chart_data.add_series(s.get("name", ""), s.get("values", []))

    xl_type = CHART_TYPE_MAP.get(chart_type_key, XL_CHART_TYPE.COLUMN_CLUSTERED)
    chart_frame = slide.shapes.add_chart(
        xl_type, Inches(0.5), Inches(1.3), Inches(9.0), Inches(5.5), chart_data
    )

    chart = chart_frame.chart
    chart.has_legend = True
    chart.legend.include_in_layout = False
    chart.legend.font.size = Pt(10)

    # 应用调色板颜色
    palette = style.get("palette", ["#4A90D9", "#7ED321", "#F5A623", "#50E3C2"])
    plot = chart.plots[0]
    for idx, series in enumerate(plot.series):
        color_idx = idx % len(palette)
        series.format.fill.solid()
        series.format.fill.fore_color.rgb = hex_to_rgb(palette[color_idx])


def add_table_slide(prs, slide_data, style):
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)

    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = hex_to_rgb(style["background"])

    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.3), prs.slide_width - Inches(1.6), Inches(0.7))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = slide_data.get("title", "表格")
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = hex_to_rgb(style["primary"])
    p.font.name = style.get("font_title", "微软雅黑")

    headers = slide_data.get("headers", [])
    rows = slide_data.get("rows", [])
    num_rows = len(rows) + 1
    num_cols = len(headers) if headers else (len(rows[0]) if rows else 1)

    table_shape = slide.shapes.add_table(
        num_rows, num_cols,
        Inches(0.8), Inches(1.3),
        prs.slide_width - Inches(1.6), Inches(0.5) * num_rows
    )
    table = table_shape.table

    for col_idx, header in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.text = header
        for para in cell.text_frame.paragraphs:
            para.font.size = Pt(16)
            para.font.bold = True
            para.font.color.rgb = RGBColor(255, 255, 255)
            para.font.name = style.get("font_title", "微软雅黑")
            para.alignment = PP_ALIGN.CENTER
        cell.fill.solid()
        cell.fill.fore_color.rgb = hex_to_rgb(style["primary"])

    for row_idx, row_data in enumerate(rows):
        for col_idx, cell_text in enumerate(row_data):
            if col_idx >= num_cols:
                break
            cell = table.cell(row_idx + 1, col_idx)
            cell.text = str(cell_text)
            for para in cell.text_frame.paragraphs:
                para.font.size = Pt(14)
                para.font.color.rgb = hex_to_rgb(style["text"])
                para.font.name = style.get("font_body", "微软雅黑")
                para.alignment = PP_ALIGN.CENTER
            if row_idx % 2 == 1:
                cell.fill.solid()
                cell.fill.fore_color.rgb = lighten(style["background"], 0.05)


def add_image_slide(prs, slide_data, style):
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)

    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = hex_to_rgb(style["background"])

    title_text = slide_data.get("title", "")
    if title_text:
        txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.3), prs.slide_width - Inches(1.6), Inches(0.7))
        tf = txBox.text_frame
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = hex_to_rgb(style["primary"])
        p.font.name = style.get("font_title", "微软雅黑")

    img_path = slide_data.get("image_path", "")
    top_offset = Inches(1.2) if title_text else Inches(0.5)
    if img_path and os.path.exists(img_path):
        try:
            slide.shapes.add_picture(img_path, Inches(0.8), top_offset,
                                      prs.slide_width - Inches(1.6), prs.slide_height - top_offset - Inches(1.0))
        except Exception:
            _add_placeholder(slide, Inches(2), top_offset + Inches(1), Inches(6), Inches(3), "[图片加载失败]")
    else:
        _add_placeholder(slide, Inches(2), top_offset + Inches(1), Inches(6), Inches(3), "[待放置图片]")


def _add_placeholder(slide, left, top, width, height, text):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(240, 240, 240)
    shape.line.color.rgb = RGBColor(200, 200, 200)
    tf = shape.text_frame
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(18)
    p.font.color.rgb = RGBColor(150, 150, 150)
    p.alignment = PP_ALIGN.CENTER


def add_flowchart_slide(prs, slide_data, style):
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)

    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = hex_to_rgb(style["background"])

    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.3), prs.slide_width - Inches(1.6), Inches(0.7))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = slide_data.get("title", "流程图")
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = hex_to_rgb(style["primary"])
    p.font.name = style.get("font_title", "微软雅黑")

    nodes = slide_data.get("nodes", [])
    if not nodes:
        return

    count = len(nodes)
    start_x = Inches(0.5)
    avail = prs.slide_width - Inches(1.0)
    node_w = min(Inches(2.0), avail / count - Inches(0.2))
    spacing = (avail - node_w * count) / (count + 1)

    palette = style.get("palette", ["#4A90D9", "#7ED321", "#F5A623", "#50E3C2", "#9013FE"])

    for i, node in enumerate(nodes):
        x = start_x + spacing * (i + 1) + node_w * i
        y = Inches(2.5)
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, node_w, Inches(1.0))
        shape.fill.solid()
        shape.fill.fore_color.rgb = hex_to_rgb(palette[i % len(palette)])
        shape.line.fill.background()
        tf2 = shape.text_frame
        tf2.word_wrap = True
        p2 = tf2.paragraphs[0]
        p2.text = node.get("label", f"步骤 {i+1}")
        p2.font.size = Pt(14)
        p2.font.bold = True
        p2.font.color.rgb = RGBColor(255, 255, 255)
        p2.font.name = style.get("font_title", "微软雅黑")
        p2.alignment = PP_ALIGN.CENTER

        if i < count - 1:
            arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                            x + node_w, y + Inches(0.5) - Pt(3),
                                            spacing, Pt(6))
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = hex_to_rgb(style["accent"])
            arrow.line.fill.background()


def add_ending_slide(prs, slide_data, style):
    add_cover_slide(prs, {"title": "谢谢", **slide_data}, style)


SLIDE_BUILDERS = {
    "cover": add_cover_slide,
    "toc": add_toc_slide,
    "section": add_section_slide,
    "content": add_content_slide,
    "chart": add_chart_slide,
    "table": add_table_slide,
    "image": add_image_slide,
    "flowchart": add_flowchart_slide,
    "ending": add_ending_slide,
}


def generate_ppt(config, output):
    """Fill the bundled layouts from JSON and save an editable presentation."""
    errors = validate_config(config)
    if errors:
        raise ValueError("\n".join(errors))
    style = dict(get_category_style(config["meta"].get("category", "汇报总结")))
    for key, value in config.get("style", {}).items():
        if key.startswith("color_"):
            style[key.removeprefix("color_")] = value
        elif key in {"font_title", "font_body", "palette"}:
            style[key] = value
    if not style.get("palette"):
        raise ValueError("palette 不能为空")
    prs = Presentation()
    prs.slide_width = Inches(13.333333)
    prs.slide_height = Inches(7.5)
    prs.core_properties.title = config["meta"].get("title", "")
    prs.core_properties.author = config["meta"].get("author", "")
    for slide in config["slides"]:
        SLIDE_BUILDERS[slide["type"]](prs, slide, style)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    prs.save(output)
    return len(prs.slides)


def main():
    parser = argparse.ArgumentParser(description="按 JSON 和预设版式生成可编辑 PPTX")
    parser.add_argument("--config", required=True, help="JSON 配置路径")
    parser.add_argument("--output", required=True, help="输出 .pptx 路径")
    args = parser.parse_args()
    try:
        if not args.output.lower().endswith(".pptx"):
            raise ValueError("输出路径必须以 .pptx 结尾")
        with open(args.config, encoding="utf-8") as file:
            config = json.load(file)
        count = generate_ppt(config, args.output)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(1, f"生成失败: {exc}\n")
    print(json.dumps({"output": os.path.abspath(args.output), "slides": count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
