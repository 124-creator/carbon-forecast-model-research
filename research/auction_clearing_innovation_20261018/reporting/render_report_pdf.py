#!/usr/bin/env python3
"""Render the complete Chinese Markdown report to a searchable, embedded-font PDF.

Requires reportlab, matplotlib, pillow and pypdf. Pass static Noto Sans SC
Regular/Bold TTF files with --font-dir; fonts are not redistributed in this repo.
This renderer does not alter any model, source data or experiment result.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/carbon_report_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Flowable, Frame, Image, KeepTogether, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents
from pypdf import PdfReader

NAVY = colors.HexColor("#16324F")
TEAL = colors.HexColor("#117D8B")
INK = colors.HexColor("#25394B")
MUTED = colors.HexColor("#647484")
LIGHT = colors.HexColor("#F0F5F7")
LINE = colors.HexColor("#D8E2E8")
AMBER = colors.HexColor("#A96826")
PAGE_W, PAGE_H = A4
MARGIN = 47
CONTENT_W = PAGE_W - 2 * MARGIN


class Rule(Flowable):
    def __init__(self, width, color=LINE):
        super().__init__()
        self.width, self.height, self.color = width, 4, color

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(0.7)
        self.canv.line(0, 2, self.width, 2)


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(MARGIN, 48, CONTENT_W, PAGE_H - 100,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="Report", frames=[frame], onPage=self.decorate)])
        self.heading_serial = 0

    def beforeDocument(self):
        self.heading_serial = 0

    def decorate(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setFont("NotoSC", 8)
            canvas.setFillColor(MUTED)
            canvas.drawString(MARGIN, PAGE_H - 30, "碳拍卖价格预测 | 模型创新研究与真实实验")
            canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 30, "2026-10-18")
            canvas.setStrokeColor(LINE)
            canvas.line(MARGIN, PAGE_H - 36, PAGE_W - MARGIN, PAGE_H - 36)
        canvas.setFont("NotoSC", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 29, "开发评价：2021-2023 | 确认评分未开启")
        canvas.drawRightString(PAGE_W - MARGIN, 29, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        level = getattr(flowable, "toc_level", None)
        if level is not None:
            self.heading_serial += 1
            key = f"section-{self.heading_serial}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(), key, level, False)
            if level == 0:
                self.notify("TOCEntry", (level, flowable.getPlainText(), self.page, key))


def inline(s: str) -> str:
    """Only parse the small, explicit Markdown subset used by this report."""
    s = html.escape(s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
               lambda m: f'<link href="{m.group(2)}" color="#117D8B">{m.group(1)}</link>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r'<font name="NotoSC-Bold">\1</font>', s)
    s = re.sub(r"`([^`]+)`", r'<font color="#405B70">\1</font>', s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r'<font color="#405B70">\1</font>', s)
    s = re.sub(r"×10\^(-?\d+)", r'×10<super>\1</super>', s)
    return s


def styles():
    common = dict(fontName="NotoSC", textColor=INK, wordWrap="CJK",
                  splitLongWords=True, allowWidows=0, allowOrphans=0)
    return {
        "body": ParagraphStyle("Body", fontSize=10.1, leading=15.7, spaceAfter=7, **common),
        "h2": ParagraphStyle("H2", fontName="NotoSC-Bold", fontSize=17.2, leading=23,
                             textColor=NAVY, spaceBefore=17, spaceAfter=10, keepWithNext=True,
                             wordWrap="CJK"),
        "h3": ParagraphStyle("H3", fontName="NotoSC-Bold", fontSize=12.4, leading=18,
                             textColor=TEAL, spaceBefore=11, spaceAfter=6, keepWithNext=True,
                             wordWrap="CJK"),
        "cell": ParagraphStyle("Cell", fontSize=8.7, leading=12.6, spaceAfter=0, **common),
        "cellhead": ParagraphStyle("CellHead", fontName="NotoSC-Bold", fontSize=8.6,
                                   leading=12.8, textColor=colors.white, wordWrap="CJK"),
        "caption": ParagraphStyle("Caption", fontSize=8.5, leading=12.2, spaceAfter=10,
                                  fontName="NotoSC", textColor=MUTED, wordWrap="CJK"),
        "reference": ParagraphStyle("Reference", fontSize=8.7, leading=13, spaceAfter=6,
                                    **common),
        "toc0": ParagraphStyle("Toc0", fontName="NotoSC-Bold", fontSize=10.4,
                               leading=18, textColor=NAVY, spaceBefore=4,
                               leftIndent=0, firstLineIndent=0, wordWrap="CJK"),
        "toc1": ParagraphStyle("Toc1", fontName="NotoSC", fontSize=8.8, leading=14,
                               textColor=MUTED, leftIndent=12, firstLineIndent=0, wordWrap="CJK"),
    }


def table_widths(header):
    n = len(header)
    text = " ".join(header)
    if "神经网络拟合" in text:
        proportions = [0.30, 0.115, 0.115, 0.14, 0.33]
    elif "R_CRPS" in text:
        proportions = [0.28, 0.15, 0.15, 0.25, 0.17]
    elif "SDMC 阶段" in text:
        proportions = [0.235, 0.12, 0.16, 0.25, 0.235]
    elif "年度等权" in text:
        proportions = [0.18, 0.20, 0.20, 0.20, 0.22]
    elif "固定比较对象" in text:
        proportions = [0.23, 0.17, 0.28, 0.32]
    elif "推进条件" in text:
        proportions = [0.22, 0.36, 0.29, 0.13]
    elif "必须完成" in text:
        proportions = [0.13, 0.36, 0.33, 0.18]
    elif n == 2:
        proportions = [0.255, 0.745]
    elif n == 3:
        proportions = [0.245, 0.385, 0.370]
    else:
        proportions = [1 / n] * n
    assert len(proportions) == n
    return [CONTENT_W * p / sum(proportions) for p in proportions]


def make_table(lines, sty):
    raw = [[v.strip() for v in line.strip().strip("|").split("|")] for line in lines]
    rows = [raw[0]] + [row for row in raw[1:] if not all(re.fullmatch(r":?-{3,}:?", v) for v in row)]
    values = [[Paragraph(inline(v), sty["cellhead" if i == 0 else "cell"])
               for v in row] for i, row in enumerate(rows)]
    t = Table(values, colWidths=table_widths(rows[0]), repeatRows=1, hAlign="LEFT")
    cmds = [("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY)]
    for i in range(1, len(rows)):
        cmds.extend([("BACKGROUND", (0, i), (-1, i), LIGHT if i % 2 else colors.white),
                     ("LINEBELOW", (0, i), (-1, i), 0.4, LINE)])
    t.setStyle(TableStyle(cmds))
    return t


def render_math(formula, out_dir, serial):
    path = out_dir / f"formula-{serial:02d}.png"
    fig = plt.figure(figsize=(11, 0.75))
    fig.text(0.5, 0.50, f"${formula}$", fontsize=17, ha="center", va="center", color="#16324F")
    fig.savefig(path, dpi=230, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    with PILImage.open(path) as im:
        w, h = im.size
    draw_w = min(CONTENT_W - 10, w / 230 * 72)
    return Image(str(path), width=draw_w, height=draw_w * h / w, hAlign="CENTER")


def cover(sty):
    story = [Spacer(1, 38)]
    top = ParagraphStyle("CoverEyebrow", fontName="NotoSC-Bold", fontSize=10,
                         leading=16, textColor=TEAL, spaceAfter=20)
    story.append(Paragraph("RESEARCH REPORT / 2026-10-18", top))
    title = ParagraphStyle("CoverTitle", fontName="NotoSC-Bold", fontSize=33,
                           leading=45, textColor=NAVY, spaceAfter=8)
    story.append(Paragraph("碳拍卖价格预测", title))
    sub = ParagraphStyle("CoverSub", fontName="NotoSC-Bold", fontSize=19,
                         leading=28, textColor=NAVY, spaceAfter=20)
    story.append(Paragraph("模型创新研究、完整实验<br/>与发表路线", sub))
    story.append(Rule(CONTENT_W, TEAL))
    story.append(Spacer(1, 19))
    story.append(Paragraph("EEX EUA 单场一级拍卖 · SDMC 与 FDMC · 2021-2023 开发评价", sty["body"]))
    story.append(Spacer(1, 20))
    label = ParagraphStyle("MetricLabel", fontName="NotoSC", fontSize=9.5, leading=15,
                           textColor=MUTED, wordWrap="CJK")
    big = ParagraphStyle("Metric", fontName="NotoSC-Bold", fontSize=25, leading=33,
                         textColor=NAVY)
    card = Table([[Paragraph("1.0050", big), Paragraph("426", big), Paragraph("未评分", big)],
                  [Paragraph("FDMC 主指标 R<br/>RW=1，越低越好", label),
                   Paragraph("实际模型拟合<br/>345 神经 + 81 经典", label),
                   Paragraph("2024-2025 确认阶段<br/>未通过推进门槛", label)]],
                 colWidths=[CONTENT_W / 3] * 3)
    card.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                              ("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 13),
                              ("TOPPADDING", (0, 0), (-1, 0), 14),
                              ("BOTTOMPADDING", (0, 1), (-1, -1), 14)]))
    story += [card, Spacer(1, 27)]
    status = ParagraphStyle("Status", fontName="NotoSC-Bold", fontSize=12.5,
                            leading=21, textColor=AMBER, wordWrap="CJK", spaceAfter=12)
    story.append(Paragraph("当前判断：已完成实验，未取得支持模型优势的证据。", status))
    story.append(Paragraph("尺度分离改善了当前参数化，但随机积分、供给算子与相同信息下的结构预测优势尚未成立。报告完整披露失败、修正、复核及继续研究所需条件。", sty["body"]))
    story += [Spacer(1, 26), Paragraph("交付包含：理论与近邻审查、三阶段全部运行、独立评分、科学图、可复现代码与完整实验材料。主投建议为条件满足后的 Journal of Forecasting；不承诺录用。", sty["caption"]), PageBreak()]
    return story


def build_markdown(path, sty, math_dir):
    lines = path.read_text().splitlines()
    # Cover carries the title and metadata; keep all substantive report sections.
    start = next(i for i, line in enumerate(lines) if line.startswith("## 结论"))
    lines = lines[start:]
    story, i, math_no, figure_no = [], 0, 0, 0
    references = False
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("## "):
            text = line[3:]
            if text.startswith(("附录", "参考资料")):
                story.append(PageBreak())
            if text.startswith("参考资料"):
                references = True
            p = Paragraph(inline(text), sty["h2"])
            p.toc_level = 0
            story.append(p)
            i += 1
            continue
        if line.startswith("### "):
            p = Paragraph(inline(line[4:]), sty["h3"])
            p.toc_level = 1
            story.append(p)
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            table = make_table(table_lines, sty)
            if "论文部分" in table_lines[0] and story and getattr(story[-1], "toc_level", None) == 1:
                heading = story.pop()
                story.append(KeepTogether([heading, Spacer(1, 3), table, Spacer(1, 10)]))
            else:
                story.extend([Spacer(1, 3), table, Spacer(1, 10)])
            continue
        match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if match:
            figure_no += 1
            source = (path.parent / match.group(2)).resolve()
            with PILImage.open(source) as im:
                w, h = im.size
            width = CONTENT_W
            height = width * h / w
            if height > 360:
                width *= 360 / height
                height = 360
            story.append(KeepTogether([Spacer(1, 4), Image(str(source), width=width, height=height),
                                        Paragraph(f"图 {figure_no}. {inline(match.group(1))}。数据来自三阶段正式独立评分。", sty["caption"])]))
            i += 1
            continue
        if line == "```math":
            i += 1
            formula = []
            while i < len(lines) and lines[i].strip() != "```":
                formula.append(lines[i].strip())
                i += 1
            math_no += 1
            story.extend([Spacer(1, 4), render_math(" ".join(formula), math_dir, math_no), Spacer(1, 9)])
            i += 1
            continue
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "![", "```")):
            para.append(lines[i].strip())
            i += 1
        story.append(Paragraph(inline(" ".join(para)), sty["reference" if references else "body"]))
    return story, math_no, figure_no


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parents[1] / "REPORT_ZH.md")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--font-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    for name, filename in [("NotoSC", "NotoSansSC-Regular.ttf"), ("NotoSC-Bold", "NotoSansSC-Bold.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(args.font_dir / filename)))
    pdfmetrics.registerFontFamily("NotoSC", normal="NotoSC", bold="NotoSC-Bold",
                                  italic="NotoSC", boldItalic="NotoSC-Bold")
    sty = styles()
    story = cover(sty)
    story.append(Paragraph("阅读导航", sty["h2"]))
    story.append(Paragraph("先看结论与第 7 节结果，再按需要阅读方法、实验协议和第 9-10 节的后续研究与发表条件。所有模型成绩均为实际运行；未来提案有明确标识。", sty["body"]))
    toc = TableOfContents()
    toc.levelStyles = [sty["toc0"], sty["toc1"]]
    toc.dotsMinLevel = 0
    story += [toc, PageBreak()]
    body, n_math, n_fig = build_markdown(args.report, sty, args.work_dir)
    story.extend(body)
    doc = ReportDoc(str(args.output), pagesize=A4, title="碳拍卖价格预测：模型创新研究、完整实验与发表路线",
                    author="Carbon Forecast Model Research", subject="EEX EUA auction forecasting; SDMC and FDMC; full development experiments",
                    pageCompression=1)
    doc.multiBuild(story)
    reader = PdfReader(args.output)
    page_text = [p.extract_text() or "" for p in reader.pages]
    receipt = {
        "report_source": str(args.report),
        "report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "output": str(args.output),
        "pdf_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "pdf_bytes": args.output.stat().st_size,
        "pages": len(reader.pages),
        "body_figures": n_fig,
        "math_formulas": n_math,
        "page_text_characters": [len(t) for t in page_text],
        "key_text_present": {s: any(s in t for t in page_text) for s in ["1.004985", "0.996836", "426", "345", "81", "2024-2025", "Journal of Forecasting", "9. 唯一建议"]},
        "visual_review": "PENDING_RENDER_INSPECTION",
        "note": "Text extraction verifies content presence only; visual review is performed separately."
    }
    (args.work_dir / "PDF_BUILD_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ["page_text_characters"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
