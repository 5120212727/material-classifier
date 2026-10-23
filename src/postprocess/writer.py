"""输出层：结果落盘。

两种模式：
- standard：独立结果表，列固定，便于评测与交付
- overlay ：回写到原始清单的副本上（对齐甲方已有模板的 R–X 列），价格列留空
"""
from __future__ import annotations

import csv
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from ..schema import Item, Prediction

HEADERS = [
    ("序号", 6), ("页签", 10), ("源行号", 8), ("物料名称", 30), ("型号、规格及技术性能", 46),
    ("预测大类", 16), ("预测中类", 18), ("预测小类", 18),
    ("置信度", 9), ("置信档", 8), ("判定策略", 18), ("判定依据", 52), ("最近邻相似度", 11),
]


def write_standard(
    items: list[Item],
    preds: list[Prediction],
    out_path: str | Path,
    also_csv: bool = True,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "匹配结果"
    _write_sheet(ws, items, preds)
    wb.save(out_path)

    if also_csv:
        cp = out_path.with_suffix(".csv")
        with open(cp, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow([h for h, _ in HEADERS])
            for i, (it, p) in enumerate(zip(items, preds), 1):
                w.writerow(_row(i, it, p))
    return out_path


def _row(i: int, it: Item, p: Prediction) -> list:
    sim = p.candidates[0].score if p.candidates else 0.0
    return [
        i, it.sheet, it.idx, it.name, it.spec,
        p.b, p.m, p.s,
        round(p.confidence, 3), p.band, p.strategy, p.reason, round(sim, 3),
    ]


def _write_sheet(ws, items: list[Item], preds: list[Prediction]) -> None:
    hf = PatternFill("solid", fgColor="1A3A5C")
    for c, (h, w) in enumerate(HEADERS, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font = Font(color="FFFFFF", bold=True, size=10)
        cell.fill = hf
        cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.freeze_panes = "A2"

    band_fill = {
        "高": PatternFill("solid", fgColor="E3F3E9"),
        "中": PatternFill("solid", fgColor="FDF3E0"),
        "低": PatternFill("solid", fgColor="FDE8E5"),
    }
    for r, (it, p) in enumerate(zip(items, preds), 2):
        for c, v in enumerate(_row(r - 1, it, p), 1):
            cell = ws.cell(row=r, column=c, value=v)
            cell.alignment = Alignment(vertical="top", wrap_text=(c in (4, 5, 12)))
            cell.font = Font(size=10)
        ws.cell(row=r, column=10).fill = band_fill.get(p.band, PatternFill())
        ws.row_dimensions[r].height = 28


def write_overlay(
    src_path: str | Path,
    items: list[Item],
    preds: list[Prediction],
    out_path: str | Path,
    columns: dict | None = None,
) -> Path:
    """回写到原清单副本。columns 可指定每个字段落在哪一列（1-based）。

    默认按我方标准列写在新 sheet，避免破坏甲方原表结构。
    """
    out_path = Path(out_path)
    wb = openpyxl.load_workbook(src_path)
    ws = wb.create_sheet("匹配结果")
    _write_sheet(ws, items, preds)
    wb.save(out_path)
    return out_path
