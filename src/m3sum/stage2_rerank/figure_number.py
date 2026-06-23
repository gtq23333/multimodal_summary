from __future__ import annotations

import re

from m3sum.data.schema import FigureMeta

_FIGURE_PATTERNS = [
    re.compile(r"图\s*(\d+(?:\.\d+)?)"),
    re.compile(r"表\s*(\d+(?:\.\d+)?)"),
    re.compile(r"(?i)figure\s*(\d+(?:\.\d+)?)"),
    re.compile(r"(?i)table\s*(\d+(?:\.\d+)?)"),
]


def parse_figure_number(caption: str) -> float | None:
    """从图注中解析图表编号；解析失败返回 None。"""
    if not caption:
        return None
    for pattern in _FIGURE_PATTERNS:
        match = pattern.search(caption)
        if match:
            return float(match.group(1))
    return None


def layout_sort_key(figure: FigureMeta) -> tuple[int, float, int]:
    """
    Layout-Order 排序键：(是否有编号, 编号, body_order)。
    无编号时编号视为 inf，回退到 body_order。
    """
    parsed = parse_figure_number(figure.caption)
    if parsed is not None:
        return (0, parsed, figure.body_order)
    return (1, float("inf"), figure.body_order)
