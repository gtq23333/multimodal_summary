from __future__ import annotations

from m3sum.data.schema import FigureMeta


def select_body_caption_figures(figures: list[FigureMeta]) -> list[FigureMeta]:
    """
    筛选正文中带非空图注的图片，与 data_annotation filters/body_with_caption 一致：
    in_body_md（pos >= 0）且 caption.strip() 非空。
    """
    return [f for f in figures if (f.caption or "").strip() and f.pos >= 0]
