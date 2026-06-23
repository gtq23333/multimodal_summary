from __future__ import annotations

import re

FigureRef = tuple[int, ...]
LabeledFigureRef = tuple[str, FigureRef]

_REF_RE = re.compile(
    r"(?:(?:见|如|参见|参考|详见)\s*)?"
    r"(?P<label>图|表|figure|fig\.?|table)"
    r"\s*"
    r"(?P<number>\d+(?:\s*\.\s*\d+)*)",
    re.IGNORECASE,
)


def parse_ref_number(text: str) -> FigureRef | None:
    """从编号文本中解析结构化图号，使用整数元组避免 float 精度歧义。"""
    if not text:
        return None
    cleaned = re.sub(r"\s+", "", text)
    if not re.fullmatch(r"\d+(?:\.\d+)*", cleaned):
        return None
    parts = tuple(int(part) for part in cleaned.split(".") if part != "")
    return parts or None


def extract_caption_refs(text: str) -> list[FigureRef]:
    """抽取正文中显式提到的图/表编号，去重并保持出现顺序。"""
    refs: list[FigureRef] = []
    seen: set[FigureRef] = set()
    for match in _REF_RE.finditer(text or ""):
        ref = parse_ref_number(match.group("number"))
        if ref is None or ref in seen:
            continue
        seen.add(ref)
        refs.append(ref)
    return refs


def _normalize_label(label: str) -> str:
    normalized = label.lower().rstrip(".")
    if normalized in {"图", "figure", "fig"}:
        return "figure"
    if normalized in {"表", "table"}:
        return "table"
    return normalized


def extract_labeled_caption_refs(text: str) -> list[LabeledFigureRef]:
    """抽取带图/表类型的编号，避免图2与表2交叉误匹配。"""
    refs: list[LabeledFigureRef] = []
    seen: set[LabeledFigureRef] = set()
    for match in _REF_RE.finditer(text or ""):
        ref = parse_ref_number(match.group("number"))
        if ref is None:
            continue
        item = (_normalize_label(match.group("label")), ref)
        if item in seen:
            continue
        seen.add(item)
        refs.append(item)
    return refs


def parse_figure_label_from_caption(caption: str) -> str | None:
    """从 caption 中解析图/表类型标签。"""
    labeled = extract_labeled_caption_refs(caption)
    return labeled[0][0] if labeled else None


def figure_ref_to_str(ref: FigureRef | None) -> str | None:
    """将内部图号元组转换为可读字符串。"""
    if ref is None:
        return None
    return ".".join(str(part) for part in ref)


def parse_figure_index_from_caption(caption: str) -> FigureRef | None:
    """从 figure caption 中解析该图/表自身编号。"""
    refs = extract_caption_refs(caption)
    return refs[0] if refs else None
