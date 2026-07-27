from __future__ import annotations

from rules.national_competition.patterns import (
    ABSTRACT_HEADING_RE,
    ABSTRACT_INLINE_RE,
)


def strip_document_title(lines: list[str]) -> tuple[list[str], int | None]:
    """若文首 # 标题后 5 行内出现摘要标记，删除标题行。"""
    if not lines:
        return lines, None

    abstract_idx: int | None = None
    for i, line in enumerate(lines[:20]):
        if find_abstract_heading_index_at(lines, i) is not None:
            abstract_idx = i
            break

    if abstract_idx is None:
        return lines, None

    first = lines[0].strip()
    if first.startswith("#") and not ABSTRACT_HEADING_RE.match(first):
        if abstract_idx <= 5:
            new_lines = lines[1:]
            while new_lines and not new_lines[0].strip():
                new_lines = new_lines[1:]
            return new_lines, max(0, abstract_idx - 1)

    return lines, abstract_idx


def find_abstract_heading_index_at(lines: list[str], idx: int) -> int | None:
    line = lines[idx].strip()
    if ABSTRACT_HEADING_RE.match(line):
        return idx
    if ABSTRACT_INLINE_RE.match(line):
        return idx
    return None


def find_abstract_heading(lines: list[str]) -> int | None:
    for i in range(len(lines)):
        if find_abstract_heading_index_at(lines, i) is not None:
            return i
    return None
