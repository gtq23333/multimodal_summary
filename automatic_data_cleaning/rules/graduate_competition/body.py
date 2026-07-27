from __future__ import annotations

import re

from rules.common.patterns import (
    BODY_SECTION_NUM_RE,
    BODY_SECTION_ONE_RE,
    BODY_SECTION_PLAIN_RE,
    KEYWORDS_RE,
    SECTION_11_HEADER_RE,
    SECTION_11_RE,
    SECTION_CN_RE,
    SECTION_NUM_RE,
    TOC_HEADING_RE,
)
from rules.national_competition.toc import _is_body_section_start, strip_toc_block


def _plain_len(line: str) -> int:
    s = re.sub(r"\$[^$]+\$", "", line)
    s = re.sub(r"[#\s]", "", s)
    return len(s)


def _is_substantial(line: str, min_len: int = 80) -> bool:
    s = line.strip()
    if not s or s.startswith("#"):
        return False
    if KEYWORDS_RE.match(s) or TOC_HEADING_RE.match(s):
        return False
    return _plain_len(s) >= min_len


def _is_header_only(line: str) -> bool:
    s = line.strip()
    if not s.startswith("#"):
        return False
    if SECTION_11_HEADER_RE.match(s):
        return True
    if SECTION_CN_RE.match(s) and _plain_len(s) < 20:
        return True
    if SECTION_NUM_RE.match(s) and _plain_len(s) < 20:
        return True
    return False


def _advance_past_headers(lines: list[str], start: int) -> int:
    i = start
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if _is_header_only(s):
            i += 1
            continue
        if SECTION_11_RE.match(s) or _is_substantial(lines[i], min_len=40):
            return i
        i += 1
    return start


def _is_body_anchor(line: str) -> bool:
    s = line.strip()
    if _is_body_section_start(s):
        return True
    if BODY_SECTION_ONE_RE.match(s):
        return True
    if BODY_SECTION_NUM_RE.match(s):
        return True
    if BODY_SECTION_PLAIN_RE.match(s):
        return True
    if SECTION_NUM_RE.match(s) and ("问题" in s or "背景" in s or "重述" in s):
        return True
    return False


def find_body_start_index(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if _is_body_anchor(line.strip()):
            return _advance_past_headers(lines, i + 1)

    for i, line in enumerate(lines):
        if SECTION_11_HEADER_RE.match(line.strip()) or SECTION_11_RE.match(line.strip()):
            idx = _advance_past_headers(lines, i)
            if idx > i or _is_substantial(lines[i], min_len=40):
                return idx

    for i, line in enumerate(lines):
        if _is_substantial(line, min_len=100):
            return i

    for i, line in enumerate(lines):
        if _is_substantial(line, min_len=80):
            return i

    return 0


def extract_body_lines(lines: list[str]) -> list[str]:
    lines, _ = strip_toc_block(lines, 0)
    idx = find_body_start_index(lines)
    return lines[idx:]
