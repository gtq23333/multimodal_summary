from __future__ import annotations

import re

from rules.national_competition.patterns import (
    BODY_SECTION_ONE_RE,
    KEYWORDS_RE,
    SECTION_11_HEADER_RE,
    SECTION_11_RE,
    SECTION_CN_RE,
    SECTION_NUM_RE,
    TOC_HEADING_RE,
)
from rules.national_competition.toc import _is_body_section_start


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


def find_body_start_index(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if _is_body_section_start(line.strip()):
            return _advance_past_headers(lines, i + 1)

    for i, line in enumerate(lines):
        s = line.strip()
        if SECTION_NUM_RE.match(s) and ("问题" in s or "背景" in s):
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


def strip_redundant_headers(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if SECTION_11_HEADER_RE.match(s) and _plain_len(s) < 25:
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if SECTION_11_RE.match(nxt) and len(nxt) > len(s) * 2:
                    i += 1
                    continue
        out.append(lines[i])
        i += 1
    return out


def extract_body_lines(lines: list[str]) -> list[str]:
    idx = find_body_start_index(lines)
    body = lines[idx:]
    return strip_redundant_headers(body)
