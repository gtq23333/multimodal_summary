from __future__ import annotations

import re

from rules.national_competition.patterns import (
    BODY_SECTION_ONE_RE,
    PAGE_REF_RE,
    PAGE_TAIL_RE,
    SECTION_CN_RE,
    SECTION_NUM_RE,
    TOC_HEADING_RE,
)


def _is_toc_line(line: str) -> bool:
    s = line.strip()
    if not s or s.startswith("$$") or s.startswith("$"):
        return False
    if TOC_HEADING_RE.match(s):
        return True
    if len(s) >= 120:
        return False
    if PAGE_TAIL_RE.search(s) and not s.startswith("#"):
        return True
    if PAGE_REF_RE.search(s) and len(s) < 120 and s.count("  ") >= 1:
        return True
    page_hits = re.findall(r"\s+\d+\s+", f" {s} ")
    if len(page_hits) >= 2 and len(s) < 200:
        return True
    return False


def _is_body_section_start(line: str) -> bool:
    s = line.strip()
    if BODY_SECTION_ONE_RE.match(s):
        return True
    if SECTION_CN_RE.match(s) and "问题" in s:
        return True
    if SECTION_NUM_RE.match(s) and ("问题" in s or "背景" in s or "重述" in s):
        return True
    return False


def strip_toc_block(lines: list[str], start_idx: int) -> tuple[list[str], int]:
    if start_idx >= len(lines):
        return lines, start_idx

    i = start_idx
    in_toc = False

    if TOC_HEADING_RE.match(lines[i].strip()):
        in_toc = True
        i += 1

    toc_run = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue

        if _is_body_section_start(s):
            return lines[i:], 0

        if TOC_HEADING_RE.match(s):
            in_toc = True
            i += 1
            continue

        if in_toc or _is_toc_line(s):
            in_toc = True
            toc_run += 1
            i += 1
            continue

        break

    return lines[i:], 0
