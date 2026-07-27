from __future__ import annotations

import re

from rules.common.patterns import (
    ABSTRACT_END_SECTION_RE,
    ABSTRACT_HEADING_RE,
    ABSTRACT_INLINE_RE,
    BODY_SECTION_NUM_RE,
    BODY_SECTION_ONE_RE,
    BODY_SECTION_PLAIN_RE,
    ENGLISH_ABSTRACT_RE,
    KEYWORDS_RE,
    SECTION_CN_RE,
    SECTION_NUM_RE,
    TOC_HEADING_RE,
)


def _is_chinese_body_start(line: str) -> bool:
    s = line.strip()
    if BODY_SECTION_ONE_RE.match(s):
        return True
    if BODY_SECTION_NUM_RE.match(s):
        return True
    if BODY_SECTION_PLAIN_RE.match(s):
        return True
    if SECTION_CN_RE.match(s) and "问题" in s:
        return True
    if SECTION_NUM_RE.match(s) and ("问题" in s or "重述" in s or "背景" in s):
        return True
    return False


def strip_front_matter(lines: list[str]) -> list[str]:
    """跳过承诺书、编号专用页等前置内容，定位到摘要或正文起点。"""
    abstract_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        if ABSTRACT_HEADING_RE.match(s) or ABSTRACT_INLINE_RE.match(s):
            abstract_idx = i
            break

    if abstract_idx is not None:
        return lines[abstract_idx:]

    for i, line in enumerate(lines):
        s = line.strip()
        if re.search(r"承诺书", s) or re.search(r"编号专用页", s):
            continue
        if SECTION_CN_RE.match(s) and "问题" in s:
            return lines[i:]
        if SECTION_NUM_RE.match(s) and ("问题" in s or "重述" in s):
            return lines[i:]
        if s.startswith("#") and re.search(r"问题重述|问题的重述|问题提出", s):
            return lines[i:]

    return lines


def strip_english_abstract_block(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if ENGLISH_ABSTRACT_RE.match(s):
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if not nxt:
                    i += 1
                    continue
                if _is_chinese_body_start(nxt) or TOC_HEADING_RE.match(nxt):
                    break
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return out


def _is_abstract_end(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if KEYWORDS_RE.match(s):
        return True
    if TOC_HEADING_RE.match(s):
        return True
    if SECTION_CN_RE.match(s) and ("问题" in s or "目录" in s):
        return True
    if SECTION_NUM_RE.match(s) and ("问题" in s or "背景" in s or "重述" in s):
        return True
    if ENGLISH_ABSTRACT_RE.match(s):
        return True
    if ABSTRACT_END_SECTION_RE.match(s):
        return True
    return False


def find_abstract_heading(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        s = line.strip()
        if ABSTRACT_HEADING_RE.match(s) or ABSTRACT_INLINE_RE.match(s):
            return i
    return None


def extract_abstract_lines(lines: list[str], abstract_heading_idx: int) -> list[str]:
    first = lines[abstract_heading_idx].strip()
    collected: list[str] = []
    start = abstract_heading_idx + 1

    if ABSTRACT_INLINE_RE.match(first):
        tail = re.sub(r"^摘\s*要\s*[:：]?\s*", "", first).strip()
        if tail:
            collected.append(tail)

    for line in lines[start:]:
        s = line.strip()
        if _is_abstract_end(s):
            break
        if s.startswith("#") and SECTION_CN_RE.match(s):
            break
        if s.startswith("#") and len(s) < 30:
            continue
        collected.append(line.rstrip())

    return _merge_paragraphs(collected)


def find_abstract_end_index(lines: list[str], abstract_heading_idx: int) -> int:
    start = abstract_heading_idx + 1
    for i in range(start, len(lines)):
        if _is_abstract_end(lines[i].strip()):
            return i
    return len(lines)


def _merge_paragraphs(raw_lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    buf: list[str] = []
    for line in raw_lines:
        if not line.strip():
            if buf:
                paragraphs.append("\n".join(buf).strip())
                buf = []
            continue
        if line.strip().startswith("#"):
            if buf:
                paragraphs.append("\n".join(buf).strip())
                buf = []
            continue
        buf.append(line.rstrip())
    if buf:
        paragraphs.append("\n".join(buf).strip())
    return [p for p in paragraphs if p]


def abstract_text(paragraphs: list[str]) -> str:
    return "\n\n".join(paragraphs).strip()
