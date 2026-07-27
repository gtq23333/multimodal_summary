from __future__ import annotations

import re

from rules.common.patterns import (
    DOT_LEADER_RE,
    TOC_HEADING_RE,
    MD_HEAD,
)

MIN_ABSTRACT_LEN = 80
MAX_ABSTRACT_LEN = 2500
MIN_BODY_LEN = 300
MAX_TOC_DOT_RATIO = 0.12


def _plain_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _count_toc_like_lines(lines: list[str]) -> int:
    count = 0
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if TOC_HEADING_RE.match(s):
            count += 1
            continue
        if DOT_LEADER_RE.search(s):
            count += 1
            continue
        if re.search(r"\s+\d{1,3}\s*$", s) and len(s) < 120 and s.count("  ") >= 1:
            count += 1
    return count


def validate_abstract_body(
    abstract: str,
    body: str,
    *,
    min_abstract_len: int = MIN_ABSTRACT_LEN,
    max_abstract_len: int = MAX_ABSTRACT_LEN,
    min_body_len: int = MIN_BODY_LEN,
) -> str | None:
    plain_abs = _plain_len(abstract)
    plain_body = _plain_len(body)

    if plain_abs < min_abstract_len:
        return f"abstract_too_short:{plain_abs}"
    if plain_abs > max_abstract_len:
        return f"abstract_too_long:{plain_abs}"
    if plain_body < min_body_len:
        return f"body_too_short:{plain_body}"

    body_lines = [ln for ln in body.splitlines() if ln.strip()]
    if body_lines:
        toc_hits = _count_toc_like_lines(body_lines)
        if toc_hits / len(body_lines) > MAX_TOC_DOT_RATIO:
            return f"toc_dot_lines:{toc_hits}"

    if TOC_HEADING_RE.search(body) or re.search(
        rf"^{MD_HEAD}目\s*录\s*$", body, re.M
    ):
        return "toc_remain_in_body"

    if "承诺书" in body[:500]:
        return "commitment_in_body"

    return None
