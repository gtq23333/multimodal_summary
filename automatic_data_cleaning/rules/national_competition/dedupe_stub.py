from __future__ import annotations

import re

from rules.national_competition.patterns import SECTION_11_HEADER_RE, SECTION_11_RE


def dedupe_11_stubs(lines: list[str]) -> list[str]:
    if len(lines) < 2:
        return lines

    out: list[str] = []
    i = 0
    while i < len(lines):
        if i + 1 < len(lines):
            a = lines[i].strip()
            b = lines[i + 1].strip()
            a_is_11 = bool(SECTION_11_RE.match(a) or SECTION_11_HEADER_RE.match(a))
            b_is_11 = bool(SECTION_11_RE.match(b))
            if a_is_11 and b_is_11 and len(a) < 120 and len(b) > len(a) * 2:
                i += 1
                continue
        out.append(lines[i])
        i += 1
    return out
