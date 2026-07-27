from __future__ import annotations

import re

from rules.national_competition.patterns import KEYWORDS_RE

_KEYWORDS_EXTENDED_RE = re.compile(
    r"^(?:#{1,3}\s*)?(?:关键词|关键字|\[关键词\])"
)


def is_keywords_line(line: str) -> bool:
    s = line.strip()
    return bool(KEYWORDS_RE.match(s) or _KEYWORDS_EXTENDED_RE.match(s))
