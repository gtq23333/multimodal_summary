from __future__ import annotations

import re

_WATERMARK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"公众号\s*[:：]?\s*建模忠哥"),
    re.compile(r"获取更多资源"),
    re.compile(r"QQ群\s*[:：]?\s*966535540"),
    re.compile(r"建模忠哥"),
    re.compile(r"www\.madio\.net", re.I),
    re.compile(r"数学中国网站"),
]

_WATERMARK_LINE_RE = re.compile(
    r"^(?:公众号|获取更多资源|QQ群|建模忠哥|数学中国网站)",
    re.I,
)


def strip_watermarks_text(text: str) -> str:
    for pat in _WATERMARK_PATTERNS:
        text = pat.sub("", text)
    return text


def is_watermark_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _WATERMARK_LINE_RE.match(s):
        return True
    for pat in _WATERMARK_PATTERNS:
        if pat.search(s) and len(s) < 120:
            return True
    return False
