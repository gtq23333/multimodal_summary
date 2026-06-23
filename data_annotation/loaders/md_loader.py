from __future__ import annotations

import re
from pathlib import Path

BODY_IMG_RE = re.compile(r"!\[\]\(images/([a-f0-9]+)\.(?:jpg|png|jpeg)\)", re.I)


def parse_md(md_path: Path, separator: str) -> tuple[str, str, set[str]]:
    text = md_path.read_text(encoding="utf-8")
    if separator in text:
        abstract_text, body_text = text.split(separator, 1)
        abstract_text = abstract_text.strip()
        body_text = body_text.strip()
    else:
        abstract_text = text.strip()
        body_text = ""

    body_hashes = set(BODY_IMG_RE.findall(body_text))
    return abstract_text, body_text, body_hashes
