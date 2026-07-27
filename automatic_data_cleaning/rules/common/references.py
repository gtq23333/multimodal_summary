from __future__ import annotations

from rules.common.patterns import REFERENCES_HEADING_RE


def truncate_at_references(lines: list[str]) -> list[str]:
    for i, line in enumerate(lines):
        if REFERENCES_HEADING_RE.match(line.strip()):
            return lines[:i]
    return lines
