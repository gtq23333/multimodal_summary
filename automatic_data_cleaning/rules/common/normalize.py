from __future__ import annotations


def split_lines(raw_md: str) -> list[str]:
    return raw_md.replace("\r\n", "\n").replace("\r", "\n").split("\n")
