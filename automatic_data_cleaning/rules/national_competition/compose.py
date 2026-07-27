from __future__ import annotations

from rules.national_competition.patterns import DEFAULT_SEPARATOR


def compose(abstract: str, body: str, separator: str = DEFAULT_SEPARATOR) -> str:
    return f"{abstract.strip()}\n\n{separator}\n\n{body.strip()}\n"
