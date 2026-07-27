from __future__ import annotations

import re

from core.types import CleanResult
from rules.national_competition.abstract import (
    abstract_text,
    extract_abstract_lines,
    find_abstract_end_index,
)
from rules.national_competition.body_anchor import extract_body_lines
from rules.national_competition.compose import compose
from rules.national_competition.dedupe_stub import dedupe_11_stubs
from rules.national_competition.front_matter import (
    find_abstract_heading,
    strip_document_title,
)
from rules.national_competition.keywords import is_keywords_line
from rules.national_competition.patterns import MD_HEAD, TOC_HEADING_RE
from rules.national_competition.toc import strip_toc_block


MIN_ABSTRACT_LEN = 80
MIN_BODY_LEN = 300


def _skip_keywords_prefix(lines: list[str]) -> list[str]:
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if is_keywords_line(s):
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue
        break
    return lines[i:]


def _validate(abstract: str, body: str) -> str | None:
    plain_abs = re.sub(r"\s+", "", abstract)
    plain_body = re.sub(r"\s+", "", body)
    if len(plain_abs) < MIN_ABSTRACT_LEN:
        return f"abstract_too_short:{len(plain_abs)}"
    if len(plain_body) < MIN_BODY_LEN:
        return f"body_too_short:{len(plain_body)}"
    if TOC_HEADING_RE.search(body) or re.search(
        rf"^{MD_HEAD}目\s*录\s*$", body, re.M
    ):
        return "toc_remain_in_body"
    return None


class NationalCompetitionProfile:
    name = "national_competition"

    def clean(self, raw_md: str, *, paper_id: str, separator: str) -> CleanResult:
        lines = raw_md.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        lines, _ = strip_document_title(lines)

        abstract_idx = find_abstract_heading(lines)
        if abstract_idx is None:
            return CleanResult(success=False, reason="abstract_heading_not_found")

        abs_paragraphs = extract_abstract_lines(lines, abstract_idx)
        abstract = abstract_text(abs_paragraphs)

        end_idx = find_abstract_end_index(lines, abstract_idx)
        post_lines = lines[end_idx:]

        post_lines = _skip_keywords_prefix(post_lines)
        post_lines, _ = strip_toc_block(post_lines, 0)
        body_lines = extract_body_lines(post_lines)
        body_lines = dedupe_11_stubs(body_lines)
        body = "\n".join(body_lines).strip()

        err = _validate(abstract, body)
        if err:
            return CleanResult(
                success=False,
                reason=err,
                abstract=abstract,
                body=body,
                meta={"paper_id": paper_id},
            )

        content = compose(abstract, body, separator=separator)
        return CleanResult(
            success=True,
            content=content,
            abstract=abstract,
            body=body,
            meta={"paper_id": paper_id},
        )
