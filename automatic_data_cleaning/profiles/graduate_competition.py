from __future__ import annotations

from core.types import CleanResult
from rules.common.noise import strip_noise_lines
from rules.common.normalize import split_lines
from rules.common.references import truncate_at_references
from rules.common.validate import validate_abstract_body
from rules.common.watermark import strip_watermarks_text
from rules.graduate_competition.abstract import (
    abstract_text,
    extract_abstract_lines,
    find_abstract_end_index,
    find_abstract_heading,
)
from rules.graduate_competition.body import extract_body_lines
from rules.national_competition.compose import compose
from rules.national_competition.dedupe_stub import dedupe_11_stubs
from rules.national_competition.keywords import is_keywords_line


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


class GraduateCompetitionProfile:
    name = "graduate_competition"

    def clean(self, raw_md: str, *, paper_id: str, separator: str) -> CleanResult:
        raw_md = strip_watermarks_text(raw_md)
        lines = split_lines(raw_md)

        abstract_idx = find_abstract_heading(lines)
        if abstract_idx is None:
            return CleanResult(success=False, reason="abstract_heading_not_found")

        abs_paragraphs = extract_abstract_lines(lines, abstract_idx)
        abstract = abstract_text(abs_paragraphs)

        end_idx = find_abstract_end_index(lines, abstract_idx)
        post_lines = lines[end_idx:]
        post_lines = _skip_keywords_prefix(post_lines)

        body_lines = extract_body_lines(post_lines)
        body_lines = truncate_at_references(body_lines)
        body_lines = dedupe_11_stubs(body_lines)
        body_lines = strip_noise_lines(body_lines, profile=self.name)
        body = "\n".join(body_lines).strip()

        err = validate_abstract_body(abstract, body)
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
