from __future__ import annotations

import re
from dataclasses import dataclass

from m3sum.data.schema import Block
from m3sum.stage2_rerank.caption_refs import figure_ref_to_str


@dataclass
class CaptionMatchResult:
    matched_blocks: list[Block]
    all_refs: list[str]


def compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in patterns]


def match_caption_blocks(
    blocks: list[Block],
    patterns: list[str] | None = None,
) -> CaptionMatchResult:
    matched: list[Block] = []
    all_refs: list[str] = []

    for block in blocks:
        if block.block_type != "text":
            continue
        refs = [ref for ref in block.caption_refs if ref]
        if refs:
            matched.append(block)
            all_refs.extend(f"图{figure_ref_to_str(ref)}" for ref in refs)

    return CaptionMatchResult(matched_blocks=matched, all_refs=all_refs)
