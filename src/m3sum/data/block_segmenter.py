from __future__ import annotations

import re
import sys
from pathlib import Path

from m3sum.data.schema import Block, FigureMeta
from m3sum.stage2_rerank.caption_refs import extract_caption_refs

BODY_IMG_RE = re.compile(r"!\[\]\(images/([a-f0-9]+)\.(?:jpg|png|jpeg)\)", re.I)


def segment_body(body_text: str) -> list[Block]:
    """Split MD body into ordered text/figure blocks with paragraph indices."""
    if not body_text.strip():
        return []

    raw_parts = re.split(r"\n\n+", body_text)
    blocks: list[Block] = []
    block_idx = 0
    char_cursor = 0

    for part in raw_parts:
        part = part.strip()
        if not part:
            continue

        subparts = _split_heading_blocks(part)
        for sub in subparts:
            sub = sub.strip()
            if not sub:
                continue

            img_match = BODY_IMG_RE.search(sub)
            if img_match and sub.strip().startswith("![]"):
                image_hash = img_match.group(1).lower()
                caption_line = sub.replace(img_match.group(0), "").strip()
                text = caption_line or sub
                caption_refs = extract_caption_refs(text)
                blocks.append(
                    Block(
                        block_id=f"b_{block_idx}",
                        block_idx=block_idx,
                        block_type="figure",
                        text=text,
                        char_start=char_cursor,
                        char_end=char_cursor + len(sub),
                        image_hash=image_hash,
                        caption_refs=caption_refs,
                        has_caption_ref=bool(caption_refs),
                    )
                )
                block_idx += 1
                char_cursor += len(sub) + 2
                continue

            caption_refs = extract_caption_refs(sub)
            blocks.append(
                Block(
                    block_id=f"b_{block_idx}",
                    block_idx=block_idx,
                    block_type="text",
                    text=sub,
                    char_start=char_cursor,
                    char_end=char_cursor + len(sub),
                    caption_refs=caption_refs,
                    has_caption_ref=bool(caption_refs),
                )
            )
            block_idx += 1
            char_cursor += len(sub) + 2

    return blocks


def _split_heading_blocks(part: str) -> list[str]:
    lines = part.split("\n")
    chunks: list[str] = []
    current: list[str] = []

    for line in lines:
        if line.startswith("# ") and current:
            chunks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        chunks.append("\n".join(current))
    return chunks if chunks else [part]


def enrich_figures(candidates: list, blocks: list[Block]) -> list[FigureMeta]:
    hash_to_pos: dict[str, int] = {}
    for b in blocks:
        if b.block_type == "figure" and b.image_hash:
            hash_to_pos[b.image_hash.lower()] = b.block_idx

    figures: list[FigureMeta] = []
    for c in candidates:
        pos = hash_to_pos.get(c.image_hash.lower(), -1)
        figures.append(
            FigureMeta(
                image_hash=c.image_hash,
                caption=c.caption,
                source_type=c.source_type,
                pos=pos,
                page_idx=c.page_idx,
                body_order=c.body_order,
                abs_image_path=c.abs_image_path,
                img_path=c.img_path,
            )
        )
    return figures
