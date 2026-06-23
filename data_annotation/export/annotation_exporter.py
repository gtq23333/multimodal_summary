from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models.paper import Paper
from models.segments import Segment, plain_text_from_segments


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def build_annotation(
    paper: Paper,
    segments: list[Segment],
    original_abstract: str,
    filter_strategy: str,
    context_window_chars: int,
    tool_version: str,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    edited_text = plain_text_from_segments(segments)
    insertions = []
    insertion_order = 0

    char_offset = 0
    for seg in segments:
        if seg.type == "text":
            char_offset += len(seg.content)
        else:
            candidate = seg.candidate
            char_index = char_offset
            before = edited_text[max(0, char_index - context_window_chars):char_index]
            after = edited_text[char_index:char_index + context_window_chars]
            insertions.append(
                {
                    "insertion_id": f"ins_{insertion_order + 1:03d}",
                    "order": insertion_order,
                    "image_hash": candidate.image_hash,
                    "image_filename": candidate.image_filename,
                    "img_path": candidate.img_path,
                    "source_type": candidate.source_type,
                    "caption": candidate.caption,
                    "body_order": candidate.body_order,
                    "position": {
                        "char_index": char_index,
                        "index_basis": "edited_plain_text",
                        "context_window_chars": context_window_chars,
                        "text_before": before,
                        "text_after": after,
                    },
                }
            )
            insertion_order += 1

    multimodal_sequence: list[dict[str, Any]] = []
    for seg in segments:
        if seg.type == "text":
            if seg.content:
                multimodal_sequence.append({"type": "text", "content": seg.content})
        else:
            c = seg.candidate
            multimodal_sequence.append(
                {
                    "type": "image",
                    "image_hash": c.image_hash,
                    "image_filename": c.image_filename,
                    "img_path": c.img_path,
                    "source_type": c.source_type,
                    "caption": c.caption,
                }
            )

    if not multimodal_sequence and edited_text:
        multimodal_sequence.append({"type": "text", "content": edited_text})

    created_at = existing.get("annotation_meta", {}).get("created_at", _now_iso()) if existing else _now_iso()

    return {
        "schema_version": "1.0.0",
        "paper_id": paper.paper_id,
        "source": {
            "md_path": str(paper.md_path),
            "content_list_path": str(paper.content_list_path),
            "mineru_dir": str(paper.mineru_dir),
        },
        "abstract": {
            "original_text": original_abstract,
            "edited_text": edited_text,
            "text_modified": edited_text != original_abstract,
        },
        "annotation_meta": {
            "annotator": existing.get("annotation_meta", {}).get("annotator", "") if existing else "",
            "created_at": created_at,
            "updated_at": _now_iso(),
            "tool_version": tool_version,
            "filter_strategy": filter_strategy,
            "context_window_chars": context_window_chars,
        },
        "insertions": insertions,
        "multimodal_sequence": multimodal_sequence,
    }


def export_annotation(
    output_dir: Path,
    paper: Paper,
    segments: list[Segment],
    original_abstract: str,
    filter_strategy: str,
    context_window_chars: int,
    tool_version: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{paper.paper_id}.json"
    existing = None
    if out_path.is_file():
        existing = json.loads(out_path.read_text(encoding="utf-8"))

    data = build_annotation(
        paper=paper,
        segments=segments,
        original_abstract=original_abstract,
        filter_strategy=filter_strategy,
        context_window_chars=context_window_chars,
        tool_version=tool_version,
        existing=existing,
    )
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def load_annotation(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
