from __future__ import annotations

from export.annotation_exporter import load_annotation
from models.paper import ImageCandidate, Paper
from models.segments import ImageSegment, Segment, TextSegment, merge_adjacent_text_segments


def segments_from_annotation(paper: Paper, annotation: dict) -> list[Segment]:
    candidate_map = {c.image_hash: c for c in paper.all_candidates}
    for c in paper.filtered_candidates:
        candidate_map[c.image_hash] = c

    sequence = annotation.get("multimodal_sequence", [])
    if not sequence:
        edited = annotation.get("abstract", {}).get("edited_text", paper.abstract_text)
        return [TextSegment(content=edited)]

    segments: list[Segment] = []
    for item in sequence:
        if item.get("type") == "text":
            segments.append(TextSegment(content=item.get("content", "")))
        elif item.get("type") == "image":
            image_hash = item.get("image_hash", "")
            candidate = candidate_map.get(image_hash)
            if candidate is None:
                candidate = ImageCandidate(
                    image_hash=image_hash,
                    image_filename=item.get("image_filename", f"{image_hash}.jpg"),
                    img_path=item.get("img_path", f"images/{image_hash}.jpg"),
                    abs_image_path=str((paper.mineru_dir / item.get("img_path", f"images/{image_hash}.jpg")).resolve()),
                    source_type=item.get("source_type", "image"),
                    caption=item.get("caption", ""),
                    captions_raw=[item.get("caption", "")],
                    page_idx=None,
                    content_list_order=-1,
                    in_body_md=True,
                )
            segments.append(ImageSegment(candidate=candidate))
    return merge_adjacent_text_segments(segments)


def used_image_hashes(segments: list[Segment]) -> set[str]:
    return {s.candidate.image_hash for s in segments if s.type == "image"}


def restore_from_file(paper: Paper, annotation_path) -> list[Segment]:
    annotation = load_annotation(annotation_path)
    return segments_from_annotation(paper, annotation)
