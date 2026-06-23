from __future__ import annotations

import json
from pathlib import Path

from models.paper import ImageCandidate


def find_content_list_path(mineru_dir: Path) -> Path | None:
    if not mineru_dir.is_dir():
        return None
    matches = list(mineru_dir.glob("*_content_list.json"))
    if not matches:
        return None
    if len(matches) > 1:
        matches.sort(key=lambda p: p.name)
    return matches[0]


def _extract_caption(item: dict) -> tuple[str, list[str]]:
    source_type = item.get("type", "")
    if source_type == "image":
        raw = item.get("img_caption") or []
    elif source_type == "table":
        raw = item.get("table_caption") or []
    else:
        return "", []
    if not isinstance(raw, list):
        raw = [str(raw)] if raw else []
    caption = " ".join(s.strip() for s in raw if s and str(s).strip())
    return caption, [str(s) for s in raw]


def load_candidates_from_content_list(
    content_list_path: Path,
    mineru_dir: Path,
    body_hashes: set[str],
) -> list[ImageCandidate]:
    data = json.loads(content_list_path.read_text(encoding="utf-8"))
    candidates: list[ImageCandidate] = []
    order = 0

    for item in data:
        item_type = item.get("type")
        if item_type not in ("image", "table"):
            continue
        img_path = item.get("img_path")
        if not img_path:
            continue

        rel_path = img_path.replace("\\", "/")
        filename = Path(rel_path).name
        image_hash = Path(filename).stem
        caption, captions_raw = _extract_caption(item)
        abs_image_path = str((mineru_dir / rel_path).resolve())

        candidates.append(
            ImageCandidate(
                image_hash=image_hash,
                image_filename=filename,
                img_path=rel_path,
                abs_image_path=abs_image_path,
                source_type=item_type,
                caption=caption,
                captions_raw=captions_raw,
                page_idx=item.get("page_idx"),
                content_list_order=order,
                in_body_md=image_hash.lower() in {h.lower() for h in body_hashes},
            )
        )
        order += 1

    body_ordered = [c for c in candidates if c.in_body_md]
    hash_to_body_order = {c.image_hash: i for i, c in enumerate(body_ordered)}
    for c in candidates:
        c.body_order = hash_to_body_order.get(c.image_hash, -1)

    return candidates
