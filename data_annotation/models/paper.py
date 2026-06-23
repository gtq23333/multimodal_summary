from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ImageCandidate:
    image_hash: str
    image_filename: str
    img_path: str
    abs_image_path: str
    source_type: str
    caption: str
    captions_raw: list[str]
    page_idx: Optional[int]
    content_list_order: int
    body_order: int = 0
    in_body_md: bool = False

    @property
    def is_table(self) -> bool:
        return self.source_type == "table"


@dataclass
class Paper:
    paper_id: str
    md_path: Path
    mineru_dir: Path
    content_list_path: Path
    abstract_text: str
    body_text: str
    body_image_hashes: set[str]
    all_candidates: list[ImageCandidate] = field(default_factory=list)
    filtered_candidates: list[ImageCandidate] = field(default_factory=list)

    @property
    def short_id(self) -> str:
        return self.paper_id[:40] + "..." if len(self.paper_id) > 40 else self.paper_id
