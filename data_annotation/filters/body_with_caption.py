from __future__ import annotations

from filters.base import ImageFilterStrategy
from models.paper import ImageCandidate, Paper


class BodyWithCaptionFilter(ImageFilterStrategy):
    name = "body_with_caption"

    def filter(self, candidates: list[ImageCandidate], paper: Paper | None) -> list[ImageCandidate]:
        result = [c for c in candidates if c.in_body_md and c.caption.strip()]
        result.sort(key=lambda c: (c.body_order if c.body_order >= 0 else c.content_list_order))
        return result
