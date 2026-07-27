from __future__ import annotations

from fusion_method.types import FusionInput


class RRFFusion:
    """Reciprocal Rank Fusion across source rankers."""

    def __init__(self, rrf_k: int = 60) -> None:
        self.rrf_k = rrf_k
        self.name = "RRF"

    def fuse(self, fusion_input: FusionInput) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for figure_id in fusion_input.all_figure_ids():
            total = 0.0
            for source in fusion_input.sources:
                rank = source.rank_of(figure_id)
                if rank is not None:
                    total += 1.0 / (self.rrf_k + rank)
            scores[figure_id] = total
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
