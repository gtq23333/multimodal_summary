from __future__ import annotations

from fusion_method.types import FusionInput


class BordaFusion:
    """Borda count: sum of (list_len - rank) across sources."""

    def __init__(self) -> None:
        self.name = "Borda"

    def fuse(self, fusion_input: FusionInput) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for figure_id in fusion_input.all_figure_ids():
            total = 0.0
            for source in fusion_input.sources:
                rank = source.rank_of(figure_id)
                if rank is not None and source.ranked_ids:
                    total += len(source.ranked_ids) - rank
            scores[figure_id] = total
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
