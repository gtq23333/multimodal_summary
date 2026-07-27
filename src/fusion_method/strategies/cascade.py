from __future__ import annotations

from fusion_method.types import FusionInput


class CascadeFusion:
    """Round-robin interleave from sources in declaration order."""

    def __init__(self) -> None:
        self.name = "Cascade"

    def fuse(self, fusion_input: FusionInput) -> list[tuple[str, float]]:
        seen: set[str] = set()
        ordered: list[str] = []
        lists = [s.ranked_ids for s in fusion_input.sources if s.ranked_ids]
        if not lists:
            return []

        max_len = max(len(lst) for lst in lists)
        for i in range(max_len):
            for lst in lists:
                if i < len(lst):
                    figure_id = lst[i]
                    if figure_id not in seen:
                        seen.add(figure_id)
                        ordered.append(figure_id)

        for source in fusion_input.sources:
            for figure_id in source.ranked_ids:
                if figure_id not in seen:
                    seen.add(figure_id)
                    ordered.append(figure_id)

        n = len(ordered)
        return [(figure_id, float(n - idx)) for idx, figure_id in enumerate(ordered)]
