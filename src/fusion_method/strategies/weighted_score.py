from __future__ import annotations

from fusion_method.types import DEFAULT_WEIGHTS, FusionInput


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class WeightedScoreFusion:
    """Per-paper min-max normalized score fusion."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.name = "Weighted"

    def fuse(self, fusion_input: FusionInput) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for figure_id in fusion_input.all_figure_ids():
            total = 0.0
            weight_sum = 0.0
            for source in fusion_input.sources:
                weight = self.weights.get(source.method_name, 0.0)
                if weight <= 0 or not source.score_by_id:
                    continue
                norm = _min_max_normalize(source.score_by_id)
                if figure_id in norm:
                    total += weight * norm[figure_id]
                    weight_sum += weight
            scores[figure_id] = total / weight_sum if weight_sum > 0 else 0.0
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
