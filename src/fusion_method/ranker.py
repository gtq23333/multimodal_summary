from __future__ import annotations

from fusion_method.load_rankings import build_fusion_input, rankings_index
from fusion_method.strategies.base import FusionStrategy
from fusion_method.strategies.borda import BordaFusion
from fusion_method.strategies.cascade import CascadeFusion
from fusion_method.strategies.rrf import RRFFusion
from fusion_method.strategies.union_pool import UnionRRFFusion
from fusion_method.strategies.weighted_score import WeightedScoreFusion
from fusion_method.types import DEFAULT_SOURCE_METHODS, DEFAULT_SOURCE_SUFFIX
from m3sum.stage2_rerank.baselines.base import RankedFigure, Stage2Sample, build_ranked_list


class MultiSourceFusionRanker:
    """Fuse multiple cached ranker outputs into one ranked list."""

    def __init__(
        self,
        method_name: str,
        strategy: FusionStrategy,
        rankings_by_method: dict[str, dict[str, dict]],
        source_methods: list[str],
    ) -> None:
        self.method_name = method_name
        self.strategy = strategy
        self.rankings_by_method = rankings_by_method
        self.source_methods = source_methods

    def rank(self, sample: Stage2Sample) -> list[RankedFigure]:
        fusion_input = build_fusion_input(
            sample.paper_id,
            self.rankings_by_method,
            self.source_methods,
        )
        scored = self.strategy.fuse(fusion_input)
        return build_ranked_list(scored, self.method_name)


def _method_label(strategy_suffix: str, source_suffix: str = DEFAULT_SOURCE_SUFFIX) -> str:
    return f"Fusion-{strategy_suffix}-{source_suffix}"


def build_fusion_rankers(
    rankings_records: list[dict],
    *,
    source_methods: list[str] | None = None,
    source_suffix: str = DEFAULT_SOURCE_SUFFIX,
    rrf_k: int = 60,
    pool_k: int = 8,
    weights: dict[str, float] | None = None,
) -> dict[str, MultiSourceFusionRanker]:
    sources = source_methods or DEFAULT_SOURCE_METHODS
    index = rankings_index(rankings_records)

    strategies: list[tuple[str, FusionStrategy]] = [
        ("RRF", RRFFusion(rrf_k=rrf_k)),
        ("Borda", BordaFusion()),
        ("Weighted", WeightedScoreFusion(weights=weights)),
        ("Cascade", CascadeFusion()),
        ("UnionRRF", UnionRRFFusion(pool_k=pool_k, rrf_k=rrf_k)),
    ]

    rankers: dict[str, MultiSourceFusionRanker] = {}
    for suffix, strategy in strategies:
        name = _method_label(suffix, source_suffix)
        rankers[name] = MultiSourceFusionRanker(
            method_name=name,
            strategy=strategy,
            rankings_by_method=index,
            source_methods=sources,
        )
    return rankers
