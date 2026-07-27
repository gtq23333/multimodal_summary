"""Multi-ranker fusion methods for Stage-2 figure reranking."""

from fusion_method.ranker import MultiSourceFusionRanker, build_fusion_rankers
from fusion_method.strategies import (
    BordaFusion,
    CascadeFusion,
    RRFFusion,
    UnionRRFFusion,
    WeightedScoreFusion,
)

__all__ = [
    "BordaFusion",
    "CascadeFusion",
    "MultiSourceFusionRanker",
    "RRFFusion",
    "UnionRRFFusion",
    "WeightedScoreFusion",
    "build_fusion_rankers",
]
