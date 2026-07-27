from __future__ import annotations

from fusion_method.strategies.rrf import RRFFusion
from fusion_method.types import FusionInput, SourceRanking


class UnionRRFFusion:
    """Union of per-source top-pool_k candidates, then RRF rerank."""

    def __init__(self, pool_k: int = 8, rrf_k: int = 60) -> None:
        self.pool_k = pool_k
        self._rrf = RRFFusion(rrf_k=rrf_k)
        self.name = "UnionRRF"

    def fuse(self, fusion_input: FusionInput) -> list[tuple[str, float]]:
        pool_ids: set[str] = set()
        restricted_sources: list[SourceRanking] = []
        for source in fusion_input.sources:
            top_ids = source.ranked_ids[: self.pool_k]
            pool_ids.update(top_ids)
            restricted_sources.append(
                SourceRanking(
                    method_name=source.method_name,
                    ranked_ids=top_ids,
                    score_by_id={
                        fid: source.score_by_id[fid]
                        for fid in top_ids
                        if fid in source.score_by_id
                    },
                )
            )

        restricted = FusionInput(paper_id=fusion_input.paper_id, sources=restricted_sources)
        fused = self._rrf.fuse(restricted)
        fused_ids = {fid for fid, _ in fused}
        missing = pool_ids - fused_ids
        if missing:
            extra = sorted(missing)
            fused.extend((fid, 0.0) for fid in extra)
        return fused
