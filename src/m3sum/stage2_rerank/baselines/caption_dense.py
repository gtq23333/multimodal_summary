from __future__ import annotations

from pathlib import Path

import numpy as np

from m3sum.stage2_rerank.baselines.base import Stage2Sample, build_ranked_list
from m3sum.stage2_rerank.co_occurrence import cosine_sim
from m3sum.stage2_rerank.text_embedding_cache import TextEmbeddingCache


class CaptionDenseRanker:
    """
    Caption-Dense-v4 baseline。
    使用 text embedding 模型对 query 与 caption 计算 cosine 相似度均值。
    """

    method_name = "Caption-Dense-v4"

    def __init__(self, cache_dir: Path, embedder, dry_run: bool = False):
        self.cache = TextEmbeddingCache(cache_dir, embedder, dry_run=dry_run)

    def rank(self, sample: Stage2Sample) -> list:
        if not sample.figures:
            return []

        query_embs, caption_embs = self.cache.load_or_compute(
            sample.paper_id,
            sample.sub_queries,
            sample.figures,
        )

        scored: list[tuple[str, float]] = []
        for fig in sample.figures:
            cap_emb = caption_embs.get(fig.image_hash)
            if cap_emb is None:
                scored.append((fig.image_hash, 0.0))
                continue
            sims: list[float] = []
            for q_emb in query_embs:
                sims.append(cosine_sim(q_emb, cap_emb))
            mean_sim = float(np.mean(sims)) if sims else 0.0
            scored.append((fig.image_hash, mean_sim))

        return build_ranked_list(scored, self.method_name)
