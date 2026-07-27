from __future__ import annotations

from pathlib import Path

import numpy as np

from m3sum.stage2_rerank.baselines.base import Stage2Sample, build_ranked_list
from m3sum.stage2_rerank.clip_utils import ChineseCLIPWrapper, ClipImageEmbeddingCache
from m3sum.stage2_rerank.co_occurrence import cosine_sim


class ZeroshotClipRanker:
    """
    Zero-shot CLIP baseline。
    Chinese-CLIP 联合隐空间内 query-image cosine 相似度均值。
    """

    method_name = "Zero-shot-CLIP"

    def __init__(
        self,
        clip_encoder: ChineseCLIPWrapper | None,
        image_cache_dir: Path,
        dry_run: bool = False,
        *,
        query_use_keywords: bool = True,
    ):
        self.clip_encoder = clip_encoder
        self.image_cache = ClipImageEmbeddingCache(
            image_cache_dir,
            clip_encoder=clip_encoder,
            dry_run=dry_run,
        )
        self.dry_run = dry_run
        self.query_use_keywords = query_use_keywords
        self._text_cache: dict[str, list[np.ndarray]] = {}

    def rank(self, sample: Stage2Sample) -> list:
        if not sample.figures:
            return []

        query_texts = [
            q.search_text(use_keywords=self.query_use_keywords)
            for q in sample.sub_queries
        ]
        cache_key = (
            f"{sample.paper_id}::kw={self.query_use_keywords}::"
            + "|".join(query_texts)
        )
        if cache_key not in self._text_cache:
            if self.dry_run:
                dim = 512
                self._text_cache[cache_key] = [
                    np.random.randn(dim).astype(np.float32) for _ in query_texts
                ]
            else:
                if self.clip_encoder is None:
                    raise RuntimeError("Zero-shot CLIP 需要 clip_encoder 或 dry_run=True")
                self._text_cache[cache_key] = self.clip_encoder.encode_texts(query_texts)
        query_embs = self._text_cache[cache_key]

        image_embs = self.image_cache.load_or_compute(sample.paper_id, sample.figures)

        scored: list[tuple[str, float]] = []
        for fig in sample.figures:
            img_emb = image_embs.get(fig.image_hash)
            if img_emb is None:
                scored.append((fig.image_hash, 0.0))
                continue
            sims: list[float] = []
            for q_emb in query_embs:
                sims.append(cosine_sim(q_emb, img_emb))
            mean_sim = float(np.mean(sims)) if sims else 0.0
            scored.append((fig.image_hash, mean_sim))

        return build_ranked_list(scored, self.method_name)
