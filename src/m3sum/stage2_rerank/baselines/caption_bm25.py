from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from m3sum.stage2_rerank.baselines.base import Stage2Sample, build_ranked_list
from m3sum.stage2_rerank.tokenize import tokenize


class CaptionBM25Ranker:
    """
    Caption-BM25 baseline。
    单篇文档内对 figure caption 建 BM25 索引，query 分数取平均。
    """

    method_name = "Caption-BM25"

    def rank(self, sample: Stage2Sample) -> list:
        if not sample.figures:
            return []

        captions = [f.caption or "" for f in sample.figures]
        corpus = [tokenize(c) for c in captions]
        bm25 = BM25Okapi(corpus)

        scored: list[tuple[str, float]] = []
        for fig_idx, (fig, cap_tokens) in enumerate(zip(sample.figures, corpus)):
            if not cap_tokens:
                scored.append((fig.image_hash, 0.0))
                continue
            query_scores: list[float] = []
            for q in sample.sub_queries:
                q_tokens = tokenize(q.query + " " + " ".join(q.keywords))
                if not q_tokens:
                    query_scores.append(0.0)
                    continue
                raw = float(bm25.get_scores(q_tokens)[fig_idx])
                query_scores.append(raw)
            mean_score = float(np.mean(query_scores)) if query_scores else 0.0
            scored.append((fig.image_hash, mean_score))

        return build_ranked_list(scored, self.method_name)
