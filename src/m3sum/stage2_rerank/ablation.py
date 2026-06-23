from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from m3sum.config import PipelineConfig
from m3sum.stage2_rerank.baselines.base import RankedFigure, Stage2Sample
from m3sum.stage2_rerank.cluster_prior import ClusterPriorScorer
from m3sum.stage2_rerank.fusion import FusionConfig, compute_fused_score


class Stage2FeatureRanker:
    """基于 Stage-2 JSON 特征的 Proposed 消融/融合 ranker。"""

    def __init__(
        self,
        config: PipelineConfig,
        fusion_config: FusionConfig,
        cluster_scorer: ClusterPriorScorer | None = None,
        image_embeddings_by_paper: dict[str, dict[str, np.ndarray | None]] | None = None,
    ):
        self.config = config
        self.fusion_config = fusion_config
        self.method_name = fusion_config.method_name
        self.cluster_scorer = cluster_scorer
        self.image_embeddings_by_paper = image_embeddings_by_paper or {}

    def _load_stage2_items(self, paper_id: str) -> list[dict[str, Any]]:
        path = self.config.stage2_dir / f"{paper_id}.json"
        if not path.is_file():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("all_scores", []))

    def _cluster_prior(
        self,
        paper_id: str,
        figure_id: str,
        query_text: str = "",
        relative: bool = False,
        all_priors: list[float] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        if self.cluster_scorer is None:
            return 0.0, {}
        emb = self.image_embeddings_by_paper.get(paper_id, {}).get(figure_id)
        prior, debug = self.cluster_scorer.score_with_context(emb, query_text=query_text)
        if relative and all_priors is not None:
            idx = all_priors.index(prior) if prior in all_priors else -1
            # relative normalization applied at rank() level
            pass
        debug.cluster_fusion_mode = self.fusion_config.cluster_fusion_mode
        return prior, debug.to_dict()

    def rank(self, sample: Stage2Sample) -> list[RankedFigure]:
        items = self._load_stage2_items(sample.paper_id)
        query_text = " ".join(q.query for q in sample.sub_queries) if sample.sub_queries else ""

        use_relative = bool(self.config.raw.get("cluster_prior", {}).get("relative_prior", True))
        raw_priors: list[float] = []
        prior_by_fid: dict[str, float] = {}
        debug_by_fid: dict[str, dict[str, Any]] = {}

        for item in items:
            figure_id = item["image_hash"]
            prior, debug = self._cluster_prior(sample.paper_id, figure_id, query_text)
            raw_priors.append(prior)
            prior_by_fid[figure_id] = prior
            debug_by_fid[figure_id] = debug

        if use_relative and self.fusion_config.use_cluster:
            from m3sum.stage2_rerank.cluster_prior import normalize_priors_relative

            figure_ids = list(prior_by_fid.keys())
            normed = normalize_priors_relative([prior_by_fid[fid] for fid in figure_ids])
            for fid, val in zip(figure_ids, normed):
                prior_by_fid[fid] = val

        scored: list[tuple[str, float]] = []
        for item in items:
            figure_id = item["image_hash"]
            cluster_prior = prior_by_fid.get(figure_id, 0.0) if self.fusion_config.use_cluster else 0.0
            score = compute_fused_score(
                item,
                self.fusion_config,
                alpha=self.config.alpha,
                cluster_prior=cluster_prior,
                rerank_raw=self.config.raw.get("rerank"),
            )
            scored.append((figure_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            RankedFigure(
                figure_id=figure_id,
                score=score,
                rank=i + 1,
                method_name=self.method_name,
            )
            for i, (figure_id, score) in enumerate(scored)
        ]

    def debug_for_sample(self, sample: Stage2Sample) -> list[dict[str, Any]]:
        """输出每张图的 cluster prior debug，用于诊断日志。"""
        rows: list[dict[str, Any]] = []
        for item in self._load_stage2_items(sample.paper_id):
            figure_id = item["image_hash"]
            cluster_prior, debug = self._cluster_prior(sample.paper_id, figure_id)
            rows.append(
                {
                    "figure_id": figure_id,
                    "caption": item.get("caption", "")[:80],
                    "cluster_prior": cluster_prior,
                    **debug,
                }
            )
        return rows
