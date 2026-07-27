"""
主方法：Legacy LG-JSSF + ClusterPrior（additive）。

同集对比中 Legacy+ClusterAdd 在 R/MAP/MRR 上最优，因此作为 Proposed 默认实现。
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from m3sum.config import PipelineConfig
from m3sum.data.schema import FigureMeta
from m3sum.stage2_rerank.baselines.base import RankedFigure, Stage2Sample
from m3sum.stage2_rerank.cluster_prior import ClusterPriorScorer
from m3sum.stage2_rerank.fusion import FusionConfig, compute_fused_score


def main_fusion_config(config: PipelineConfig) -> FusionConfig:
    cp = config.raw.get("cluster_prior", {})
    beta = float(cp.get("main_beta", cp.get("beta", 0.25)))
    use_cluster = bool(cp.get("enabled", config.cluster_prior_enabled))
    return FusionConfig(
        "Proposed",
        use_cluster=use_cluster,
        cluster_fusion_mode="additive",
        beta=beta,
    )


def main_cluster_scorer(config: PipelineConfig) -> ClusterPriorScorer | None:
    cp = config.raw.get("cluster_prior", {})
    if not cp.get("enabled", config.cluster_prior_enabled):
        return None
    return ClusterPriorScorer.from_json(
        config.cluster_prior_path,
        tau=float(cp.get("main_tau", cp.get("tau", 0.72))),
        margin_tau=float(cp.get("main_margin_tau", cp.get("margin_tau", 0.03))),
        threshold_mode=str(cp.get("threshold_mode", config.cluster_prior_threshold_mode)),
    )


def stage2_query_config_matches(config: PipelineConfig, paper_id: str) -> bool:
    """stage2 缓存是否与当前 query_use_keywords 配置一致。"""
    path = config.stage2_dir / f"{paper_id}.json"
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    stored = data.get("recall_debug", {}).get("query_use_keywords")
    if stored is None:
        # 旧缓存默认按 query+keywords 生成
        return config.query_use_keywords is True
    return stored == config.query_use_keywords


def _load_stage2_items(config: PipelineConfig, paper_id: str) -> list[dict[str, Any]]:
    if not stage2_query_config_matches(config, paper_id):
        return []
    path = config.stage2_dir / f"{paper_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("all_scores", []))


def rank_main_method(
    config: PipelineConfig,
    sample: Stage2Sample,
    image_embeddings: dict[str, np.ndarray | None] | None = None,
) -> list[RankedFigure]:
    """Legacy LG-JSSF 基础分 + ClusterPrior additive 融合。"""
    items = _load_stage2_items(config, sample.paper_id)
    if not items:
        return []

    fusion = main_fusion_config(config)
    scorer = main_cluster_scorer(config)
    rerank_raw = config.raw.get("rerank", {})
    alpha = float(rerank_raw.get("alpha", 0.5))
    emb_map = image_embeddings or {}

    scored: list[tuple[str, float]] = []
    for item in items:
        figure_id = item["image_hash"]
        cluster_prior = 0.0
        if fusion.use_cluster and scorer is not None:
            emb = emb_map.get(figure_id)
            cluster_prior, _ = scorer.score(emb)
        score = compute_fused_score(
            item,
            fusion,
            alpha=alpha,
            cluster_prior=cluster_prior,
            rerank_raw=rerank_raw,
        )
        scored.append((figure_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [
        RankedFigure(
            figure_id=figure_id,
            score=score,
            rank=i + 1,
            method_name=fusion.method_name,
        )
        for i, (figure_id, score) in enumerate(scored)
    ]


def finalize_stage2_with_cluster(
    config: PipelineConfig,
    result: dict[str, Any],
    figures: list[FigureMeta],
    image_embeddings: dict[str, np.ndarray | None],
    top_k: int = 3,
) -> dict[str, Any]:
    """在 legacy stage2 结果上叠加 ClusterPrior，更新 score / rank / top3。"""
    fusion = main_fusion_config(config)
    scorer = main_cluster_scorer(config)
    if not fusion.use_cluster or scorer is None:
        return result

    rerank_raw = config.raw.get("rerank", {})
    alpha = float(rerank_raw.get("alpha", 0.5))
    all_scores = list(result.get("all_scores", []))

    for item in all_scores:
        figure_id = item["image_hash"]
        emb = image_embeddings.get(figure_id)
        cluster_prior, cluster_debug = scorer.score(emb)
        base_score = float(item.get("score", 0.0))
        final_score = compute_fused_score(
            item,
            fusion,
            alpha=alpha,
            cluster_prior=cluster_prior,
            rerank_raw=rerank_raw,
        )
        item["score_base"] = round(base_score, 6)
        item["score"] = round(final_score, 6)
        item.setdefault("debug", {})
        item["debug"]["cluster"] = cluster_debug.to_dict()
        item["debug"]["cluster_fusion"] = {
            "mode": "additive",
            "beta": fusion.beta,
            "cluster_prior": round(cluster_prior, 6),
        }

    all_scores.sort(key=lambda x: x["score"], reverse=True)
    for i, item in enumerate(all_scores):
        item["rank"] = i + 1

    result["all_scores"] = all_scores
    result["top3_figures"] = all_scores[:top_k]
    result.setdefault("recall_debug", {})
    result["recall_debug"]["main_method"] = "LG-JSSF-Legacy+ClusterAdd"
    return result
