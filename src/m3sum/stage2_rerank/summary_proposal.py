from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from m3sum.config import PipelineConfig
from m3sum.data.schema import Block, FigureMeta, SubQuery
from m3sum.stage2_rerank.caption_refs import parse_figure_index_from_caption
from m3sum.stage2_rerank.cluster_prior import normalize_priors_relative
from m3sum.stage2_rerank.co_occurrence import (
    QueryBlockPair,
    collect_figure_evidence_blocks,
    cosine_sim,
    evidence_debug,
    link_score_summary_features,
)
from m3sum.stage2_rerank.hybrid_retriever import HybridRetriever


DEFAULT_SUMMARY_PROPOSAL_WEIGHTS = {
    "layout": 0.30,
    "relevance": 0.35,
    "type": 0.12,
    "generality": 0.13,
    "cluster": 0.10,
}


@dataclass
class SummaryFigureFeatures:
    """Proposed-v2 的单图可解释特征。"""

    figure_id: str
    caption: str
    raw: dict[str, float]
    normalized: dict[str, float] = field(default_factory=dict)
    groups: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    debug: dict[str, Any] = field(default_factory=dict)


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    arr = np.array(values, dtype=np.float64)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [float((v - lo) / (hi - lo)) for v in values]


def _direct_summary_features(
    sub_queries: list[SubQuery],
    query_embeddings: list[np.ndarray | None],
    figure: FigureMeta,
    figure_embeddings: dict[str, np.ndarray],
    *,
    coverage_threshold: float,
) -> dict[str, float]:
    fig_emb = figure_embeddings.get(figure.image_hash)
    if fig_emb is None:
        return {
            "direct_peak": 0.0,
            "direct_mean_top2": 0.0,
            "direct_coverage": 0.0,
        }

    sims: list[float] = []
    for q_emb in query_embeddings[: len(sub_queries)]:
        if q_emb is None:
            continue
        sims.append(cosine_sim(q_emb, fig_emb))
    if not sims:
        return {
            "direct_peak": 0.0,
            "direct_mean_top2": 0.0,
            "direct_coverage": 0.0,
        }

    top2 = sorted(sims, reverse=True)[:2]
    return {
        "direct_peak": max(sims),
        "direct_mean_top2": float(np.mean(top2)) if top2 else 0.0,
        "direct_coverage": sum(1 for s in sims if s >= coverage_threshold) / len(sims),
    }


def build_query_block_pairs(
    sub_queries: list[SubQuery],
    query_embeddings: list[np.ndarray | None],
    blocks: list[Block],
    block_embeddings: dict[str, np.ndarray],
    hybrid: HybridRetriever,
) -> list[QueryBlockPair]:
    pairs: list[QueryBlockPair] = []
    for i, q in enumerate(sub_queries):
        q_emb = query_embeddings[i] if i < len(query_embeddings) else None
        for block in hybrid.search(q, blocks, block_embeddings, q_emb):
            pairs.append(QueryBlockPair(query_idx=i, block=block, query_embedding=q_emb))
    return pairs


def _raw_weights(config: PipelineConfig) -> dict[str, float]:
    cfg = config.raw.get("summary_proposal", {}).get("weights", {})
    weights = dict(DEFAULT_SUMMARY_PROPOSAL_WEIGHTS)
    for key in weights:
        if key in cfg:
            weights[key] = float(cfg[key])
    return weights


def _feature_config(config: PipelineConfig) -> dict[str, Any]:
    return dict(config.raw.get("summary_proposal", {}))


def _weighted_mean(values: dict[str, float], weights: dict[str, float]) -> float:
    denom = sum(weights.values())
    if denom <= 0:
        return 0.0
    return sum(values.get(key, 0.0) * weight for key, weight in weights.items()) / denom


def build_summary_figure_features(
    config: PipelineConfig,
    *,
    sub_queries: list[SubQuery],
    figures: list[FigureMeta],
    blocks: list[Block],
    block_embeddings: dict[str, np.ndarray],
    figure_embeddings: dict[str, np.ndarray],
    query_embeddings: list[np.ndarray | None],
    stage2_items: list[dict[str, Any]],
    cluster_priors_by_figure: dict[str, float] | None = None,
) -> list[SummaryFigureFeatures]:
    cfg = _feature_config(config)
    link_top_k = int(cfg.get("link_top_k", 3))
    link_coverage_threshold = float(cfg.get("link_coverage_threshold", 0.35))
    direct_coverage_threshold = float(cfg.get("direct_coverage_threshold", 0.35))

    item_by_id = {str(item.get("image_hash")): item for item in stage2_items}
    hybrid = HybridRetriever(
        bm25_weight=config.bm25_weight,
        vector_weight=config.vector_weight,
        top_p=config.top_p,
        query_use_keywords=config.query_use_keywords,
    )
    query_block_pairs = build_query_block_pairs(
        sub_queries,
        query_embeddings,
        blocks,
        block_embeddings,
        hybrid,
    )

    raw_features: list[SummaryFigureFeatures] = []
    for figure in figures:
        item = item_by_id.get(figure.image_hash, {})
        figure_index = parse_figure_index_from_caption(figure.caption)
        evidence_blocks = collect_figure_evidence_blocks(
            figure,
            figure_index,
            blocks,
            local_window_mode=str(config.raw.get("rerank", {}).get("local_window_mode", "always")),
        )
        link = link_score_summary_features(
            query_block_pairs,
            evidence_blocks,
            block_embeddings,
            config.distance_tiers,
            top_k=link_top_k,
            coverage_threshold=link_coverage_threshold,
        )
        direct = _direct_summary_features(
            sub_queries,
            query_embeddings,
            figure,
            figure_embeddings,
            coverage_threshold=direct_coverage_threshold,
        )
        explicit_count = sum(1 for ev in evidence_blocks if ev.source == "explicit_ref")
        local_count = sum(1 for ev in evidence_blocks if ev.source in {"local_prev", "local_next"})
        raw = {
            "legacy_score": float(item.get("score", 0.0) or 0.0),
            "legacy_semantic": float(item.get("debug", {}).get("semantic_base", 0.0) or 0.0),
            "layout": float(item.get("p_layout", 0.0) or 0.0),
            "type": float(item.get("p_type", 1.0) or 1.0),
            "cluster": float((cluster_priors_by_figure or {}).get(figure.image_hash, 0.0)),
            "direct_peak": direct["direct_peak"],
            "direct_mean_top2": direct["direct_mean_top2"],
            "direct_coverage": direct["direct_coverage"],
            "link_peak": link.link_peak,
            "link_topk_mean": link.link_topk_mean,
            "link_query_coverage": link.link_query_coverage,
            "link_source_balance": link.link_source_balance,
            "link_explicit_strength": link.link_explicit_strength,
            "link_local_strength": link.link_local_strength,
            "evidence_count": float(link.evidence_count),
            "explicit_count": float(explicit_count),
            "local_count": float(local_count),
            "early_body_order": 1.0 / max(float(figure.body_order + 1), 1.0),
        }
        raw_features.append(
            SummaryFigureFeatures(
                figure_id=figure.image_hash,
                caption=figure.caption,
                raw=raw,
                debug={
                    "legacy": {
                        "score": item.get("score"),
                        "s_direct": item.get("s_direct"),
                        "s_link": item.get("s_link", item.get("s_co")),
                        "p_layout": item.get("p_layout"),
                        "p_type": item.get("p_type"),
                    },
                    "link": link.debug,
                    "direct": direct,
                    "evidence_blocks": evidence_debug(evidence_blocks),
                },
            )
        )

    return score_summary_figure_features(config, raw_features)


def score_summary_figure_features(
    config: PipelineConfig,
    features: list[SummaryFigureFeatures],
) -> list[SummaryFigureFeatures]:
    if not features:
        return features

    proposal_cfg = _feature_config(config)
    legacy_blend = float(proposal_cfg.get("legacy_blend", 0.55))
    feature_names = sorted({name for feature in features for name in feature.raw})
    for name in feature_names:
        normed = _minmax([feature.raw.get(name, 0.0) for feature in features])
        for feature, value in zip(features, normed):
            feature.normalized[name] = value

    weights = _raw_weights(config)
    for feature in features:
        n = feature.normalized
        relevance = _weighted_mean(
            n,
            {
                "legacy_score": 0.30,
                "direct_peak": 0.25,
                "direct_mean_top2": 0.10,
                "direct_coverage": 0.10,
                "link_topk_mean": 0.10,
                "link_query_coverage": 0.10,
                "link_source_balance": 0.05,
            },
        )
        generality = _weighted_mean(
            n,
            {
                "explicit_count": 0.35,
                "evidence_count": 0.25,
                "local_count": 0.15,
                "early_body_order": 0.25,
            },
        )
        type_group = _weighted_mean(
            n,
            {
                "type": 0.70,
                "cluster": 0.30,
            },
        )
        groups = {
            "layout": n.get("layout", 0.0),
            "relevance": relevance,
            "type": type_group,
            "generality": generality,
            "cluster": n.get("cluster", 0.0),
        }
        feature.groups = groups
        proposal_score = sum(groups.get(k, 0.0) * v for k, v in weights.items())
        feature.score = (
            legacy_blend * n.get("legacy_score", 0.0)
            + (1.0 - legacy_blend) * proposal_score
        )
        feature.debug["summary_proposal"] = {
            "raw": {k: round(v, 6) for k, v in feature.raw.items()},
            "normalized": {k: round(v, 6) for k, v in feature.normalized.items()},
            "groups": {k: round(v, 6) for k, v in groups.items()},
            "weights": weights,
            "legacy_blend": round(legacy_blend, 6),
            "proposal_score": round(proposal_score, 6),
            "score": round(feature.score, 6),
        }

    return features


def normalize_cluster_priors(priors_by_figure: dict[str, float]) -> dict[str, float]:
    figure_ids = list(priors_by_figure)
    normed = normalize_priors_relative([priors_by_figure[fid] for fid in figure_ids])
    return {fid: val for fid, val in zip(figure_ids, normed)}
