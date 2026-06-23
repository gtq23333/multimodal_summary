from __future__ import annotations

import math
from typing import Any

import numpy as np

from m3sum.data.schema import Block, FigureMeta, SubQuery
from m3sum.stage2_rerank.caption_refs import (
    FigureRef,
    figure_ref_to_str,
    parse_figure_index_from_caption,
)
from m3sum.stage2_rerank.co_occurrence import (
    QueryBlockPair,
    LinkFusionParams,
    collect_figure_evidence_blocks,
    cosine_sim,
    evidence_debug,
    fuse_direct_and_link,
    link_score_gated,
)
from m3sum.stage2_rerank.hybrid_retriever import HybridRetriever
from m3sum.stage2_rerank.caption_regex import match_caption_blocks

METHOD_FIGURE_KEYWORDS = (
    "流程",
    "框图",
    "步骤",
    "架构",
    "算法",
    "pipeline",
    "framework",
    "flowchart",
    "overview",
)

DATA_FIGURE_KEYWORDS = (
    "折线",
    "曲线",
    "统计",
    "散点",
    "数据",
    "结果",
    "comparison",
    "trend",
)


def direct_similarity(
    sub_queries: list[SubQuery],
    figure: FigureMeta,
    query_embeddings: list[np.ndarray],
    figure_embeddings: dict[str, np.ndarray],
) -> float:
    fig_emb = figure_embeddings.get(figure.image_hash)
    if fig_emb is None:
        return 0.0

    scores: list[float] = []
    for q, q_emb in zip(sub_queries, query_embeddings):
        if q_emb is None:
            continue
        sim = cosine_sim(q_emb, fig_emb)
        scores.append(sim)
    return max(scores) if scores else 0.0


def _figure_ref_to_layout_number(ref: FigureRef | None) -> float | None:
    if ref is None:
        return None
    if len(ref) == 1:
        return float(ref[0])
    return float(f"{ref[0]}." + "".join(str(part) for part in ref[1:]))


def layout_prior(figure: FigureMeta, figure_index: FigureRef | None) -> tuple[float, float]:
    """全局布局顺序衰减先验；无图号时回退 body_order。"""
    layout_index = _figure_ref_to_layout_number(figure_index)
    if layout_index is None or layout_index <= 0:
        layout_index = float(max(figure.body_order + 1, 1))
    return 1.0 / math.log2(1.0 + layout_index), layout_index


def type_prior(caption: str, *, method_boost: float = 1.15, data_penalty: float = 0.92) -> float:
    """启发式图表类型先验：方法图略升权，局部数据图略降权（幅度已弱化）。"""
    lower_caption = (caption or "").lower()
    if any(keyword in lower_caption for keyword in METHOD_FIGURE_KEYWORDS):
        return method_boost
    if any(keyword in lower_caption for keyword in DATA_FIGURE_KEYWORDS):
        return data_penalty
    return 1.0


def _link_params_from_config(raw: dict[str, Any] | None) -> LinkFusionParams:
    rr = raw or {}
    return LinkFusionParams(
        alpha=float(rr.get("alpha", 0.5)),
        alpha_local=float(rr.get("alpha_local", 0.75)),
        local_gamma=float(rr.get("local_gamma", 0.35)),
        local_cap=float(rr.get("local_cap", 0.75)),
        local_window_mode=str(rr.get("local_window_mode", "deictic_only")),
        explicit_link_threshold=float(rr.get("explicit_link_threshold", 0.12)),
    )


def rerank_figures(
    paper_id: str,
    sub_queries: list[SubQuery],
    blocks: list[Block],
    figures: list[FigureMeta],
    block_embeddings: dict[str, np.ndarray],
    figure_embeddings: dict[str, np.ndarray],
    query_embeddings: list[np.ndarray],
    caption_patterns: list[str],
    alpha: float,
    distance_tiers: list[float],
    hybrid: HybridRetriever,
    top_k: int = 3,
    cluster_debug_by_hash: dict[str, dict[str, Any]] | None = None,
    rerank_raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    query_block_pairs: list[QueryBlockPair] = []
    cq_blocks_for_debug: list[Block] = []
    seen_cq_for_debug: set[str] = set()
    for i, q in enumerate(sub_queries):
        q_emb = query_embeddings[i] if query_embeddings and i < len(query_embeddings) else None
        for b in hybrid.search(q, blocks, block_embeddings, q_emb):
            query_block_pairs.append(
                QueryBlockPair(query_idx=i, block=b, query_embedding=q_emb)
            )
            if b.block_id not in seen_cq_for_debug:
                seen_cq_for_debug.add(b.block_id)
                cq_blocks_for_debug.append(b)

    caption_result = match_caption_blocks(blocks, caption_patterns)
    cf_blocks = caption_result.matched_blocks

    link_params = _link_params_from_config(rerank_raw)
    type_method_boost = float((rerank_raw or {}).get("type_method_boost", 1.15))
    type_data_penalty = float((rerank_raw or {}).get("type_data_penalty", 0.92))

    all_scores: list[dict[str, Any]] = []
    for fig in figures:
        figure_index = parse_figure_index_from_caption(fig.caption)
        evidence_blocks = collect_figure_evidence_blocks(
            fig,
            figure_index,
            blocks,
            local_window_mode=link_params.local_window_mode,
        )
        s_direct = direct_similarity(sub_queries, fig, query_embeddings, figure_embeddings)
        s_link, _, _, link_debug = link_score_gated(
            query_block_pairs,
            evidence_blocks,
            block_embeddings,
            distance_tiers,
            link_params,
        )
        effective_alpha = float(link_debug.get("effective_alpha", alpha))
        semantic_base = fuse_direct_and_link(s_direct, s_link, effective_alpha)
        p_layout, layout_index = layout_prior(fig, figure_index)
        p_type = type_prior(
            fig.caption,
            method_boost=type_method_boost,
            data_penalty=type_data_penalty,
        )
        score = semantic_base * p_layout * p_type
        item = {
                "image_hash": fig.image_hash,
                "caption": fig.caption,
                "score": round(score, 6),
                "s_direct": round(s_direct, 6),
                "s_link": round(s_link, 6),
                "s_co": round(s_link, 6),
                "p_layout": round(p_layout, 6),
                "p_type": round(p_type, 6),
                "figure_index": figure_ref_to_str(figure_index),
                "layout_index": round(layout_index, 6),
                "evidence_blocks": evidence_debug(evidence_blocks),
                "debug": {
                    "link": link_debug,
                    "evidence_count": len(evidence_blocks),
                    "effective_alpha": effective_alpha,
                    "semantic_base": round(semantic_base, 6),
                },
                "pos": fig.pos,
            }
        if cluster_debug_by_hash and fig.image_hash in cluster_debug_by_hash:
            item["debug"]["cluster_prior"] = cluster_debug_by_hash[fig.image_hash]
        all_scores.append(item)

    all_scores.sort(key=lambda x: x["score"], reverse=True)
    for i, item in enumerate(all_scores):
        item["rank"] = i + 1

    top3 = all_scores[:top_k]
    for item in top3:
        item.setdefault("debug", {})
        item["debug"].update(
            {
                "matched_caption_blocks": len(cf_blocks),
                "query_block_pairs": len(query_block_pairs),
                "top_cq_blocks": [b.block_id for b in cq_blocks_for_debug[:5]],
            }
        )

    return {
        "paper_id": paper_id,
        "top3_figures": top3,
        "all_scores": all_scores,
        "recall_debug": {
            "cq_count": len(cq_blocks_for_debug),
            "query_block_pair_count": len(query_block_pairs),
            "cf_count": len(cf_blocks),
            "caption_refs_sample": caption_result.all_refs[:10],
        },
    }
