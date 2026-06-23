"""
改造前 LG-JSSF 重排逻辑（trial_10 时代）。

- 无条件 local 上下窗口
- S_link 全局 max 聚合（无分源门控、无自匹配跳过）
- 固定 alpha=0.5 融合 S_direct / S_link
- P_type 使用 1.5 / 0.8 强先验
"""

from __future__ import annotations

from typing import Any

import numpy as np

from m3sum.data.schema import Block, FigureMeta, SubQuery
from m3sum.stage2_rerank.caption_refs import (
    figure_ref_to_str,
    parse_figure_index_from_caption,
)
from m3sum.stage2_rerank.co_occurrence import (
    QueryBlockPair,
    collect_figure_evidence_blocks,
    cosine_sim,
    evidence_debug,
    link_score_legacy,
)
from m3sum.stage2_rerank.hybrid_retriever import HybridRetriever
from m3sum.stage2_rerank.caption_regex import match_caption_blocks
from m3sum.stage2_rerank.reranker import (
    direct_similarity,
    layout_prior,
    type_prior,
)


def type_prior_legacy(caption: str) -> float:
    """改造前 P_type：方法图 1.5x，数据图 0.8x。"""
    return type_prior(caption, method_boost=1.5, data_penalty=0.8)


def rerank_figures_legacy(
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
) -> dict[str, Any]:
    """运行改造前 LG-JSSF 重排，输出格式与 rerank_figures 一致。"""
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

    all_scores: list[dict[str, Any]] = []
    for fig in figures:
        figure_index = parse_figure_index_from_caption(fig.caption)
        evidence_blocks = collect_figure_evidence_blocks(
            fig,
            figure_index,
            blocks,
            local_window_mode="always",
        )
        s_direct = direct_similarity(sub_queries, fig, query_embeddings, figure_embeddings)
        s_link, link_debug = link_score_legacy(
            query_block_pairs,
            evidence_blocks,
            block_embeddings,
            distance_tiers,
        )
        semantic_base = alpha * s_direct + (1.0 - alpha) * s_link
        p_layout, layout_index = layout_prior(fig, figure_index)
        p_type = type_prior_legacy(fig.caption)
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
                "effective_alpha": alpha,
                "semantic_base": round(semantic_base, 6),
                "rerank_profile": "legacy",
            },
            "pos": fig.pos,
        }
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
            "rerank_profile": "legacy",
        },
    }
