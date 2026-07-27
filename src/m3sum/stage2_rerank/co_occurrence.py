from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from m3sum.data.schema import Block, FigureMeta
from m3sum.stage2_rerank.caption_refs import (
    FigureRef,
    extract_labeled_caption_refs,
    figure_ref_to_str,
    parse_figure_label_from_caption,
)
from m3sum.stage2_rerank.layout_weights import block_distance, distance_weight


DEICTIC_PATTERN = re.compile(
    r"(上[图图表]|下[图图表]|如[图图表][所示]*|[见如][图图表]|所示|上述|下图|上图)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LinkFusionParams:
    """S_link 分源门控与自适应 alpha 参数。"""

    alpha: float = 0.5
    alpha_local: float = 0.75
    local_gamma: float = 0.35
    local_cap: float = 0.75
    local_window_mode: str = "deictic_only"
    explicit_link_threshold: float = 0.12


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class QueryBlockPair:
    """赛题子查询召回出的正文块及其所属 query。"""

    query_idx: int
    block: Block
    query_embedding: np.ndarray | None = None


@dataclass
class FigureEvidenceBlock:
    """某张图的候选证据块：显式图号引用或图片邻近上下文。"""

    block: Block
    source: str
    matched_ref: FigureRef | None = None
    also_local: bool = False


@dataclass(frozen=True)
class LinkSummaryFeatures:
    """面向摘要候选图 proposal 的可解释 Link 特征。"""

    link_peak: float
    link_topk_mean: float
    link_query_coverage: float
    link_source_balance: float
    link_explicit_strength: float
    link_local_strength: float
    evidence_count: int
    debug: dict[str, Any]


def _nearest_text_block(
    blocks: list[Block],
    pos: int,
    direction: int,
) -> Block | None:
    if pos < 0:
        return None

    if direction < 0:
        candidates = [
            b for b in blocks if b.block_type == "text" and b.block_idx < pos
        ]
        return max(candidates, key=lambda b: b.block_idx, default=None)

    candidates = [
        b for b in blocks if b.block_type == "text" and b.block_idx > pos
    ]
    return min(candidates, key=lambda b: b.block_idx, default=None)


def collect_figure_evidence_blocks(
    figure: FigureMeta,
    figure_index: FigureRef | None,
    blocks: list[Block],
    *,
    local_window_mode: str = "deictic_only",
) -> list[FigureEvidenceBlock]:
    """为单张图收集显式引用块；邻近窗口仅在 deictic 语境或 always 模式下启用。"""
    evidence_by_block: dict[str, FigureEvidenceBlock] = {}
    figure_label = parse_figure_label_from_caption(figure.caption)

    if figure_index is not None:
        for block in blocks:
            if block.block_type != "text":
                continue
            labeled_refs = extract_labeled_caption_refs(block.text)
            if any(
                ref == figure_index and (figure_label is None or label == figure_label)
                for label, ref in labeled_refs
            ):
                evidence_by_block[block.block_id] = FigureEvidenceBlock(
                    block=block,
                    source="explicit_ref",
                    matched_ref=figure_index,
                )

    allow_local = local_window_mode == "always" or (
        local_window_mode == "deictic_only"
        and caption_or_neighbors_have_deictic(figure, blocks)
    )
    if allow_local:
        for source, direction in (("local_prev", -1), ("local_next", 1)):
            block = _nearest_text_block(blocks, figure.pos, direction)
            if block is None:
                continue
            existing = evidence_by_block.get(block.block_id)
            if existing is not None:
                existing.also_local = True
                continue
            evidence_by_block[block.block_id] = FigureEvidenceBlock(
                block=block,
                source=source,
            )

    return list(evidence_by_block.values())


def evidence_debug(evidence_blocks: list[FigureEvidenceBlock]) -> list[dict[str, Any]]:
    """将 evidence blocks 转为 JSON 可序列化 debug 结构。"""
    return [
        {
            "block_id": ev.block.block_id,
            "source": ev.source,
            "matched_ref": figure_ref_to_str(ev.matched_ref),
            "also_local": ev.also_local,
            "text_snippet": ev.block.text[:80],
        }
        for ev in evidence_blocks
    ]


def link_score(
    query_block_pairs: list[QueryBlockPair],
    evidence_blocks: list[FigureEvidenceBlock],
    block_embeddings: dict[str, np.ndarray],
    distance_tiers: list[float],
    link_params: LinkFusionParams | None = None,
) -> tuple[float, dict[str, Any]]:
    """计算 LG-JSSF 的 S_link（分源门控 + local gamma/cap）。"""
    params = link_params or LinkFusionParams()
    s_link, _, _, debug = link_score_gated(
        query_block_pairs,
        evidence_blocks,
        block_embeddings,
        distance_tiers,
        params,
    )
    return s_link, debug


def link_score_legacy(
    query_block_pairs: list[QueryBlockPair],
    evidence_blocks: list[FigureEvidenceBlock],
    block_embeddings: dict[str, np.ndarray],
    distance_tiers: list[float],
) -> tuple[float, dict[str, Any]]:
    """改造前 S_link：全局 max 聚合，含无条件 local 窗口，无自匹配过滤。"""
    best_score = 0.0
    best_debug: dict[str, Any] = {
        "matched_query_block": None,
        "matched_query_idx": None,
        "matched_evidence_block": None,
        "evidence_source": None,
        "matched_ref": None,
        "raw_cosine": 0.0,
        "distance_weight": 0.0,
        "s_link": 0.0,
        "rerank_profile": "legacy",
    }

    if not query_block_pairs or not evidence_blocks:
        return best_score, best_debug

    for pair in query_block_pairs:
        q_emb = block_embeddings.get(pair.block.block_id)
        if q_emb is None:
            continue
        for evidence in evidence_blocks:
            e_emb = block_embeddings.get(evidence.block.block_id)
            if e_emb is None:
                continue
            raw_sim = cosine_sim(q_emb, e_emb)
            weight = distance_weight(
                block_distance(pair.block.block_idx, evidence.block.block_idx),
                distance_tiers,
            )
            score = raw_sim * weight
            if score > best_score:
                best_score = score
                best_debug = {
                    "matched_query_block": pair.block.block_id,
                    "matched_query_idx": pair.query_idx,
                    "matched_evidence_block": evidence.block.block_id,
                    "evidence_source": evidence.source,
                    "matched_ref": figure_ref_to_str(evidence.matched_ref),
                    "raw_cosine": round(raw_sim, 6),
                    "distance_weight": round(weight, 6),
                    "s_link": round(score, 6),
                    "rerank_profile": "legacy",
                }

    return best_score, best_debug


def link_score_summary_features(
    query_block_pairs: list[QueryBlockPair],
    evidence_blocks: list[FigureEvidenceBlock],
    block_embeddings: dict[str, np.ndarray],
    distance_tiers: list[float],
    *,
    top_k: int = 3,
    coverage_threshold: float = 0.35,
) -> LinkSummaryFeatures:
    """
    计算比 legacy global-max 更稳健的 Link 特征。

    - peak: 保留最强单点匹配；
    - topk_mean: source-wise Top-K 均值，降低偶然峰值影响；
    - query_coverage: sub-query 被 evidence 稳定响应的比例；
    - source_balance: explicit/local_prev/local_next 响应是否均衡。
    """
    source_scores: dict[str, list[float]] = {
        "explicit_ref": [],
        "local_prev": [],
        "local_next": [],
    }
    query_best: dict[int, float] = {}
    best_score = 0.0
    best_debug: dict[str, Any] = {
        "matched_query_block": None,
        "matched_query_idx": None,
        "matched_evidence_block": None,
        "evidence_source": None,
        "matched_ref": None,
        "raw_cosine": 0.0,
        "distance_weight": 0.0,
        "s_link": 0.0,
        "rerank_profile": "summary_features",
    }

    for pair in query_block_pairs:
        q_emb = block_embeddings.get(pair.block.block_id)
        if q_emb is None:
            continue
        for evidence in evidence_blocks:
            if pair.block.block_id == evidence.block.block_id:
                continue
            e_emb = block_embeddings.get(evidence.block.block_id)
            if e_emb is None:
                continue
            raw_sim = cosine_sim(q_emb, e_emb)
            weight = distance_weight(
                block_distance(pair.block.block_idx, evidence.block.block_idx),
                distance_tiers,
            )
            score = raw_sim * weight
            source_scores.setdefault(evidence.source, []).append(score)
            query_best[pair.query_idx] = max(query_best.get(pair.query_idx, 0.0), score)
            if score > best_score:
                best_score = score
                best_debug = {
                    "matched_query_block": pair.block.block_id,
                    "matched_query_idx": pair.query_idx,
                    "matched_evidence_block": evidence.block.block_id,
                    "evidence_source": evidence.source,
                    "matched_ref": figure_ref_to_str(evidence.matched_ref),
                    "raw_cosine": round(raw_sim, 6),
                    "distance_weight": round(weight, 6),
                    "s_link": round(score, 6),
                    "rerank_profile": "summary_features",
                }

    def topk_mean(scores: list[float]) -> float:
        positive = sorted((s for s in scores if s > 0.0), reverse=True)[: max(top_k, 1)]
        return float(np.mean(positive)) if positive else 0.0

    source_topk = {source: topk_mean(scores) for source, scores in source_scores.items()}
    nonzero_source_values = [v for v in source_topk.values() if v > 0.0]
    link_topk_mean = float(np.mean(nonzero_source_values)) if nonzero_source_values else 0.0
    explicit_strength = source_topk.get("explicit_ref", 0.0)
    local_values = [
        source_topk.get("local_prev", 0.0),
        source_topk.get("local_next", 0.0),
    ]
    local_strength = max(local_values)

    n_query_indices = len({pair.query_idx for pair in query_block_pairs})
    if n_query_indices:
        covered = sum(1 for score in query_best.values() if score >= coverage_threshold)
        query_coverage = covered / n_query_indices
    else:
        query_coverage = 0.0

    if len(nonzero_source_values) <= 1:
        source_balance = 0.0
    else:
        mean_val = float(np.mean(nonzero_source_values))
        std_val = float(np.std(nonzero_source_values))
        source_balance = max(0.0, 1.0 - std_val / (mean_val + 1e-9))

    debug = dict(best_debug)
    debug.update(
        {
            "link_peak": round(best_score, 6),
            "link_topk_mean": round(link_topk_mean, 6),
            "link_query_coverage": round(query_coverage, 6),
            "link_source_balance": round(source_balance, 6),
            "link_explicit_strength": round(explicit_strength, 6),
            "link_local_strength": round(local_strength, 6),
            "source_topk": {k: round(v, 6) for k, v in source_topk.items()},
            "query_best": {str(k): round(v, 6) for k, v in query_best.items()},
            "coverage_threshold": coverage_threshold,
            "top_k": top_k,
        }
    )
    return LinkSummaryFeatures(
        link_peak=best_score,
        link_topk_mean=link_topk_mean,
        link_query_coverage=query_coverage,
        link_source_balance=source_balance,
        link_explicit_strength=explicit_strength,
        link_local_strength=local_strength,
        evidence_count=len(evidence_blocks),
        debug=debug,
    )


def select_best_link_evidence(
    query_block_pairs: list[QueryBlockPair],
    evidence_blocks: list[FigureEvidenceBlock],
    block_embeddings: dict[str, np.ndarray],
    distance_tiers: list[float],
) -> tuple[FigureEvidenceBlock | None, dict[str, Any]]:
    """返回 S_link 最大匹配对应的 evidence block（legacy 全局 max 逻辑）。"""
    _, debug = link_score_legacy(
        query_block_pairs,
        evidence_blocks,
        block_embeddings,
        distance_tiers,
    )
    matched_id = debug.get("matched_evidence_block")
    if not matched_id:
        return None, debug
    for evidence in evidence_blocks:
        if evidence.block.block_id == matched_id:
            return evidence, debug
    return None, debug


def caption_or_neighbors_have_deictic(
    figure: FigureMeta,
    blocks: list[Block],
) -> bool:
    """判断图注或邻近正文是否含指代性表述。"""
    if DEICTIC_PATTERN.search(figure.caption or ""):
        return True
    pos = figure.pos
    if pos < 0:
        return False
    for direction in (-1, 1):
        candidates = (
            [b for b in blocks if b.block_type == "text" and b.block_idx < pos]
            if direction < 0
            else [b for b in blocks if b.block_type == "text" and b.block_idx > pos]
        )
        if not candidates:
            continue
        block = max(candidates, key=lambda b: b.block_idx) if direction < 0 else min(
            candidates, key=lambda b: b.block_idx
        )
        if DEICTIC_PATTERN.search(block.text):
            return True
    return False


def _is_explicit_evidence(evidence: FigureEvidenceBlock) -> bool:
    return evidence.source == "explicit_ref" or evidence.matched_ref is not None


def link_score_gated(
    query_block_pairs: list[QueryBlockPair],
    evidence_blocks: list[FigureEvidenceBlock],
    block_embeddings: dict[str, np.ndarray],
    distance_tiers: list[float],
    params: LinkFusionParams,
) -> tuple[float, float, float, dict[str, Any]]:
    """分源计算 S_link；跳过 query/evidence 同 block 的自匹配。"""
    best_explicit = 0.0
    best_local = 0.0
    best_debug: dict[str, Any] = {
        "matched_query_block": None,
        "matched_query_idx": None,
        "matched_evidence_block": None,
        "evidence_source": None,
        "matched_ref": None,
        "raw_cosine": 0.0,
        "distance_weight": 0.0,
        "s_link_explicit": 0.0,
        "s_link_local": 0.0,
        "s_link": 0.0,
        "effective_alpha": 1.0,
    }

    if not query_block_pairs or not evidence_blocks:
        return 0.0, 0.0, 0.0, best_debug

    for pair in query_block_pairs:
        q_emb = block_embeddings.get(pair.block.block_id)
        if q_emb is None:
            continue
        for evidence in evidence_blocks:
            if pair.block.block_id == evidence.block.block_id:
                continue
            e_emb = block_embeddings.get(evidence.block.block_id)
            if e_emb is None:
                continue
            raw_sim = cosine_sim(q_emb, e_emb)
            weight = distance_weight(
                block_distance(pair.block.block_idx, evidence.block.block_idx),
                distance_tiers,
            )
            score = raw_sim * weight
            is_explicit = _is_explicit_evidence(evidence)
            is_local_only = evidence.source in {"local_prev", "local_next"} and not is_explicit

            if is_local_only:
                if score > best_local:
                    best_local = score
                    best_debug = _link_debug_entry(
                        pair, evidence, raw_sim, weight, score, 0.0, score, "local"
                    )
            elif score > best_explicit:
                best_explicit = score
                best_debug = _link_debug_entry(
                    pair, evidence, raw_sim, weight, score, score, 0.0, "explicit"
                )

    local_adj = min(best_local * params.local_gamma, params.local_cap)
    s_link = max(best_explicit, local_adj)

    if best_explicit >= params.explicit_link_threshold:
        effective_alpha = params.alpha
        link_mode = "explicit"
    elif local_adj > 1e-6:
        effective_alpha = params.alpha_local
        link_mode = "local"
    else:
        effective_alpha = 1.0
        link_mode = "direct_only"

    best_debug.update(
        {
            "s_link_explicit": round(best_explicit, 6),
            "s_link_local": round(best_local, 6),
            "s_link_local_adjusted": round(local_adj, 6),
            "s_link": round(s_link, 6),
            "effective_alpha": round(effective_alpha, 6),
            "link_mode": link_mode,
        }
    )
    return s_link, best_explicit, local_adj, best_debug


def _link_debug_entry(
    pair: QueryBlockPair,
    evidence: FigureEvidenceBlock,
    raw_sim: float,
    weight: float,
    score: float,
    s_explicit: float,
    s_local: float,
    bucket: str,
) -> dict[str, Any]:
    return {
        "matched_query_block": pair.block.block_id,
        "matched_query_idx": pair.query_idx,
        "matched_evidence_block": evidence.block.block_id,
        "evidence_source": evidence.source,
        "matched_ref": figure_ref_to_str(evidence.matched_ref),
        "raw_cosine": round(raw_sim, 6),
        "distance_weight": round(weight, 6),
        "s_link_explicit": round(s_explicit, 6),
        "s_link_local": round(s_local, 6),
        "s_link": round(score, 6),
        "link_bucket": bucket,
    }


def fuse_direct_and_link(
    s_direct: float,
    s_link: float,
    effective_alpha: float,
) -> float:
    return effective_alpha * s_direct + (1.0 - effective_alpha) * s_link


def co_occurrence_score(
    cq_blocks: list[Block],
    cf_blocks: list[Block],
    figure: FigureMeta,
    block_embeddings: dict[str, np.ndarray],
    distance_tiers: list[float],
) -> float:
    """兼容旧 API；LG-JSSF reranker 不再调用此无归一化共现分。"""
    return 0.0
