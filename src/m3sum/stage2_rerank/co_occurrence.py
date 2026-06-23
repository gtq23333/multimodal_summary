from __future__ import annotations

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
) -> list[FigureEvidenceBlock]:
    """为单张图收集显式引用块与图片上下各一个正文 chunk。"""
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
) -> tuple[float, dict[str, Any]]:
    """计算 LG-JSSF 的 S_link：query block 与 evidence block 竞争取最大相似度。"""
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
                }

    return best_score, best_debug


def co_occurrence_score(
    cq_blocks: list[Block],
    cf_blocks: list[Block],
    figure: FigureMeta,
    block_embeddings: dict[str, np.ndarray],
    distance_tiers: list[float],
) -> float:
    """兼容旧 API；LG-JSSF reranker 不再调用此无归一化共现分。"""
    return 0.0
