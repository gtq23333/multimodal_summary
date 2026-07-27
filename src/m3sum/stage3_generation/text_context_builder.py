from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from m3sum.config import PipelineConfig
from m3sum.data.schema import Block, DocumentBundle, SubQuery
from m3sum.stage2_rerank.hybrid_retriever import HybridRetriever

BODY_IMG_RE = re.compile(r"!\[\]\(images/[^)]+\)", re.I)


@dataclass
class TextEvidence:
    block_id: str
    block_idx: int
    query_dimension: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "block_idx": self.block_idx,
            "query_dimension": self.query_dimension,
            "text": self.text,
        }


def strip_body_images(body_text: str) -> str:
    cleaned = BODY_IMG_RE.sub("", body_text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + f"\n\n[内容已截断，原长度 {len(text)} 字符]"


def collect_text_evidence(
    config: PipelineConfig,
    doc: DocumentBundle,
    sub_queries: list[SubQuery],
    *,
    top_p: int | None = None,
    max_chars: int | None = None,
) -> list[TextEvidence]:
    gen_cfg = config.stage3_generation_config
    top_p = int(top_p or gen_cfg.get("text_context_top_p", 8))
    max_chars = int(max_chars or gen_cfg.get("max_retrieved_chars", 18000))
    retriever = HybridRetriever(
        bm25_weight=config.bm25_weight,
        vector_weight=0.0,
        top_p=top_p,
        query_use_keywords=config.query_use_keywords,
    )

    seen: set[str] = set()
    evidence: list[TextEvidence] = []
    used_chars = 0
    for query in sub_queries:
        blocks = retriever.search(query, doc.blocks, block_embeddings={}, query_embedding=None)
        for block in blocks:
            text = _clean_block_text(block)
            if not text or block.block_id in seen:
                continue
            if used_chars + len(text) > max_chars:
                remaining = max_chars - used_chars
                if remaining <= 200:
                    return evidence
                text = truncate_text(text, remaining)
            evidence.append(
                TextEvidence(
                    block_id=block.block_id,
                    block_idx=block.block_idx,
                    query_dimension=query.dimension,
                    text=text,
                )
            )
            seen.add(block.block_id)
            used_chars += len(text)
            if used_chars >= max_chars:
                return evidence
    return evidence


def format_evidence_block(evidence: list[TextEvidence]) -> str:
    lines: list[str] = []
    for idx, item in enumerate(evidence, start=1):
        lines.append(
            f"[E{idx} | {item.query_dimension} | block={item.block_id}]\n{item.text}"
        )
    return "\n\n".join(lines)


def build_generation_context(
    config: PipelineConfig,
    doc: DocumentBundle,
    sub_queries: list[SubQuery],
) -> dict[str, Any]:
    gen_cfg = config.stage3_generation_config
    max_body_chars = int(gen_cfg.get("max_body_chars", 120000))
    evidence = collect_text_evidence(config, doc, sub_queries)
    return {
        "body_text": truncate_text(strip_body_images(doc.body_text), max_body_chars),
        "retrieved_evidence": [e.to_dict() for e in evidence],
        "retrieved_evidence_text": format_evidence_block(evidence),
    }


def _clean_block_text(block: Block) -> str:
    text = strip_body_images(block.text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text
