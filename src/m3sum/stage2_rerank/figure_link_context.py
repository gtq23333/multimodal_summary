"""
为 Qwen3-VL-Rerank 选取与 LG-JSSF S_link 一致的最佳关联正文 chunk。

候选 evidence = 图号显式引用 + 图注上下 local 窗口 + 正文 caption_ref 块；
在 (赛题 hybrid 召回块 × evidence) 中取加权 cosine 最大者作为 context。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from m3sum.clients.openai_embedder import OpenAIEmbedder
from m3sum.config import PipelineConfig, resolve_api_credentials
from m3sum.data.corpus_adapter import CorpusAdapter
from m3sum.data.schema import Block, FigureMeta, SubQuery
from m3sum.stage2_rerank.baselines.base import Stage2Sample
from m3sum.stage2_rerank.caption_refs import (
    extract_labeled_caption_refs,
    parse_figure_index_from_caption,
    parse_figure_label_from_caption,
)
from m3sum.stage2_rerank.caption_regex import match_caption_blocks
from m3sum.stage2_rerank.co_occurrence import (
    FigureEvidenceBlock,
    QueryBlockPair,
    collect_figure_evidence_blocks,
    select_best_link_evidence,
)
from m3sum.stage2_rerank.hybrid_retriever import EmbeddingCache, HybridRetriever

_MAX_CONTEXT_CHARS = 1200


def _merge_caption_ref_evidence(
    figure: FigureMeta,
    figure_index: tuple[int, ...] | None,
    evidence_blocks: list[FigureEvidenceBlock],
    caption_ref_blocks: list[Block],
) -> list[FigureEvidenceBlock]:
    """将正文 caption_ref / 正则匹配块中引用该图号的块并入 evidence。"""
    if figure_index is None:
        return evidence_blocks

    figure_label = parse_figure_label_from_caption(figure.caption)
    by_id = {ev.block.block_id: ev for ev in evidence_blocks}

    for block in caption_ref_blocks:
        if block.block_id in by_id:
            continue
        labeled_refs = extract_labeled_caption_refs(block.text)
        if not any(
            ref == figure_index and (figure_label is None or label == figure_label)
            for label, ref in labeled_refs
        ):
            continue
        by_id[block.block_id] = FigureEvidenceBlock(
            block=block,
            source="caption_ref",
            matched_ref=figure_index,
        )

    return list(by_id.values())


def _trim_context(text: str, max_chars: int = _MAX_CONTEXT_CHARS) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


class FigureLinkContextSelector:
    """按主方法 legacy S_link 逻辑，为每张图挑选最佳关联 chunk。"""

    def __init__(
        self,
        config: PipelineConfig,
        embedder: OpenAIEmbedder | None,
        dry_run: bool = False,
    ):
        self.config = config
        self.dry_run = dry_run
        self.corpus = CorpusAdapter(config)
        self.hybrid = HybridRetriever(
            bm25_weight=config.bm25_weight,
            vector_weight=config.vector_weight,
            top_p=config.top_p,
        )
        self.embed_cache = EmbeddingCache(
            config.embed_cache_dir,
            embedder,
        )
        self._query_emb_cache: dict[str, list[np.ndarray | None]] = {}

    def _query_embeddings(
        self,
        paper_id: str,
        sub_queries: list[SubQuery],
        block_embs: dict[str, np.ndarray],
    ) -> list[np.ndarray | None]:
        if paper_id in self._query_emb_cache:
            return self._query_emb_cache[paper_id]

        query_embeddings: list[np.ndarray | None] = []
        for q in sub_queries:
            if self.dry_run:
                dim = next(iter(block_embs.values())).shape[0] if block_embs else 64
                query_embeddings.append(np.random.randn(dim).astype(np.float32))
            else:
                vec = self.embedder.embed_one(q.query + " " + " ".join(q.keywords))
                query_embeddings.append(np.array(vec, dtype=np.float32))
        self._query_emb_cache[paper_id] = query_embeddings
        return query_embeddings

    @property
    def embedder(self) -> OpenAIEmbedder | None:
        return self.embed_cache.embedder

    def contexts_for_sample(
        self,
        sample: Stage2Sample,
    ) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
        """
        返回 (figure_hash -> context_text, figure_hash -> debug)。
        无匹配 evidence 时 context 为空字符串（VL 文档退化为 caption+image）。
        """
        doc = self.corpus.load_document(sample.paper_id)
        blocks = doc.blocks
        block_embs, _ = self.embed_cache.load_or_compute(
            sample.paper_id,
            blocks,
            sample.figures,
            dry_run=self.dry_run,
        )
        query_embeddings = self._query_embeddings(
            sample.paper_id,
            sample.sub_queries,
            block_embs,
        )

        query_block_pairs: list[QueryBlockPair] = []
        for i, q in enumerate(sample.sub_queries):
            q_emb = query_embeddings[i] if i < len(query_embeddings) else None
            for b in self.hybrid.search(q, blocks, block_embs, q_emb):
                query_block_pairs.append(
                    QueryBlockPair(query_idx=i, block=b, query_embedding=q_emb)
                )

        caption_result = match_caption_blocks(blocks, self.config.caption_patterns)
        caption_ref_blocks = caption_result.matched_blocks
        local_mode = str(
            self.config.raw.get("rerank", {}).get("local_window_mode", "always")
        )

        contexts: dict[str, str] = {}
        debug_by_fig: dict[str, dict[str, Any]] = {}

        for fig in sample.figures:
            figure_index = parse_figure_index_from_caption(fig.caption)
            evidence_blocks = collect_figure_evidence_blocks(
                fig,
                figure_index,
                blocks,
                local_window_mode=local_mode,
            )
            evidence_blocks = _merge_caption_ref_evidence(
                fig,
                figure_index,
                evidence_blocks,
                caption_ref_blocks,
            )
            best_evidence, link_debug = select_best_link_evidence(
                query_block_pairs,
                evidence_blocks,
                block_embs,
                self.config.distance_tiers,
            )
            context_text = ""
            if best_evidence is not None:
                context_text = _trim_context(best_evidence.block.text)

            contexts[fig.image_hash] = context_text
            debug_by_fig[fig.image_hash] = {
                "figure_index": figure_index,
                "evidence_count": len(evidence_blocks),
                "context_chars": len(context_text),
                "link": link_debug,
            }

        return contexts, debug_by_fig


def build_figure_link_context_selector(
    config: PipelineConfig,
    dry_run: bool = False,
) -> FigureLinkContextSelector:
    embedder = None
    if not dry_run:
        creds = resolve_api_credentials(config)
        embedder = OpenAIEmbedder(config.embed_model, creds)
    return FigureLinkContextSelector(config, embedder, dry_run=dry_run)
