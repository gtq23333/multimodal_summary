from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from m3sum.data.schema import Block, FigureMeta, SubQuery
from m3sum.stage2_rerank.tokenize import tokenize as _tokenize


class EmbeddingCache:
    def __init__(self, cache_dir: Path, embedder):
        self.cache_dir = cache_dir
        self.embedder = embedder
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, paper_id: str) -> Path:
        return self.cache_dir / f"{paper_id}.npz"

    def load_or_compute(
        self,
        paper_id: str,
        blocks: list[Block],
        figures: list[FigureMeta],
        dry_run: bool = False,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        cache_path = self._cache_path(paper_id)
        block_ids = [b.block_id for b in blocks]
        fig_ids = [f.image_hash for f in figures]

        if cache_path.is_file():
            data = np.load(cache_path, allow_pickle=True)
            block_embs = {bid: data[f"block_{bid}"] for bid in block_ids if f"block_{bid}" in data}
            fig_embs = {fid: data[f"fig_{fid}"] for fid in fig_ids if f"fig_{fid}" in data}
            if len(block_embs) == len(blocks) and len(fig_embs) == len(figures):
                return block_embs, fig_embs

        if dry_run:
            dim = 64
            block_embs = {b.block_id: np.random.randn(dim).astype(np.float32) for b in blocks}
            fig_embs = {f.image_hash: np.random.randn(dim).astype(np.float32) for f in figures}
            return block_embs, fig_embs

        block_texts = [b.text for b in blocks]
        fig_texts = [f.caption or f.image_hash for f in figures]

        block_vectors = self.embedder.embed_batch(block_texts, batch_size=10) if block_texts else []
        fig_vectors = self.embedder.embed_batch(fig_texts, batch_size=10) if fig_texts else []

        block_embs = {b.block_id: np.array(v, dtype=np.float32) for b, v in zip(blocks, block_vectors)}
        fig_embs = {f.image_hash: np.array(v, dtype=np.float32) for f, v in zip(figures, fig_vectors)}

        save_dict = {}
        for bid, emb in block_embs.items():
            save_dict[f"block_{bid}"] = emb
        for fid, emb in fig_embs.items():
            save_dict[f"fig_{fid}"] = emb
        np.savez(cache_path, **save_dict)
        return block_embs, fig_embs


class HybridRetriever:
    def __init__(
        self,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        top_p: int = 20,
    ):
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.top_p = top_p

    def search(
        self,
        query: SubQuery,
        blocks: list[Block],
        block_embeddings: dict[str, np.ndarray],
        query_embedding: np.ndarray | None = None,
    ) -> list[Block]:
        text_blocks = [b for b in blocks if b.block_type == "text" and b.text.strip()]
        if not text_blocks:
            return []

        corpus = [_tokenize(b.text) for b in text_blocks]
        bm25 = BM25Okapi(corpus)
        q_tokens = _tokenize(query.query + " " + " ".join(query.keywords))
        bm25_scores = bm25.get_scores(q_tokens)

        if bm25_scores.max() > bm25_scores.min():
            bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        else:
            bm25_norm = bm25_scores

        vec_scores = np.zeros(len(text_blocks))
        if query_embedding is not None:
            for i, b in enumerate(text_blocks):
                emb = block_embeddings.get(b.block_id)
                if emb is not None:
                    vec_scores[i] = float(np.dot(query_embedding, emb) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-9
                    ))
            if vec_scores.max() > vec_scores.min():
                vec_norm = (vec_scores - vec_scores.min()) / (vec_scores.max() - vec_scores.min())
            else:
                vec_norm = vec_scores
        else:
            vec_norm = vec_scores

        combined = self.bm25_weight * bm25_norm + self.vector_weight * vec_norm
        ranked_idx = np.argsort(combined)[::-1][: self.top_p]
        return [text_blocks[i] for i in ranked_idx]
