from __future__ import annotations

from pathlib import Path

import numpy as np

from m3sum.data.schema import FigureMeta, SubQuery


class TextEmbeddingCache:
    """缓存 query 与 caption 文本 embedding（NPZ）。"""

    def __init__(self, cache_dir: Path, embedder, dry_run: bool = False):
        self.cache_dir = cache_dir
        self.embedder = embedder
        self.dry_run = dry_run
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, paper_id: str) -> Path:
        return self.cache_dir / f"{paper_id}.npz"

    def load_or_compute(
        self,
        paper_id: str,
        sub_queries: list[SubQuery],
        figures: list[FigureMeta],
    ) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
        cache_path = self._cache_path(paper_id)
        query_keys = [f"q_{i}" for i in range(len(sub_queries))]
        caption_keys = [f"cap_{f.image_hash}" for f in figures]

        if cache_path.is_file():
            data = np.load(cache_path, allow_pickle=True)
            if all(k in data for k in query_keys) and all(k in data for k in caption_keys):
                query_embs = [data[k] for k in query_keys]
                caption_embs = {f.image_hash: data[f"cap_{f.image_hash}"] for f in figures}
                return query_embs, caption_embs

        if self.dry_run:
            dim = 64
            query_embs = [np.random.randn(dim).astype(np.float32) for _ in sub_queries]
            caption_embs = {
                f.image_hash: np.random.randn(dim).astype(np.float32) for f in figures
            }
            return query_embs, caption_embs

        query_texts = [q.query + " " + " ".join(q.keywords) for q in sub_queries]
        caption_texts = [f.caption or "" for f in figures]

        query_vectors = (
            self.embedder.embed_batch(query_texts, batch_size=10) if query_texts else []
        )
        caption_vectors = (
            self.embedder.embed_batch(caption_texts, batch_size=10) if caption_texts else []
        )

        query_embs = [np.array(v, dtype=np.float32) for v in query_vectors]
        caption_embs = {
            f.image_hash: np.array(v, dtype=np.float32)
            for f, v in zip(figures, caption_vectors)
        }

        save_dict = {f"q_{i}": emb for i, emb in enumerate(query_embs)}
        for f in figures:
            save_dict[f"cap_{f.image_hash}"] = caption_embs[f.image_hash]
        np.savez(cache_path, **save_dict)
        return query_embs, caption_embs
