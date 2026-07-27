from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from m3sum.data.schema import FigureMeta, SubQuery
from m3sum.stage2_rerank.query_text import sub_query_search_texts


class TextEmbeddingCache:
    """缓存 query 与 caption 文本 embedding（NPZ）。"""

    def __init__(
        self,
        cache_dir: Path,
        embedder,
        dry_run: bool = False,
        *,
        query_use_keywords: bool = True,
    ):
        self.cache_dir = cache_dir
        self.embedder = embedder
        self.dry_run = dry_run
        self.query_use_keywords = query_use_keywords
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, paper_id: str) -> Path:
        return self.cache_dir / f"{paper_id}.npz"

    def _meta_path(self, paper_id: str) -> Path:
        return self.cache_dir / f"{paper_id}.meta.json"

    def _load_meta(self, paper_id: str) -> dict | None:
        meta_path = self._meta_path(paper_id)
        if not meta_path.is_file():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _meta_matches(
        self,
        paper_id: str,
        sub_queries: list[SubQuery],
    ) -> bool:
        meta = self._load_meta(paper_id)
        if not meta:
            return False
        expected = sub_query_search_texts(
            sub_queries,
            use_keywords=self.query_use_keywords,
        )
        return (
            meta.get("query_use_keywords") == self.query_use_keywords
            and meta.get("query_texts") == expected
        )

    def load_or_compute(
        self,
        paper_id: str,
        sub_queries: list[SubQuery],
        figures: list[FigureMeta],
    ) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
        cache_path = self._cache_path(paper_id)
        query_keys = [f"q_{i}" for i in range(len(sub_queries))]
        caption_keys = [f"cap_{f.image_hash}" for f in figures]

        if cache_path.is_file() and self._meta_matches(paper_id, sub_queries):
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

        query_texts = sub_query_search_texts(
            sub_queries,
            use_keywords=self.query_use_keywords,
        )
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
        self._meta_path(paper_id).write_text(
            json.dumps(
                {
                    "query_use_keywords": self.query_use_keywords,
                    "query_texts": query_texts,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return query_embs, caption_embs
