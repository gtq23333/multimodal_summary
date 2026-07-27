from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from m3sum.clients.dashscope_vl_rerank import DashScopeVLRerankClient, DocumentMode
from m3sum.data.schema import FigureMeta, SubQuery
from m3sum.stage2_rerank.figure_filter import select_body_caption_figures
from m3sum.stage2_rerank.query_text import sub_query_search_texts

logger = logging.getLogger(__name__)

_CACHE_META_KEY = "_meta"


def _query_texts_for_cache(
    sub_queries: list[SubQuery],
    *,
    query_use_keywords: bool,
) -> list[str]:
    return sub_query_search_texts(sub_queries, use_keywords=query_use_keywords)


def _load_vl_rerank_cache(
    cache_path: Path,
    sub_queries: list[SubQuery],
    *,
    query_use_keywords: bool,
) -> dict[str, dict[str, float]]:
    if not cache_path.is_file():
        return {}
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    meta = raw.get(_CACHE_META_KEY)
    expected = _query_texts_for_cache(
        sub_queries,
        query_use_keywords=query_use_keywords,
    )
    if (
        not meta
        or meta.get("query_texts") != expected
        or meta.get("query_use_keywords") != query_use_keywords
    ):
        return {}
    return {k: v for k, v in raw.items() if k != _CACHE_META_KEY}


class VLRerankScoreCache:
    """Cache qwen3-vl-rerank relevance scores per paper, query index, and document mode."""

    def __init__(
        self,
        cache_dir: Path,
        client: DashScopeVLRerankClient | None,
        *,
        document_mode: DocumentMode,
        dry_run: bool = False,
        query_use_keywords: bool = True,
    ):
        self.cache_dir = cache_dir
        self.client = client
        self.document_mode = document_mode
        self.dry_run = dry_run
        self.query_use_keywords = query_use_keywords
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, paper_id: str) -> Path:
        return self.cache_dir / f"{paper_id}.json"

    def load_or_compute(
        self,
        paper_id: str,
        sub_queries: list[SubQuery],
        all_figures: list[FigureMeta],
        context_by_figure: dict[str, str] | None = None,
    ) -> list[dict[str, float]]:
        """
        Return per-query score maps for all_figures (figure_hash -> score).
        API 仅对带图注的正文图片打分，其余 figure 分数为 0。
        """
        captioned_figs = select_body_caption_figures(all_figures)
        all_hashes = [f.image_hash for f in all_figures]
        candidate_hashes = [f.image_hash for f in captioned_figs]

        cache_path = self._cache_path(paper_id)
        cached = _load_vl_rerank_cache(
            cache_path,
            sub_queries,
            query_use_keywords=self.query_use_keywords,
        )
        if not cached and self.document_mode == DocumentMode.IMAGE_ONLY:
            legacy_path = self.cache_dir.parent / f"{paper_id}.json"
            cached = _load_vl_rerank_cache(
                legacy_path,
                sub_queries,
                query_use_keywords=self.query_use_keywords,
            )

        query_scores: list[dict[str, float]] = []

        for q_idx, sub_query in enumerate(sub_queries):
            key = str(q_idx)
            if key in cached and all(h in cached[key] for h in candidate_hashes):
                merged = {h: 0.0 for h in all_hashes}
                merged.update(cached[key])
                query_scores.append(merged)
                continue

            if self.dry_run:
                candidate_scores = _dry_run_scores(
                    paper_id,
                    q_idx,
                    candidate_hashes,
                    variant=self.document_mode.value,
                )
            else:
                if self.client is None:
                    raise RuntimeError("Qwen3-VL-Rerank 需要 client 或 dry_run=True")
                query_text = sub_query.search_text(
                    use_keywords=self.query_use_keywords
                )
                logger.info(
                    "  [VL-Rerank] mode=%s paper=%s query_idx=%d text=%r candidates=%d/%d",
                    self.document_mode.value,
                    paper_id,
                    q_idx,
                    query_text[:80],
                    len(captioned_figs),
                    len(all_figures),
                )
                reranked = self.client.rerank_figures(
                    query_text,
                    captioned_figs,
                    mode=self.document_mode,
                    context_by_figure=context_by_figure,
                )
                candidate_scores = {
                    captioned_figs[idx].image_hash: score for idx, score in reranked
                }

            for h in candidate_hashes:
                candidate_scores.setdefault(h, 0.0)

            cached[key] = candidate_scores
            merged = {h: 0.0 for h in all_hashes}
            merged.update(candidate_scores)
            query_scores.append(merged)

            if not self.dry_run:
                payload = dict(cached)
                payload[_CACHE_META_KEY] = {
                    "query_use_keywords": self.query_use_keywords,
                    "query_texts": _query_texts_for_cache(
                        sub_queries,
                        query_use_keywords=self.query_use_keywords,
                    ),
                }
                cache_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        return query_scores


def _dry_run_scores(
    paper_id: str,
    query_idx: int,
    figure_hashes: list[str],
    *,
    variant: str,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for fig_hash in figure_hashes:
        seed = f"{variant}:{paper_id}:{query_idx}:{fig_hash}".encode()
        digest = hashlib.md5(seed).hexdigest()
        scores[fig_hash] = int(digest[:8], 16) / 0xFFFFFFFF
    return scores
