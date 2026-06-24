from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from m3sum.clients.dashscope_vl_rerank import DashScopeVLRerankClient, DocumentMode
from m3sum.data.schema import FigureMeta, SubQuery
from m3sum.stage2_rerank.figure_filter import select_body_caption_figures

logger = logging.getLogger(__name__)


class VLRerankScoreCache:
    """Cache qwen3-vl-rerank relevance scores per paper, query index, and document mode."""

    def __init__(
        self,
        cache_dir: Path,
        client: DashScopeVLRerankClient | None,
        *,
        document_mode: DocumentMode,
        dry_run: bool = False,
    ):
        self.cache_dir = cache_dir
        self.client = client
        self.document_mode = document_mode
        self.dry_run = dry_run
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
        cached: dict[str, dict[str, float]] = {}

        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        elif self.document_mode == DocumentMode.IMAGE_ONLY:
            legacy_path = self.cache_dir.parent / f"{paper_id}.json"
            if legacy_path.is_file():
                cached = json.loads(legacy_path.read_text(encoding="utf-8"))

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
                logger.info(
                    "  [VL-Rerank] mode=%s paper=%s query_idx=%d text=%r candidates=%d/%d",
                    self.document_mode.value,
                    paper_id,
                    q_idx,
                    sub_query.query[:80],
                    len(captioned_figs),
                    len(all_figures),
                )
                reranked = self.client.rerank_figures(
                    sub_query.query,
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
            cache_path.write_text(
                json.dumps(cached, ensure_ascii=False, indent=2),
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
