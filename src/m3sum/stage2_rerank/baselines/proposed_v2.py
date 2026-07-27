from __future__ import annotations

import json
import logging

import numpy as np

from m3sum.clients.openai_embedder import OpenAIEmbedder
from m3sum.config import PipelineConfig, resolve_api_credentials
from m3sum.data.corpus_adapter import CorpusAdapter
from m3sum.pipeline.runner import PipelineRunner
from m3sum.stage2_rerank.baselines.base import RankedFigure, Stage2Sample, build_ranked_list
from m3sum.stage2_rerank.clip_utils import ClipImageEmbeddingCache, load_clip_model
from m3sum.stage2_rerank.hybrid_retriever import EmbeddingCache
from m3sum.stage2_rerank.main_method import main_cluster_scorer, stage2_query_config_matches
from m3sum.stage2_rerank.summary_proposal import (
    build_summary_figure_features,
    normalize_cluster_priors,
)

logger = logging.getLogger(__name__)


class ProposedV2Ranker:
    """
    增强版启发式候选图排序：summary figure proposal features + paper-level normalization。

    不替换 legacy Proposed；用于比较更适合作为候选池 proposal 的可解释特征组合。
    """

    method_name = "Proposed-v2"

    def __init__(
        self,
        config: PipelineConfig,
        dry_run: bool = False,
        image_cache: ClipImageEmbeddingCache | None = None,
    ):
        self.config = config
        self.dry_run = dry_run
        self._runner: PipelineRunner | None = None
        self._corpus = CorpusAdapter(config)
        self._embedder = None if dry_run else OpenAIEmbedder(config.embed_model, resolve_api_credentials(config))
        self._embed_cache = EmbeddingCache(config.embed_cache_dir, self._embedder)
        self._image_cache = image_cache
        self._owns_clip_cache = image_cache is None
        self._embeddings_by_paper: dict[str, dict[str, np.ndarray | None]] = {}
        self._debug_by_paper: dict[str, list[dict]] = {}
        self._query_embeddings_by_paper: dict[str, list[np.ndarray | None]] = {}

    def _get_runner(self) -> PipelineRunner:
        if self._runner is None:
            self._runner = PipelineRunner(
                self.config,
                dry_run=self.dry_run,
                from_cache=True,
            )
        return self._runner

    def _load_stage2_items(self, paper_id: str) -> list[dict]:
        path = self.config.stage2_dir / f"{paper_id}.json"
        if not path.is_file() or not stage2_query_config_matches(self.config, paper_id):
            self._get_runner().run_stage2(paper_id)
        if not path.is_file():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("all_scores", []))

    def _get_image_cache(self) -> ClipImageEmbeddingCache | None:
        if self.dry_run or main_cluster_scorer(self.config) is None:
            return None
        if self._image_cache is None and self._owns_clip_cache:
            encoder = load_clip_model(self.config.cluster_prior_clip_model)
            self._image_cache = ClipImageEmbeddingCache(
                self.config.stage2_eval_clip_cache_dir,
                clip_encoder=encoder,
                dry_run=False,
            )
        return self._image_cache

    def _cluster_priors(self, sample: Stage2Sample) -> dict[str, float]:
        scorer = main_cluster_scorer(self.config)
        cache = self._get_image_cache()
        if scorer is None or cache is None:
            return {}
        if sample.paper_id not in self._embeddings_by_paper:
            self._embeddings_by_paper[sample.paper_id] = cache.load_or_compute(
                sample.paper_id,
                sample.figures,
            )
        query_text = " ".join(q.query for q in sample.sub_queries)
        priors: dict[str, float] = {}
        for fig in sample.figures:
            emb = self._embeddings_by_paper[sample.paper_id].get(fig.image_hash)
            prior, _ = scorer.score_with_context(emb, query_text=query_text)
            priors[fig.image_hash] = prior
        if bool(self.config.raw.get("cluster_prior", {}).get("relative_prior", True)):
            return normalize_cluster_priors(priors)
        return priors

    def _query_embeddings(
        self,
        sample: Stage2Sample,
        block_embeddings: dict[str, np.ndarray],
    ) -> list[np.ndarray | None]:
        if sample.paper_id in self._query_embeddings_by_paper:
            return self._query_embeddings_by_paper[sample.paper_id]

        embeddings: list[np.ndarray | None] = []
        for query in sample.sub_queries:
            if self.dry_run:
                dim = next(iter(block_embeddings.values())).shape[0] if block_embeddings else 64
                embeddings.append(np.random.randn(dim).astype(np.float32))
            else:
                assert self._embedder is not None
                vec = self._embedder.embed_one(
                    query.search_text(use_keywords=self.config.query_use_keywords)
                )
                embeddings.append(np.array(vec, dtype=np.float32))
        self._query_embeddings_by_paper[sample.paper_id] = embeddings
        return embeddings

    def rank(self, sample: Stage2Sample) -> list[RankedFigure]:
        stage2_items = self._load_stage2_items(sample.paper_id)
        if not stage2_items:
            logger.error("Proposed-v2：无法读取 Stage-2 特征 %s", sample.paper_id)
            return []

        doc = self._corpus.load_document(sample.paper_id)
        block_embs, fig_embs = self._embed_cache.load_or_compute(
            sample.paper_id,
            doc.blocks,
            doc.figures,
            dry_run=self.dry_run,
        )
        query_embeddings = self._query_embeddings(sample, block_embs)
        cluster_priors = self._cluster_priors(sample)
        features = build_summary_figure_features(
            self.config,
            sub_queries=sample.sub_queries,
            figures=sample.figures,
            blocks=doc.blocks,
            block_embeddings=block_embs,
            figure_embeddings=fig_embs,
            query_embeddings=query_embeddings,
            stage2_items=stage2_items,
            cluster_priors_by_figure=cluster_priors,
        )
        self._debug_by_paper[sample.paper_id] = [
            {
                "figure_id": feature.figure_id,
                "caption": feature.caption[:120],
                **feature.debug["summary_proposal"],
                "link": feature.debug.get("link", {}),
                "direct": feature.debug.get("direct", {}),
                "legacy": feature.debug.get("legacy", {}),
            }
            for feature in sorted(features, key=lambda f: f.score, reverse=True)
        ]
        return build_ranked_list(
            [(feature.figure_id, feature.score) for feature in features],
            self.method_name,
        )

    def debug_for_sample(self, sample: Stage2Sample) -> list[dict]:
        if sample.paper_id not in self._debug_by_paper:
            self.rank(sample)
        return self._debug_by_paper.get(sample.paper_id, [])
