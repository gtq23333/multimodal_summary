from __future__ import annotations

import json
import logging

from m3sum.config import PipelineConfig
from m3sum.pipeline.runner import PipelineRunner
from m3sum.stage2_rerank.baselines.base import RankedFigure, Stage2Sample
from m3sum.stage2_rerank.clip_utils import ClipImageEmbeddingCache, load_clip_model
from m3sum.stage2_rerank.main_method import main_cluster_scorer, rank_main_method

logger = logging.getLogger(__name__)


class ProposedRanker:
    """
    主方法：Legacy LG-JSSF + ClusterPrior（additive）。

    stage2 JSON 已含 cluster 融合分时直接读取；否则在评估时在线叠加 ClusterPrior。
    """

    method_name = "Proposed"

    def __init__(
        self,
        config: PipelineConfig,
        dry_run: bool = False,
        image_cache: ClipImageEmbeddingCache | None = None,
    ):
        self.config = config
        self.dry_run = dry_run
        self._runner: PipelineRunner | None = None
        self._image_cache = image_cache
        self._embeddings_by_paper: dict[str, dict] = {}
        self._owns_clip_cache = image_cache is None

    def _get_runner(self) -> PipelineRunner:
        if self._runner is None:
            self._runner = PipelineRunner(
                self.config,
                dry_run=self.dry_run,
                from_cache=True,
            )
        return self._runner

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

    def _load_from_json(self, paper_id: str) -> list[RankedFigure] | None:
        stage2_path = self.config.stage2_dir / f"{paper_id}.json"
        if not stage2_path.is_file():
            return None
        data = json.loads(stage2_path.read_text(encoding="utf-8"))
        all_scores = data.get("all_scores", [])
        if not all_scores:
            return None

        has_cluster_fusion = bool(
            data.get("recall_debug", {}).get("main_method")
            or any("score_base" in item for item in all_scores)
        )
        if not has_cluster_fusion and main_cluster_scorer(self.config) is not None:
            return None

        sorted_scores = sorted(all_scores, key=lambda x: x.get("score", 0), reverse=True)
        return [
            RankedFigure(
                figure_id=item["image_hash"],
                score=float(item.get("score", 0)),
                rank=i + 1,
                method_name=self.method_name,
            )
            for i, item in enumerate(sorted_scores)
        ]

    def _embeddings_for(self, sample: Stage2Sample) -> dict:
        cache = self._get_image_cache()
        if cache is None:
            return {}
        if sample.paper_id not in self._embeddings_by_paper:
            self._embeddings_by_paper[sample.paper_id] = cache.load_or_compute(
                sample.paper_id,
                sample.figures,
            )
        return self._embeddings_by_paper[sample.paper_id]

    def rank(self, sample: Stage2Sample) -> list[RankedFigure]:
        ranked = self._load_from_json(sample.paper_id)
        if ranked is not None:
            logger.debug("Proposed：从 stage2 缓存加载 %s", sample.paper_id)
            return ranked

        if self.config.stage2_dir.joinpath(f"{sample.paper_id}.json").is_file():
            embs = self._embeddings_for(sample)
            ranked = rank_main_method(self.config, sample, embs)
            if ranked:
                return ranked

        logger.info("Proposed：stage2 缓存缺失，fallback 运行 run_stage2(%s)", sample.paper_id)
        self._get_runner().run_stage2(sample.paper_id)
        ranked = self._load_from_json(sample.paper_id)
        if ranked is not None:
            return ranked
        embs = self._embeddings_for(sample)
        ranked = rank_main_method(self.config, sample, embs)
        if not ranked:
            logger.error("Proposed：run_stage2 后仍无法读取 %s 的排序结果", sample.paper_id)
        return ranked
