from __future__ import annotations

import json
import logging
from pathlib import Path

from m3sum.config import PipelineConfig
from m3sum.pipeline.runner import PipelineRunner
from m3sum.stage2_rerank.baselines.base import RankedFigure, Stage2Sample

logger = logging.getLogger(__name__)


class ProposedRanker:
    """
    Proposed method adapter。
    优先读取 stage2 缓存 JSON 的 all_scores；缺失时 fallback 到 PipelineRunner.run_stage2()。
    """

    method_name = "Proposed"

    def __init__(self, config: PipelineConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self._runner: PipelineRunner | None = None

    def _get_runner(self) -> PipelineRunner:
        if self._runner is None:
            self._runner = PipelineRunner(
                self.config,
                dry_run=self.dry_run,
                from_cache=True,
            )
        return self._runner

    def _load_from_json(self, paper_id: str) -> list[RankedFigure] | None:
        stage2_path = self.config.stage2_dir / f"{paper_id}.json"
        if not stage2_path.is_file():
            return None
        data = json.loads(stage2_path.read_text(encoding="utf-8"))
        all_scores = data.get("all_scores", [])
        if not all_scores:
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

    def rank(self, sample: Stage2Sample) -> list[RankedFigure]:
        ranked = self._load_from_json(sample.paper_id)
        if ranked is not None:
            logger.debug("Proposed：从 stage2 缓存加载 %s", sample.paper_id)
            return ranked

        logger.info("Proposed：stage2 缓存缺失，fallback 运行 run_stage2(%s)", sample.paper_id)
        self._get_runner().run_stage2(sample.paper_id)
        ranked = self._load_from_json(sample.paper_id)
        if ranked is None:
            logger.error("Proposed：run_stage2 后仍无法读取 %s 的排序结果", sample.paper_id)
            return []
        return ranked
