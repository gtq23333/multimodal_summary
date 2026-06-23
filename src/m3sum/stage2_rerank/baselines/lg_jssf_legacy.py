from __future__ import annotations

import json
import logging
from pathlib import Path

from m3sum.config import PipelineConfig
from m3sum.stage2_rerank.baselines.base import RankedFigure, Stage2Sample

logger = logging.getLogger(__name__)


class LegacyLGJSSFRanker:
    """改造前 LG-JSSF：读取 stage2_legacy 缓存 JSON。"""

    method_name = "LG-JSSF-Legacy"

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stage2_dir = config.output_dir / "stage2_legacy"

    def _load_from_json(self, paper_id: str) -> list[RankedFigure] | None:
        stage2_path = self.stage2_dir / f"{paper_id}.json"
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
            return ranked
        logger.warning("Legacy stage2 缓存缺失: %s", sample.paper_id)
        return []
