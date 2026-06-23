from __future__ import annotations

import json
import logging
from pathlib import Path

from m3sum.config import PipelineConfig
from m3sum.data.corpus_adapter import CorpusAdapter
from m3sum.data.schema import SubQuery
from m3sum.stage2_rerank.baselines.base import Stage2Sample

logger = logging.getLogger(__name__)


def load_stage2_sample(config: PipelineConfig, paper_id: str) -> Stage2Sample | None:
    """
    加载单篇 Stage-2 评估样本。
    stage1 或 ground_truth 缺失时返回 None 并记录日志。
    """
    stage1_path = config.stage1_dir / f"{paper_id}.json"
    gt_path = config.ground_truth_dir / f"{paper_id}.json"

    if not stage1_path.is_file():
        logger.warning("跳过样本 %s：缺少 stage1 输出 %s", paper_id, stage1_path)
        return None
    if not gt_path.is_file():
        logger.warning("跳过样本 %s：缺少 ground truth %s", paper_id, gt_path)
        return None

    stage1 = json.loads(stage1_path.read_text(encoding="utf-8"))
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    gold_ids = set(gt.get("retrieval_gt", {}).get("relevant_figure_hashes", []))

    if not gold_ids:
        logger.warning("跳过样本 %s：ground truth 相关图片集合为空", paper_id)
        return None

    corpus = CorpusAdapter(config)
    doc = corpus.load_document(paper_id)

    sub_queries = [
        SubQuery(
            dimension=q["dimension"],
            query=q["query"],
            keywords=q.get("keywords", []),
        )
        for q in stage1.get("sub_queries", [])
    ]
    if not sub_queries:
        logger.warning("跳过样本 %s：sub_queries 为空", paper_id)
        return None

    return Stage2Sample(
        paper_id=paper_id,
        figures=doc.figures,
        sub_queries=sub_queries,
        ground_truth_ids=gold_ids,
    )


def load_all_stage2_samples(config: PipelineConfig) -> list[Stage2Sample]:
    """加载配置中所有有效 Stage-2 样本。"""
    samples: list[Stage2Sample] = []
    for paper_id in config.resolved_sample_ids():
        sample = load_stage2_sample(config, paper_id)
        if sample is not None:
            samples.append(sample)
    return samples
