"""
改造前 vs 改造后 LG-JSSF 同集对比评估。

- 改造前：stage2_legacy/ + LegacyLGJSSFRanker + legacy cluster 消融
- 改造后：stage2/ + Proposed + 现有 cluster 消融
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from m3sum.config import PipelineConfig
from m3sum.eval.stage2_rerank_metrics import (
    average_precision,
    compute_mrr,
    image_precision_at_k,
    image_recall_at_k,
    jaccard_at_k,
    maxsim_at_k,
    r_precision,
)
from m3sum.pipeline.runner import PipelineRunner
from m3sum.stage2_rerank.ablation import Stage2FeatureRanker
from m3sum.stage2_rerank.baselines.base import Stage2Sample
from m3sum.stage2_rerank.baselines.lg_jssf_legacy import LegacyLGJSSFRanker
from m3sum.stage2_rerank.baselines.proposed import ProposedRanker
from m3sum.stage2_rerank.clip_utils import ClipImageEmbeddingCache, load_clip_model
from m3sum.stage2_rerank.cluster_prior import ClusterPriorScorer
from m3sum.stage2_rerank.fusion import FusionConfig, compute_fused_score
from m3sum.stage2_rerank.sample_loader import load_all_stage2_samples

logger = logging.getLogger(__name__)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    df = pd.DataFrame(rows)
    return {
        "r_precision": float(df["r_precision"].mean()),
        "ip@3": float(df["ip@3"].mean()),
        "ir@3": float(df["ir@3"].mean()),
        "map": float(df["map"].mean()),
        "mrr": float(df["mrr"].mean()),
        "jaccard@3": float(df["jaccard@3"].mean()),
        "maxsim@3": float(df["maxsim@3"].mean()),
    }


def _metric_row(
    sample: Stage2Sample,
    method_name: str,
    ranked_ids: list[str],
    maxsim_cache: ClipImageEmbeddingCache | None,
    k_jaccard: int,
    k_maxsim: int,
) -> dict[str, Any]:
    gold = sample.ground_truth_ids
    if maxsim_cache is None:
        ms = float("nan")
    else:
        ms = maxsim_at_k(
            ranked_ids,
            gold,
            sample.figures,
            maxsim_cache,
            sample.paper_id,
            k=k_maxsim,
        )
    return {
        "paper_id": sample.paper_id,
        "method_name": method_name,
        "r_precision": round(r_precision(ranked_ids, gold), 6),
        "ip@3": round(image_precision_at_k(ranked_ids, gold, k=k_jaccard), 6),
        "ir@3": round(image_recall_at_k(ranked_ids, gold, k=k_jaccard), 6),
        "jaccard@3": round(jaccard_at_k(ranked_ids, gold, k=k_jaccard), 6),
        "maxsim@3": round(ms, 6) if ms == ms else None,
        "map": round(average_precision(ranked_ids, gold), 6),
        "mrr": round(compute_mrr(ranked_ids, gold), 6),
    }


class LegacyStage2FeatureRanker(Stage2FeatureRanker):
    """从 stage2_legacy/ 读取特征并应用 legacy 融合公式。"""

    def _load_stage2_items(self, paper_id: str) -> list[dict[str, Any]]:
        path = self.config.output_dir / "stage2_legacy" / f"{paper_id}.json"
        if not path.is_file():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("all_scores", []))

    def rank(self, sample: Stage2Sample) -> list:
        items = self._load_stage2_items(sample.paper_id)
        legacy_rr = self.config.raw.get("rerank_legacy", {})
        alpha = float(legacy_rr.get("alpha", 0.5))
        scored: list[tuple[str, float]] = []
        for item in items:
            figure_id = item["image_hash"]
            cluster_prior = 0.0
            if self.fusion_config.use_cluster and self.cluster_scorer:
                emb = self.image_embeddings_by_paper.get(sample.paper_id, {}).get(figure_id)
                cluster_prior, _ = self.cluster_scorer.score(emb)
            score = compute_fused_score(
                item,
                self.fusion_config,
                alpha=alpha,
                cluster_prior=cluster_prior,
                rerank_raw=legacy_rr,
            )
            scored.append((figure_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        from m3sum.stage2_rerank.baselines.base import RankedFigure

        return [
            RankedFigure(
                figure_id=figure_id,
                score=score,
                rank=i + 1,
                method_name=self.fusion_config.method_name,
            )
            for i, (figure_id, score) in enumerate(scored)
        ]


def run_legacy_stage2_pipeline(config: PipelineConfig, force: bool = False) -> None:
    """为全部样本生成 stage2_legacy JSON。"""
    runner = PipelineRunner(config, dry_run=config.dry_run, from_cache=True)
    samples = load_all_stage2_samples(config)
    for sample in samples:
        logger.info("Legacy stage2: %s", sample.paper_id)
        runner.run_stage2_legacy(sample.paper_id, force=force)


def run_legacy_compare_eval(
    config: PipelineConfig,
    skip_clip: bool = False,
    force_legacy_rerun: bool = False,
) -> pd.DataFrame:
    """运行改造前/后同集对比，输出 legacy_compare_results.csv。"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    samples = load_all_stage2_samples(config)
    if not samples:
        logger.warning("无有效样本")
        return pd.DataFrame()

    run_legacy_stage2_pipeline(config, force=force_legacy_rerun)

    clip_cache: ClipImageEmbeddingCache | None = None
    if not skip_clip and not config.dry_run:
        clip_encoder = load_clip_model(config.cluster_prior_clip_model)
        clip_cache = ClipImageEmbeddingCache(
            config.stage2_eval_clip_cache_dir,
            clip_encoder=clip_encoder,
            dry_run=False,
        )

    jaccard_k = config.stage2_eval_jaccard_k
    maxsim_k = config.stage2_eval_maxsim_k
    rows: list[dict[str, Any]] = []

    legacy_ranker = LegacyLGJSSFRanker(config)
    new_ranker = ProposedRanker(config, dry_run=config.dry_run)

    for label, ranker in [
        ("LG-JSSF-Legacy", legacy_ranker),
        ("LG-JSSF-New (Proposed)", new_ranker),
    ]:
        for sample in samples:
            ranked = ranker.rank(sample)
            ranked_ids = [r.figure_id for r in ranked]
            rows.append(
                _metric_row(
                    sample,
                    label,
                    ranked_ids,
                    clip_cache,
                    jaccard_k,
                    maxsim_k,
                )
            )

    legacy_cp = config.raw.get("legacy_cluster_prior", {})
    if not skip_clip and legacy_cp.get("enabled", True):
        clip_encoder = load_clip_model(config.cluster_prior_clip_model)
        image_cache = ClipImageEmbeddingCache(
            config.stage2_eval_clip_cache_dir,
            clip_encoder=clip_encoder,
            dry_run=False,
        )
        image_embeddings_by_paper = {
            s.paper_id: image_cache.load_or_compute(s.paper_id, s.figures) for s in samples
        }

        def legacy_scorer() -> ClusterPriorScorer:
            return ClusterPriorScorer.from_json(
                config.cluster_prior_path,
                tau=float(legacy_cp.get("tau", 0.72)),
                margin_tau=float(legacy_cp.get("margin_tau", 0.03)),
                threshold_mode=str(
                    legacy_cp.get("threshold_mode", config.cluster_prior_threshold_mode)
                ),
            )

        legacy_fc = FusionConfig(
            "Legacy+ClusterAdd",
            use_cluster=True,
            cluster_fusion_mode="additive",
            beta=float(legacy_cp.get("beta", 0.25)),
        )
        legacy_cluster_ranker = LegacyStage2FeatureRanker(
            config,
            legacy_fc,
            cluster_scorer=legacy_scorer(),
            image_embeddings_by_paper=image_embeddings_by_paper,  # type: ignore[arg-type]
        )
        for sample in samples:
            ranked = legacy_cluster_ranker.rank(sample)
            rows.append(
                _metric_row(
                    sample,
                    "Legacy+ClusterAdd",
                    [r.figure_id for r in ranked],
                    clip_cache,
                    jaccard_k,
                    maxsim_k,
                )
            )

        new_cp = config.raw.get("cluster_prior", {})
        new_fc = FusionConfig(
            "New+ClusterAdd",
            use_cluster=True,
            cluster_fusion_mode="additive",
            beta=float(new_cp.get("best_beta", 0.12)),
        )
        new_cluster_ranker = Stage2FeatureRanker(
            config,
            new_fc,
            cluster_scorer=ClusterPriorScorer.from_json(
                config.cluster_prior_path,
                tau=float(new_cp.get("best_tau", 0.78)),
                margin_tau=float(new_cp.get("margin_tau", 0.05)),
                threshold_mode=config.cluster_prior_threshold_mode,
            ),
            image_embeddings_by_paper=image_embeddings_by_paper,  # type: ignore[arg-type]
        )
        for sample in samples:
            ranked = new_cluster_ranker.rank(sample)
            rows.append(
                _metric_row(
                    sample,
                    "New+ClusterAdd",
                    [r.figure_id for r in ranked],
                    clip_cache,
                    jaccard_k,
                    maxsim_k,
                )
            )

    df = pd.DataFrame(rows)
    out_dir = config.eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    compare_path = out_dir / "legacy_compare_results.csv"
    df.to_csv(compare_path, index=False, encoding="utf-8-sig")

    summary_rows = []
    for method in df["method_name"].unique():
        sub = df[df.method_name == method]
        agg = _aggregate(sub.to_dict("records"))
        summary_rows.append({"method_name": method, **{k: round(v, 4) for k, v in agg.items()}})

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "legacy_compare_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    logger.info("Legacy 对比结果: %s", compare_path)
    logger.info("Legacy 对比汇总:\n%s", summary_df.to_string(index=False))

    return df
