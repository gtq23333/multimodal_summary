from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from fusion_method.load_rankings import load_rankings_jsonl, rankings_index
from fusion_method.ranker import MultiSourceFusionRanker, build_fusion_rankers
from fusion_method.types import DEFAULT_SOURCE_METHODS, DEFAULT_SOURCE_SUFFIX
from m3sum.config import PipelineConfig
from m3sum.eval.stage2_rerank_metrics import (
    average_precision,
    compute_mrr,
    image_precision_at_k,
    image_recall_at_k,
    image_recall_at_ks,
    jaccard_at_k,
    r_precision,
)
from m3sum.stage2_rerank.baselines.base import Stage2Sample
from m3sum.stage2_rerank.sample_loader import load_all_stage2_samples

logger = logging.getLogger(__name__)

RECALL_KS_DEFAULT = [3, 4, 5, 6, 7]

BASELINE_METHODS = [
    "Proposed",
    "Proposed-v2",
    "Qwen3-VL-Rerank-ImgCap+Link",
    "Layout-Order",
]


def _recall_k_column(k: int) -> str:
    return f"ir@{k}"


def _compute_paper_metrics(
    ranked_ids: list[str],
    gold: set[str],
    recall_ks: list[int],
    jaccard_k: int = 3,
) -> dict[str, Any]:
    ir_by_k = image_recall_at_ks(ranked_ids, gold, recall_ks)
    row: dict[str, Any] = {
        "r_precision": round(r_precision(ranked_ids, gold), 6),
        "ip@3": round(image_precision_at_k(ranked_ids, gold, k=jaccard_k), 6),
        "ir@3": round(image_recall_at_k(ranked_ids, gold, k=jaccard_k), 6),
        "jaccard@3": round(jaccard_at_k(ranked_ids, gold, k=jaccard_k), 6),
        "map": round(average_precision(ranked_ids, gold), 6),
        "mrr": round(compute_mrr(ranked_ids, gold), 6),
    }
    for k, score in ir_by_k.items():
        row[_recall_k_column(k)] = round(score, 6)
    return row


def _evaluate_ranker_on_samples(
    ranker: MultiSourceFusionRanker,
    samples: list[Stage2Sample],
    recall_ks: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if not sample.ground_truth_ids:
            continue
        ranked = ranker.rank(sample)
        ranked_ids = [r.figure_id for r in ranked]
        metrics = _compute_paper_metrics(ranked_ids, sample.ground_truth_ids, recall_ks)
        rows.append(
            {
                "paper_id": sample.paper_id,
                "method_name": ranker.method_name,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _evaluate_baselines_from_rankings(
    rankings_by_method: dict[str, dict[str, dict[str, Any]]],
    samples: list[Stage2Sample],
    methods: list[str],
    recall_ks: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method_name in methods:
        method_records = rankings_by_method.get(method_name, {})
        for sample in samples:
            if not sample.ground_truth_ids:
                continue
            rec = method_records.get(sample.paper_id)
            if rec is None:
                logger.warning("Baseline missing: %s / %s", method_name, sample.paper_id)
                continue
            ranked_ids = list(rec.get("ranked_ids", []))
            metrics = _compute_paper_metrics(ranked_ids, sample.ground_truth_ids, recall_ks)
            rows.append(
                {
                    "paper_id": sample.paper_id,
                    "method_name": method_name,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def aggregate_method_summary(df: pd.DataFrame, recall_ks: list[int]) -> pd.DataFrame:
    metric_cols = ["r_precision", "ip@3", "jaccard@3", "map", "mrr"]
    metric_cols.extend(_recall_k_column(k) for k in recall_ks)
    seen: set[str] = set()
    deduped: list[str] = []
    for col in metric_cols:
        if col not in seen and col in df.columns:
            deduped.append(col)
            seen.add(col)
    metric_cols = deduped
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby("method_name")[metric_cols].mean().reset_index()
    for col in metric_cols:
        agg[col] = agg[col].round(4)
    return agg


def union_oracle_ir(
    rankings_by_method: dict[str, dict[str, dict[str, Any]]],
    samples: list[Stage2Sample],
    source_methods: list[str],
    ks: list[int],
    *,
    union_name: str = "Union-Oracle-PQL",
) -> pd.DataFrame:
    rows: list[dict] = []
    for k in ks:
        total_gt = 0
        hit_gt = 0
        for sample in samples:
            gold = sample.ground_truth_ids
            if not gold:
                continue
            total_gt += len(gold)
            for figure_id in gold:
                found = False
                for method in source_methods:
                    rec = rankings_by_method.get(method, {}).get(sample.paper_id)
                    if not rec:
                        continue
                    if figure_id in rec.get("ranked_ids", [])[:k]:
                        found = True
                        break
                if found:
                    hit_gt += 1
        rows.append(
            {
                "method_name": union_name,
                "metric_type": "union_oracle_ir",
                "k": k,
                "ir@k": round(hit_gt / max(total_gt, 1), 4),
                "gt_hits": hit_gt,
                "gt_total": total_gt,
            }
        )
    return pd.DataFrame(rows)


def pool_union_coverage(
    rankings_by_method: dict[str, dict[str, dict[str, Any]]],
    samples: list[Stage2Sample],
    source_methods: list[str],
    pool_k: int,
    *,
    method_name: str = "Pool-Union-PQL",
) -> pd.DataFrame:
    total_gt = 0
    hit_gt = 0
    for sample in samples:
        gold = sample.ground_truth_ids
        if not gold:
            continue
        pool: set[str] = set()
        for method in source_methods:
            rec = rankings_by_method.get(method, {}).get(sample.paper_id)
            if not rec:
                continue
            pool.update(rec.get("ranked_ids", [])[:pool_k])
        total_gt += len(gold)
        hit_gt += len(gold & pool)
    return pd.DataFrame(
        [
            {
                "method_name": method_name,
                "metric_type": "pool_coverage",
                "pool_k": pool_k,
                "pool_gt_coverage": round(hit_gt / max(total_gt, 1), 4),
                "gt_hits": hit_gt,
                "gt_total": total_gt,
            }
        ]
    )


def run_fusion_eval(
    config: PipelineConfig,
    *,
    rankings_path: Path,
    output_dir: Path,
    source_methods: list[str] | None = None,
    source_suffix: str = DEFAULT_SOURCE_SUFFIX,
    recall_ks: list[int] | None = None,
    pool_k: int = 8,
    rrf_k: int = 60,
    baseline_methods: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run dual-track fusion evaluation. Returns (per_paper, fixed_summary, pool_recall)."""
    sources = source_methods or DEFAULT_SOURCE_METHODS
    recall_ks = recall_ks or RECALL_KS_DEFAULT
    baselines = baseline_methods or BASELINE_METHODS

    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_rankings_jsonl(rankings_path)
    if not records:
        raise FileNotFoundError(f"No rankings found: {rankings_path}")

    rankings_by_method = rankings_index(records)
    samples = load_all_stage2_samples(config)

    fusion_rankers = build_fusion_rankers(
        records,
        source_methods=sources,
        source_suffix=source_suffix,
        rrf_k=rrf_k,
        pool_k=pool_k,
    )

    fusion_rows: list[pd.DataFrame] = []
    for ranker in fusion_rankers.values():
        fusion_rows.append(_evaluate_ranker_on_samples(ranker, samples, recall_ks))
    fusion_df = pd.concat(fusion_rows, ignore_index=True) if fusion_rows else pd.DataFrame()

    baseline_df = _evaluate_baselines_from_rankings(
        rankings_by_method, samples, baselines, recall_ks
    )

    per_paper_df = pd.concat([baseline_df, fusion_df], ignore_index=True)
    per_paper_path = output_dir / "fusion_eval_per_paper.csv"
    per_paper_df.to_csv(per_paper_path, index=False, encoding="utf-8-sig")

    fixed_summary = aggregate_method_summary(per_paper_df, recall_ks)
    fixed_path = output_dir / "fusion_eval_fixed_budget.csv"
    fixed_summary.to_csv(fixed_path, index=False, encoding="utf-8-sig")

    pool_rows: list[pd.DataFrame] = []
    pool_rows.append(union_oracle_ir(rankings_by_method, samples, sources, recall_ks))
    pool_rows.append(pool_union_coverage(rankings_by_method, samples, sources, pool_k))

    union_rrf_name = f"Fusion-UnionRRF-{source_suffix}"
    if not fixed_summary.empty and union_rrf_name in fixed_summary["method_name"].values:
        union_rrf_summary = fixed_summary[fixed_summary["method_name"] == union_rrf_name].iloc[0]
        for k in recall_ks:
            col = _recall_k_column(k)
            if col in fixed_summary.columns:
                pool_rows.append(
                    pd.DataFrame(
                        [
                            {
                                "method_name": union_rrf_name,
                                "metric_type": "fusion_pool_ir",
                                "k": k,
                                "ir@k": float(union_rrf_summary[col]),
                                "pool_k": pool_k,
                            }
                        ]
                    )
                )

    pool_df = pd.concat(pool_rows, ignore_index=True)
    pool_path = output_dir / "fusion_eval_pool_recall.csv"
    pool_df.to_csv(pool_path, index=False, encoding="utf-8-sig")

    logger.info("Per-paper results: %s", per_paper_path)
    logger.info("Fixed budget summary: %s", fixed_path)
    logger.info("Pool recall summary: %s", pool_path)

    return per_paper_df, fixed_summary, pool_df
