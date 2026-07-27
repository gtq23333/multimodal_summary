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
    image_recall_at_ks,
    jaccard_at_k,
    maxsim_at_k,
    r_precision,
)
from m3sum.stage2_rerank.ablation import Stage2FeatureRanker
from m3sum.stage2_rerank.baselines.base import Stage2Sample
from m3sum.stage2_rerank.clip_utils import ClipImageEmbeddingCache, load_clip_model
from m3sum.stage2_rerank.cluster_prior import ClusterPriorScorer
from m3sum.stage2_rerank.fusion import (
    FusionConfig,
    drop_one_configs,
    incremental_configs,
)

logger = logging.getLogger(__name__)


def _metric_row(
    sample: Stage2Sample,
    method_name: str,
    ranked_ids: list[str],
    maxsim_cache: ClipImageEmbeddingCache | None,
    k_jaccard: int,
    k_maxsim: int,
    recall_ks: list[int],
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
    row: dict[str, Any] = {
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
    for k, score in image_recall_at_ks(ranked_ids, gold, recall_ks).items():
        row[f"ir@{k}"] = round(score, 6)
    return row


def _evaluate_config(
    config: PipelineConfig,
    samples: list[Stage2Sample],
    fusion_config: FusionConfig,
    cluster_scorer: ClusterPriorScorer | None,
    image_embeddings_by_paper: dict[str, dict[str, object]],
    maxsim_cache: ClipImageEmbeddingCache | None,
) -> pd.DataFrame:
    ranker = Stage2FeatureRanker(
        config,
        fusion_config,
        cluster_scorer=cluster_scorer,
        image_embeddings_by_paper=image_embeddings_by_paper,  # type: ignore[arg-type]
    )
    rows: list[dict[str, Any]] = []
    for sample in samples:
        ranked = ranker.rank(sample)
        rows.append(
            _metric_row(
                sample,
                fusion_config.method_name,
                [r.figure_id for r in ranked],
                maxsim_cache,
                config.stage2_eval_jaccard_k,
                config.stage2_eval_maxsim_k,
                config.stage2_eval_recall_ks,
            )
        )
    return pd.DataFrame(rows)


def _aggregate(df: pd.DataFrame, recall_ks: list[int] | None = None) -> dict[str, float]:
    result = {
        "r_precision": float(df["r_precision"].mean()),
        "ip@3": float(df["ip@3"].mean()),
        "ir@3": float(df["ir@3"].mean()),
        "map": float(df["map"].mean()),
        "mrr": float(df["mrr"].mean()),
        "jaccard@3": float(df["jaccard@3"].mean()),
        "maxsim@3": float(df["maxsim@3"].mean()),
    }
    for k in recall_ks or []:
        col = f"ir@{k}"
        if col in df.columns:
            result[col] = float(df[col].mean())
    return result


def _best_grid_by_fusion(grid_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for fusion_mode, sub in grid_df.groupby("fusion_mode"):
        sub = sub.sort_values(
            ["map", "mrr", "r_precision"], ascending=[False, False, False]
        )
        best[fusion_mode] = sub.iloc[0].to_dict()
    return best


def run_stage2_ablation_eval(
    config: PipelineConfig,
    samples: list[Stage2Sample],
    maxsim_cache: ClipImageEmbeddingCache | None,
    skip_clip: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """运行 ClusterPrior grid search 与 Proposed 变体消融。"""
    if skip_clip or not config.cluster_prior_enabled:
        logger.info("跳过 ClusterPrior 消融（skip_clip=%s enabled=%s）", skip_clip, config.cluster_prior_enabled)
        return pd.DataFrame(), pd.DataFrame()

    clip_encoder = load_clip_model(config.cluster_prior_clip_model)
    image_cache = ClipImageEmbeddingCache(
        config.stage2_eval_clip_cache_dir,
        clip_encoder=clip_encoder,
        dry_run=config.dry_run,
    )
    image_embeddings_by_paper = {
        sample.paper_id: image_cache.load_or_compute(sample.paper_id, sample.figures)
        for sample in samples
    }
    scorer_cache: dict[tuple[float, str], ClusterPriorScorer] = {}

    def scorer_for(tau: float) -> ClusterPriorScorer:
        key = (tau, config.cluster_prior_threshold_mode)
        if key not in scorer_cache:
            scorer_cache[key] = ClusterPriorScorer.from_json(
                config.cluster_prior_path,
                tau=tau,
                margin_tau=config.cluster_prior_margin_tau,
                threshold_mode=config.cluster_prior_threshold_mode,
            )
        return scorer_cache[key]

    grid_rows: list[dict[str, Any]] = []
    for fusion_mode in config.cluster_prior_fusion_modes:
        if fusion_mode not in {"additive", "multiplicative"}:
            continue
        method_name = (
            "LG-JSSF+ClusterAdd"
            if fusion_mode == "additive"
            else "LG-JSSF+ClusterMul"
        )
        for tau in config.cluster_prior_tau_grid:
            for beta in config.cluster_prior_beta_grid:
                fc = FusionConfig(
                    method_name=method_name,
                    use_cluster=True,
                    cluster_fusion_mode=fusion_mode,  # type: ignore[arg-type]
                    beta=beta,
                )
                df = _evaluate_config(
                    config,
                    samples,
                    fc,
                    scorer_for(tau),
                    image_embeddings_by_paper,
                    maxsim_cache,
                )
                agg = _aggregate(df, config.stage2_eval_recall_ks)
                grid_rows.append(
                    {
                        "fusion_mode": fusion_mode,
                        "tau": tau,
                        "beta": beta,
                        **{k: round(v, 6) for k, v in agg.items()},
                    }
                )

    grid_df = pd.DataFrame(grid_rows)
    grid_path = config.eval_dir / "stage2_cluster_grid_search.csv"
    grid_df.to_csv(grid_path, index=False, encoding="utf-8-sig")

    best_by_fusion = _best_grid_by_fusion(grid_df)
    ablation_frames: list[pd.DataFrame] = []
    diagnostics_path = config.eval_dir / "stage2_cluster_prior_diagnostics.jsonl"
    diagnostics_file = diagnostics_path.open("w", encoding="utf-8")

    # Baseline LG-JSSF and non-cluster incremental configs.
    seen_methods: set[str] = set()
    for fc in incremental_configs():
        if fc.method_name in seen_methods:
            continue
        seen_methods.add(fc.method_name)
        ablation_frames.append(
            _evaluate_config(
                config, samples, fc, None, image_embeddings_by_paper, maxsim_cache
            )
        )

    for fusion_mode, best in best_by_fusion.items():
        beta = float(best["beta"])
        tau = float(best["tau"])
        scorer = scorer_for(tau)
        for fc in incremental_configs(beta=beta, fusion_mode=fusion_mode):  # type: ignore[arg-type]
            if fc.method_name in seen_methods:
                continue
            seen_methods.add(fc.method_name)
            ablation_frames.append(
                _evaluate_config(
                    config, samples, fc, scorer, image_embeddings_by_paper, maxsim_cache
                )
            )
            if fc.method_name in {"LG-JSSF+ClusterAdd", "LG-JSSF+ClusterMul"}:
                ranker = Stage2FeatureRanker(
                    config,
                    fc,
                    cluster_scorer=scorer,
                    image_embeddings_by_paper=image_embeddings_by_paper,  # type: ignore[arg-type]
                )
                for sample in samples:
                    for debug in ranker.debug_for_sample(sample):
                        diagnostics_file.write(
                            json.dumps(
                                {
                                    "paper_id": sample.paper_id,
                                    "method_name": fc.method_name,
                                    "tau": tau,
                                    "beta": beta,
                                    **debug,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
        for fc in drop_one_configs(beta=beta, fusion_mode=fusion_mode):  # type: ignore[arg-type]
            if fc.method_name in seen_methods:
                continue
            seen_methods.add(fc.method_name)
            ablation_frames.append(
                _evaluate_config(
                    config, samples, fc, scorer, image_embeddings_by_paper, maxsim_cache
                )
            )

    ablation_df = pd.concat(ablation_frames, ignore_index=True) if ablation_frames else pd.DataFrame()
    ablation_path = config.eval_dir / "stage2_ablation_results.csv"
    ablation_df.to_csv(ablation_path, index=False, encoding="utf-8-sig")
    diagnostics_file.close()

    logger.info("ClusterPrior grid search: %s", grid_path)
    logger.info("ClusterPrior 消融结果: %s", ablation_path)
    logger.info("ClusterPrior 诊断日志: %s", diagnostics_path)
    return ablation_df, grid_df
