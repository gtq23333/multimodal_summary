from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from . import paths as _paths  # noqa: F401

from m3sum.config import PipelineConfig
from m3sum.eval.stage2_reranking_eval import build_stage2_rankers
from m3sum.stage2_rerank.ablation import Stage2FeatureRanker
from m3sum.stage2_rerank.baselines.base import Stage2Sample
from m3sum.stage2_rerank.clip_utils import ClipImageEmbeddingCache, load_clip_model
from m3sum.stage2_rerank.cluster_prior import ClusterPriorScorer
from m3sum.stage2_rerank.fusion import FusionConfig, compute_fused_score, drop_one_configs, incremental_configs
from m3sum.stage2_rerank.main_method import main_cluster_scorer, main_fusion_config
from m3sum.stage2_rerank.sample_loader import load_all_stage2_samples

from .paths import PRIMARY_METHODS

QWEN_CACHE_SUBDIRS = {
    "Qwen3-VL-Rerank-Img": "img",
    "Qwen3-VL-Rerank-ImgCap": "img_cap",
    "Qwen3-VL-Rerank-ImgCap+Link": "img_cap_link",
}

logger = logging.getLogger(__name__)


def _best_grid_by_fusion(grid_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    if grid_df.empty:
        return best
    for fusion_mode, sub in grid_df.groupby("fusion_mode"):
        sub = sub.sort_values(
            ["map", "mrr", "r_precision"], ascending=[False, False, False]
        )
        best[fusion_mode] = sub.iloc[0].to_dict()
    return best


def _ranking_record(sample: Stage2Sample, method_name: str, ranked) -> dict[str, Any]:
    ranked_ids = [r.figure_id for r in ranked]
    score_by_id = {r.figure_id: float(r.score) for r in ranked}
    return {
        "paper_id": sample.paper_id,
        "method_name": method_name,
        "ranked_ids": ranked_ids,
        "score_by_id": score_by_id,
        "n_candidates": len(sample.figures),
    }


def _ranking_from_stage2_json(
    config: PipelineConfig,
    sample: Stage2Sample,
    method_name: str = "Proposed",
) -> dict[str, Any] | None:
    """Read legacy LG-JSSF ranking from stage2 all_scores (no cluster prior)."""
    import json

    path = config.stage2_dir / f"{sample.paper_id}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    all_scores = data.get("all_scores", [])
    if not all_scores:
        return None
    sorted_scores = sorted(all_scores, key=lambda x: float(x.get("score", 0) or 0), reverse=True)
    ranked_ids = [str(item["image_hash"]) for item in sorted_scores]
    score_by_id = {str(item["image_hash"]): float(item.get("score", 0) or 0) for item in sorted_scores}
    return {
        "paper_id": sample.paper_id,
        "method_name": method_name,
        "ranked_ids": ranked_ids,
        "score_by_id": score_by_id,
        "n_candidates": len(sample.figures),
    }


def _load_stage2_items_relaxed(config: PipelineConfig, paper_id: str) -> list[dict[str, Any]]:
    import json

    path = config.stage2_dir / f"{paper_id}.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("all_scores", []))


def _ranking_proposed_main(
    config: PipelineConfig,
    sample: Stage2Sample,
    image_embeddings: dict[str, object],
) -> dict[str, Any] | None:
    """Proposed = legacy all_scores + ClusterPrior additive (matches eval ranker)."""
    items = _load_stage2_items_relaxed(config, sample.paper_id)
    if not items:
        return None

    fusion = main_fusion_config(config)
    scorer = main_cluster_scorer(config)
    rerank_raw = config.raw.get("rerank", {})
    alpha = float(rerank_raw.get("alpha", 0.5))

    scored: list[tuple[str, float]] = []
    for item in items:
        figure_id = str(item["image_hash"])
        cluster_prior = 0.0
        if fusion.use_cluster and scorer is not None:
            emb = image_embeddings.get(figure_id)
            cluster_prior, _ = scorer.score(emb)  # type: ignore[arg-type]
        score = compute_fused_score(
            item,
            fusion,
            alpha=alpha,
            cluster_prior=cluster_prior,
            rerank_raw=rerank_raw,
        )
        scored.append((figure_id, float(score)))

    scored.sort(key=lambda x: x[1], reverse=True)
    ranked_ids = [fid for fid, _ in scored]
    score_by_id = {fid: sc for fid, sc in scored}
    return {
        "paper_id": sample.paper_id,
        "method_name": "Proposed",
        "ranked_ids": ranked_ids,
        "score_by_id": score_by_id,
        "n_candidates": len(sample.figures),
    }


def _load_vl_cache_relaxed(cache_path: Path) -> dict[str, dict[str, float]]:
    """Load VL rerank cache without query meta validation (legacy caches)."""
    import json

    if not cache_path.is_file():
        return {}
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if k != "_meta"}


def _ranking_qwen_from_cache(
    config: PipelineConfig,
    sample: Stage2Sample,
    method_name: str,
) -> dict[str, Any] | None:
    import numpy as np

    subdir = QWEN_CACHE_SUBDIRS.get(method_name)
    if not subdir:
        return None
    cache_path = config.stage2_eval_vl_rerank_cache_dir / subdir / f"{sample.paper_id}.json"
    cached = _load_vl_cache_relaxed(cache_path)
    if not cached:
        return None

    scored: list[tuple[str, float]] = []
    for fig in sample.figures:
        per_query = [
            float(scores.get(fig.image_hash, 0.0))
            for q_idx, scores in sorted(cached.items(), key=lambda x: int(x[0]))
            if isinstance(scores, dict)
        ]
        mean_score = float(np.mean(per_query)) if per_query else 0.0
        scored.append((fig.image_hash, mean_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    ranked_ids = [fid for fid, _ in scored]
    score_by_id = {fid: sc for fid, sc in scored}
    return {
        "paper_id": sample.paper_id,
        "method_name": method_name,
        "ranked_ids": ranked_ids,
        "score_by_id": score_by_id,
        "n_candidates": len(sample.figures),
    }


def build_ablation_rankers(
    config: PipelineConfig,
    *,
    fusion_mode: str = "additive",
) -> dict[str, Stage2FeatureRanker]:
    grid_path = config.eval_dir / "stage2_cluster_grid_search.csv"
    grid_df = pd.read_csv(grid_path) if grid_path.is_file() else pd.DataFrame()
    best = _best_grid_by_fusion(grid_df).get(fusion_mode, {})

    beta = float(best.get("beta", config.raw.get("cluster_prior", {}).get("main_beta", 0.25)))
    tau = float(best.get("tau", config.raw.get("cluster_prior", {}).get("main_tau", 0.72)))

    clip_encoder = None
    if not config.dry_run and config.cluster_prior_enabled:
        clip_encoder = load_clip_model(config.cluster_prior_clip_model)

    image_cache = ClipImageEmbeddingCache(
        config.stage2_eval_clip_cache_dir,
        clip_encoder=clip_encoder,
        dry_run=config.dry_run,
    )
    samples = load_all_stage2_samples(config)
    image_embeddings_by_paper = {
        sample.paper_id: image_cache.load_or_compute(sample.paper_id, sample.figures)
        for sample in samples
    }

    scorer: ClusterPriorScorer | None = None
    if config.cluster_prior_enabled:
        scorer = ClusterPriorScorer.from_json(
            config.cluster_prior_path,
            tau=tau,
            margin_tau=config.cluster_prior_margin_tau,
            threshold_mode=config.cluster_prior_threshold_mode,
        )

    rankers: dict[str, Stage2FeatureRanker] = {}
    seen: set[str] = set()

    for fc in incremental_configs(beta=beta, fusion_mode=fusion_mode):  # type: ignore[arg-type]
        if fc.method_name in seen:
            continue
        seen.add(fc.method_name)
        use_scorer = scorer if fc.use_cluster else None
        rankers[fc.method_name] = Stage2FeatureRanker(
            config,
            fc,
            cluster_scorer=use_scorer,
            image_embeddings_by_paper=image_embeddings_by_paper,  # type: ignore[arg-type]
        )

    for fc in drop_one_configs(beta=beta, fusion_mode=fusion_mode):  # type: ignore[arg-type]
        if fc.method_name in seen:
            continue
        seen.add(fc.method_name)
        use_scorer = scorer if fc.use_cluster else None
        rankers[fc.method_name] = Stage2FeatureRanker(
            config,
            fc,
            cluster_scorer=use_scorer,
            image_embeddings_by_paper=image_embeddings_by_paper,  # type: ignore[arg-type]
        )

    logger.info(
        "Ablation rankers: fusion=%s beta=%.3f tau=%.3f count=%d",
        fusion_mode,
        beta,
        tau,
        len(rankers),
    )
    return rankers


def export_all_rankings(
    config: PipelineConfig,
    *,
    methods: list[str] | None = None,
    include_ablation: bool = True,
    skip_clip: bool = False,
    force: bool = False,
    existing_methods: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Rebuild full ranked_ids for main baselines and ablation variants."""
    samples = load_all_stage2_samples(config)
    if not samples:
        logger.warning("No Stage-2 samples found")
        return []

    main_methods = methods or list(PRIMARY_METHODS)
    existing = existing_methods or set()
    to_run = [m for m in main_methods if force or m not in existing]

    records: list[dict[str, Any]] = []

    if to_run:
        qwen_methods = set(QWEN_CACHE_SUBDIRS)
        rankers, clip_cache, _ = build_stage2_rankers(
            config,
            skip_clip=skip_clip,
            active_methods=[m for m in to_run if m not in {"Proposed"} | qwen_methods],
        )
        proposed_embs: dict[str, dict] = {}
        if "Proposed" in to_run and main_cluster_scorer(config) is not None:
            clip_encoder = None if config.dry_run else load_clip_model(config.cluster_prior_clip_model)
            clip_cache_proposed = ClipImageEmbeddingCache(
                config.stage2_eval_clip_cache_dir,
                clip_encoder=clip_encoder,
                dry_run=config.dry_run,
            )
            for sample in samples:
                proposed_embs[sample.paper_id] = clip_cache_proposed.load_or_compute(
                    sample.paper_id, sample.figures
                )

        for method_name in to_run:
            if method_name == "Proposed":
                logger.info("Ranking: %s (legacy + ClusterPrior)", method_name)
                for sample in samples:
                    embs = proposed_embs.get(sample.paper_id, {})
                    rec = _ranking_proposed_main(config, sample, embs)
                    if rec:
                        records.append(rec)
                    else:
                        logger.warning("Missing stage2 JSON for Proposed: %s", sample.paper_id)
                continue
            if method_name in QWEN_CACHE_SUBDIRS:
                logger.info("Ranking: %s (from VL cache)", method_name)
                for sample in samples:
                    rec = _ranking_qwen_from_cache(config, sample, method_name)
                    if rec:
                        records.append(rec)
                    else:
                        logger.warning("Missing VL cache for %s: %s", method_name, sample.paper_id)
                continue
            ranker = rankers.get(method_name)
            if ranker is None:
                logger.warning("Ranker not available: %s", method_name)
                continue
            logger.info("Ranking: %s", method_name)
            for sample in samples:
                ranked = ranker.rank(sample)
                records.append(_ranking_record(sample, method_name, ranked))

    if include_ablation:
        ablation_rankers = build_ablation_rankers(config, fusion_mode="additive")
        ablation_methods = sorted(ablation_rankers.keys())
        ablation_to_run = [m for m in ablation_methods if force or m not in existing]
        for method_name in ablation_to_run:
            ranker = ablation_rankers[method_name]
            logger.info("Ranking ablation: %s", method_name)
            for sample in samples:
                ranked = ranker.rank(sample)
                records.append(_ranking_record(sample, method_name, ranked))

    return records
