from __future__ import annotations

from typing import Any

import pandas as pd

from . import paths as _paths  # noqa: F401

from m3sum.eval.stage2_rerank_metrics import (
    average_precision,
    compute_mrr,
    image_precision_at_k,
    image_recall_at_k,
    jaccard_at_k,
    r_precision,
)
from m3sum.stage2_rerank.sample_loader import load_all_stage2_samples

from .paths import RECALL_KS


def _rank_of(figure_id: str, ranked_ids: list[str]) -> int | None:
    try:
        return ranked_ids.index(figure_id) + 1
    except ValueError:
        return None


def build_gt_outcome_matrix(
    rankings_by_method: dict[str, dict[str, dict[str, Any]]],
    samples,
    *,
    ks: list[int] | None = None,
) -> pd.DataFrame:
    """One row per GT figure with per-method hit/rank columns."""
    ks = ks or RECALL_KS
    rows: list[dict[str, Any]] = []

    for sample in samples:
        gold = sorted(sample.ground_truth_ids)
        n_candidates = len(sample.figures)
        fig_by_id = {f.image_hash: f for f in sample.figures}

        for figure_id in gold:
            fig = fig_by_id.get(figure_id)
            row: dict[str, Any] = {
                "paper_id": sample.paper_id,
                "figure_id": figure_id,
                "caption": (fig.caption if fig else "")[:120],
                "n_candidates": n_candidates,
                "n_gt": len(gold),
            }
            for method_name, paper_map in rankings_by_method.items():
                rec = paper_map.get(sample.paper_id)
                if not rec:
                    for k in ks:
                        row[f"{method_name}|hit@{k}"] = False
                    row[f"{method_name}|rank"] = None
                    row[f"{method_name}|rank_pct"] = None
                    continue
                ranked_ids = rec["ranked_ids"]
                rank = _rank_of(figure_id, ranked_ids)
                row[f"{method_name}|rank"] = rank
                row[f"{method_name}|rank_pct"] = (
                    round(rank / max(n_candidates, 1), 4) if rank else None
                )
                for k in ks:
                    row[f"{method_name}|hit@{k}"] = (
                        rank is not None and rank <= k
                    )
            rows.append(row)

    return pd.DataFrame(rows)


def miss_set(
    gt_df: pd.DataFrame,
    method_name: str,
    k: int,
) -> set[tuple[str, str]]:
    col = f"{method_name}|hit@{k}"
    if col not in gt_df.columns:
        return set()
    sub = gt_df[~gt_df[col].astype(bool)]
    return {(r.paper_id, r.figure_id) for r in sub.itertuples()}


def hit_set(
    gt_df: pd.DataFrame,
    method_name: str,
    k: int,
) -> set[tuple[str, str]]:
    col = f"{method_name}|hit@{k}"
    if col not in gt_df.columns:
        return set()
    sub = gt_df[gt_df[col].astype(bool)]
    return {(r.paper_id, r.figure_id) for r in sub.itertuples()}


def aggregate_paper_metrics(
    rankings_by_method: dict[str, dict[str, dict[str, Any]]],
    samples,
    *,
    ks: list[int] | None = None,
) -> pd.DataFrame:
    ks = ks or RECALL_KS
    rows: list[dict[str, Any]] = []
    for method_name, paper_map in rankings_by_method.items():
        for sample in samples:
            rec = paper_map.get(sample.paper_id)
            if not rec:
                continue
            ranked_ids = rec["ranked_ids"]
            gold = sample.ground_truth_ids
            row = {
                "paper_id": sample.paper_id,
                "method_name": method_name,
                "r_precision": round(r_precision(ranked_ids, gold), 6),
                "ip@3": round(image_precision_at_k(ranked_ids, gold, k=3), 6),
                "ir@3": round(image_recall_at_k(ranked_ids, gold, k=3), 6),
                "jaccard@3": round(jaccard_at_k(ranked_ids, gold, k=3), 6),
                "map": round(average_precision(ranked_ids, gold), 6),
                "mrr": round(compute_mrr(ranked_ids, gold), 6),
            }
            for k in ks:
                if k != 3:
                    row[f"ir@{k}"] = round(image_recall_at_k(ranked_ids, gold, k=k), 6)
            rows.append(row)
    return pd.DataFrame(rows)
