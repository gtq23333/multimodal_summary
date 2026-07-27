from __future__ import annotations

import json
from itertools import combinations
from typing import Any

import pandas as pd

from .overlap_metrics import jaccard


PQL_METHODS = [
    "Proposed",
    "Qwen3-VL-Rerank-ImgCap+Link",
    "Layout-Order",
]


def _short_name(methods: list[str]) -> str:
    if len(methods) == 1:
        return methods[0]
    aliases = {
        "Proposed": "P",
        "Proposed-v2": "Pv2",
        "Qwen3-VL-Rerank-ImgCap+Link": "Q",
        "Qwen3-VL-Rerank-ImgCap": "Qcap",
        "Qwen3-VL-Rerank-Img": "Qimg",
        "Layout-Order": "L",
        "Caption-BM25": "BM25",
        "Caption-Dense-v4": "Dense",
        "Zero-shot-CLIP": "CLIP",
    }
    if set(methods) == set(PQL_METHODS):
        return "Proposed+Qwen+Layout"
    if len(methods) >= 7:
        return "PRIMARY_ALL_9"
    return "+".join(aliases.get(m, m[:6]) for m in methods)


def top_k_set(
    rankings_by_method: dict[str, dict[str, dict[str, Any]]],
    paper_id: str,
    method: str,
    k: int,
) -> set[str]:
    rec = rankings_by_method.get(method, {}).get(paper_id)
    if not rec:
        return set()
    return set(rec.get("ranked_ids", [])[:k])


def pairwise_pool_jaccard_mean(sets: dict[str, set[str]]) -> float:
    methods = list(sets.keys())
    if len(methods) < 2:
        return 1.0
    scores = [jaccard(sets[a], sets[b]) for a, b in combinations(methods, 2)]
    return sum(scores) / len(scores) if scores else 0.0


def triple_intersection_size(sets: dict[str, set[str]]) -> int | None:
    if len(sets) != 3:
        return None
    values = list(sets.values())
    return len(values[0] & values[1] & values[2])


def gt_unique_contributions(
    gold: set[str],
    sets: dict[str, set[str]],
) -> dict[str, int]:
    out: dict[str, int] = {}
    methods = list(sets.keys())
    for method in methods:
        others = set()
        for other in methods:
            if other != method:
                others |= sets[other]
        unique_hits = (gold & sets[method]) - others
        out[method] = len(unique_hits)
    return out


def compute_paper_pool_row(
    rankings_by_method: dict[str, dict[str, dict[str, Any]]],
    paper_id: str,
    gold: set[str],
    methods: list[str],
    k: int,
) -> dict[str, Any]:
    sets = {m: top_k_set(rankings_by_method, paper_id, m, k) for m in methods}
    pool: set[str] = set()
    for s in sets.values():
        pool |= s

    n_gt = len(gold)
    n_methods = len(methods)
    nominal = n_methods * k
    actual = len(pool)
    compression = actual / nominal if nominal else 0.0
    redundancy_saved = nominal - actual

    single_recalls = {
        m: (len(gold & s) / n_gt if n_gt else 0.0) for m, s in sets.items()
    }
    best_single = max(single_recalls.values()) if single_recalls else 0.0
    gt_in_pool = len(gold & pool)
    pool_gt_recall = gt_in_pool / n_gt if n_gt else 0.0
    union_gain = pool_gt_recall - best_single

    shared_all = set(gold)
    for s in sets.values():
        shared_all &= s
    gt_shared_all = len(shared_all) if n_methods > 1 else gt_in_pool
    gt_miss_all = n_gt - gt_in_pool
    unique_map = gt_unique_contributions(gold, sets) if n_methods > 1 else {}

    row: dict[str, Any] = {
        "paper_id": paper_id,
        "group_name": _short_name(methods),
        "methods": "+".join(methods),
        "n_methods": n_methods,
        "k": k,
        "nominal_budget": nominal,
        "actual_budget": actual,
        "compression_ratio": round(compression, 4),
        "redundancy_saved": redundancy_saved,
        "pairwise_pool_jaccard_mean": round(pairwise_pool_jaccard_mean(sets), 4),
        "triple_intersection_size": triple_intersection_size(sets),
        "n_gt": n_gt,
        "gt_in_pool": gt_in_pool,
        "pool_gt_recall": round(pool_gt_recall, 4),
        "best_single_gt_recall": round(best_single, 4),
        "union_gain": round(union_gain, 4),
        "gt_shared_all": gt_shared_all,
        "gt_miss_all": gt_miss_all,
        "recall_per_slot": round(pool_gt_recall / actual, 6) if actual else 0.0,
        "marginal_gt_per_redundant_slot": round(
            union_gain * n_gt / max(redundancy_saved, 1), 6
        ),
        "gt_unique_contrib_json": json.dumps(unique_map, ensure_ascii=False),
    }
    for method, count in unique_map.items():
        safe = method.replace(" ", "_").replace("+", "_").replace("-", "_")
        row[f"gt_unique_{safe}"] = count
    return row


def build_method_groups(primary_methods: list[str]) -> list[tuple[str, list[str]]]:
    groups: list[tuple[str, list[str]]] = []
    seen: set[str] = set()

    def add(methods: list[str], name: str | None = None) -> None:
        key = tuple(sorted(methods))
        if key in seen or not methods:
            return
        seen.add(key)
        groups.append((name or _short_name(methods), methods))

    for m in primary_methods:
        add([m])

    for a, b in combinations(primary_methods, 2):
        add([a, b])

    pql = [m for m in PQL_METHODS if m in primary_methods]
    if len(pql) == 3:
        add(pql, "Proposed+Qwen+Layout")

    add(list(primary_methods), "PRIMARY_ALL_9")
    return groups


def build_per_paper_stats(
    rankings_by_method: dict[str, dict[str, dict[str, Any]]],
    samples,
    groups: list[tuple[str, list[str]]],
    ks: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        gold = set(sample.ground_truth_ids)
        if not gold:
            continue
        for _name, methods in groups:
            available = [m for m in methods if m in rankings_by_method]
            if not available:
                continue
            for k in ks:
                rows.append(
                    compute_paper_pool_row(
                        rankings_by_method,
                        sample.paper_id,
                        gold,
                        available,
                        k,
                    )
                )
    return pd.DataFrame(rows)


def aggregate_distribution(per_paper_df: pd.DataFrame) -> pd.DataFrame:
    if per_paper_df.empty:
        return pd.DataFrame()

    def q(series: pd.Series, p: float) -> float:
        return float(series.quantile(p))

    rows: list[dict[str, Any]] = []
    for (group_name, k), sub in per_paper_df.groupby(["group_name", "k"]):
        rows.append(
            {
                "group_name": group_name,
                "k": k,
                "n_papers": len(sub),
                "actual_budget_mean": round(sub["actual_budget"].mean(), 2),
                "actual_budget_median": round(sub["actual_budget"].median(), 2),
                "actual_budget_p25": round(q(sub["actual_budget"], 0.25), 2),
                "actual_budget_p75": round(q(sub["actual_budget"], 0.75), 2),
                "actual_budget_min": int(sub["actual_budget"].min()),
                "actual_budget_max": int(sub["actual_budget"].max()),
                "compression_ratio_mean": round(sub["compression_ratio"].mean(), 4),
                "compression_ratio_median": round(sub["compression_ratio"].median(), 4),
                "pool_gt_recall_micro": round(
                    sub["gt_in_pool"].sum() / max(sub["n_gt"].sum(), 1), 4
                ),
                "union_gain_mean": round(sub["union_gain"].mean(), 4),
                "pairwise_pool_jaccard_mean": round(
                    sub["pairwise_pool_jaccard_mean"].mean(), 4
                ),
            }
        )
    return pd.DataFrame(rows)


def build_pairwise_summary(
    per_paper_df: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    sub = per_paper_df[(per_paper_df["k"] == k) & (per_paper_df["n_methods"] == 2)].copy()
    if sub.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for group_name, grp in sub.groupby("group_name"):
        methods = grp.iloc[0]["methods"].split("+")
        rows.append(
            {
                "k": k,
                "group_name": group_name,
                "method_a": methods[0],
                "method_b": methods[1],
                "pool_jaccard_mean": round(grp["pairwise_pool_jaccard_mean"].mean(), 4),
                "actual_budget_median": round(grp["actual_budget"].median(), 2),
                "actual_budget_mean": round(grp["actual_budget"].mean(), 2),
                "union_gain_mean": round(grp["union_gain"].mean(), 4),
                "pool_gt_recall_micro": round(
                    grp["gt_in_pool"].sum() / max(grp["n_gt"].sum(), 1), 4
                ),
                "n_papers": len(grp),
            }
        )
    return pd.DataFrame(rows)


def assign_quadrant(
    overlap: float,
    gain: float,
    overlap_median: float,
    gain_median: float,
) -> str:
    high_overlap = overlap >= overlap_median
    high_gain = gain >= gain_median
    if high_overlap and not high_gain:
        return "Q1_redundant"
    if high_overlap and high_gain:
        return "Q2_consensus_hit"
    if not high_overlap and high_gain:
        return "Q3_complementary"
    return "Q4_dispersed_failure"


def build_quadrant_labels(
    per_paper_df: pd.DataFrame,
    focus_groups: list[str],
    k: int,
) -> pd.DataFrame:
    sub = per_paper_df[
        (per_paper_df["k"] == k) & (per_paper_df["group_name"].isin(focus_groups))
    ].copy()
    if sub.empty:
        return pd.DataFrame()

    overlap_med = sub["pairwise_pool_jaccard_mean"].median()
    gain_med = sub["union_gain"].median()

    rows: list[dict[str, Any]] = []
    for row in sub.itertuples():
        rows.append(
            {
                "paper_id": row.paper_id,
                "group_name": row.group_name,
                "k": k,
                "pairwise_pool_jaccard_mean": row.pairwise_pool_jaccard_mean,
                "union_gain": row.union_gain,
                "actual_budget": row.actual_budget,
                "pool_gt_recall": row.pool_gt_recall,
                "overlap_median_ref": round(overlap_med, 4),
                "gain_median_ref": round(gain_med, 4),
                "quadrant": assign_quadrant(
                    row.pairwise_pool_jaccard_mean,
                    row.union_gain,
                    overlap_med,
                    gain_med,
                ),
            }
        )
    return pd.DataFrame(rows)
