from __future__ import annotations

from itertools import combinations
from typing import Iterable

import pandas as pd

from .failure_sets import hit_set, miss_set


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def cohens_kappa(labels_a: list[bool], labels_b: list[bool]) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0
    n = len(labels_a)
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = agree / n
    p_a1 = sum(labels_a) / n
    p_b1 = sum(labels_b) / n
    p_e = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    if abs(1 - p_e) < 1e-12:
        return 0.0
    return (p_o - p_e) / (1 - p_e)


def unique_rescue(miss_a: set, hit_b: set) -> set:
    return miss_a & hit_b


def method_pair_stats(
    gt_df: pd.DataFrame,
    methods: list[str],
    k: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for a, b in combinations(methods, 2):
        miss_a = miss_set(gt_df, a, k)
        miss_b = miss_set(gt_df, b, k)
        hit_a = hit_set(gt_df, a, k)
        hit_b = hit_set(gt_df, b, k)

        rescue_a_to_b = unique_rescue(miss_a, hit_b)
        rescue_b_to_a = unique_rescue(miss_b, hit_a)
        miss_union = miss_a | miss_b

        col_a = f"{a}|hit@{k}"
        col_b = f"{b}|hit@{k}"
        kappa = cohens_kappa(
            gt_df[col_a].astype(bool).tolist(),
            gt_df[col_b].astype(bool).tolist(),
        )

        rows.append(
            {
                "k": k,
                "method_a": a,
                "method_b": b,
                "miss_jaccard": round(jaccard(miss_a, miss_b), 4),
                "hit_jaccard": round(jaccard(hit_a, hit_b), 4),
                "cohen_kappa": round(kappa, 4),
                "miss_a": len(miss_a),
                "miss_b": len(miss_b),
                "miss_intersection": len(miss_a & miss_b),
                "miss_union": len(miss_union),
                "rescue_a_to_b": len(rescue_a_to_b),
                "rescue_b_to_a": len(rescue_b_to_a),
                "complementarity_a_to_b": round(
                    len(rescue_a_to_b) / max(len(miss_union), 1), 4
                ),
                "complementarity_b_to_a": round(
                    len(rescue_b_to_a) / max(len(miss_union), 1), 4
                ),
            }
        )
    return pd.DataFrame(rows)


def union_oracle_ir(
    rankings_by_method: dict[str, dict[str, dict[str, Any]]],
    samples,
    method_groups: dict[str, list[str]],
    ks: Iterable[int],
) -> pd.DataFrame:
    rows: list[dict] = []
    for union_name, methods in method_groups.items():
        for k in ks:
            total_gt = 0
            hit_gt = 0
            papers_full = 0
            n_papers = 0
            for sample in samples:
                gold = sample.ground_truth_ids
                if not gold:
                    continue
                n_papers += 1
                total_gt += len(gold)
                paper_hits = 0
                for figure_id in gold:
                    found = False
                    for method in methods:
                        rec = rankings_by_method.get(method, {}).get(sample.paper_id)
                        if not rec:
                            continue
                        ranked_ids = rec["ranked_ids"]
                        if figure_id in ranked_ids[:k]:
                            found = True
                            break
                    if found:
                        hit_gt += 1
                        paper_hits += 1
                if paper_hits == len(gold):
                    papers_full += 1
            rows.append(
                {
                    "union_name": union_name,
                    "methods": "+".join(methods),
                    "k": k,
                    "ir@k": round(hit_gt / max(total_gt, 1), 4),
                    "papers_full_recall": papers_full,
                    "n_papers": n_papers,
                    "gt_hits": hit_gt,
                    "gt_total": total_gt,
                }
            )
    return pd.DataFrame(rows)


def single_method_ir(
    rankings_by_method: dict[str, dict[str, dict[str, Any]]],
    samples,
    methods: list[str],
    ks: Iterable[int],
) -> pd.DataFrame:
    groups = {m: [m] for m in methods}
    df = union_oracle_ir(rankings_by_method, samples, groups, ks)
    df = df.rename(columns={"union_name": "method_name"})
    df["method_name"] = df["methods"]
    return df[["method_name", "k", "ir@k", "gt_hits", "gt_total", "n_papers"]]
