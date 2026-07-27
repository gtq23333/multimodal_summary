from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import paths as _paths  # noqa: F401

from m3sum.config import PipelineConfig

from .io import load_stage2_item_map


def _percentile_rank(value: float, values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    return float((arr < value).sum() / len(arr))


def _type_category(caption: str, p_type: float) -> str:
    lower = caption.lower()
    if any(k in caption for k in ("流程", "框架", "架构", "算法")) or "flow" in lower:
        return "framework"
    if any(k in caption for k in ("结果", "对比", "曲线", "统计", "分布")):
        return "result_plot"
    if any(k in caption for k in ("示意", "结构", "模型")):
        return "schematic"
    if p_type > 1.05:
        return "type_boosted"
    if p_type < 0.98:
        return "type_penalized"
    return "other"


def enrich_gt_profiles(
    gt_df: pd.DataFrame,
    config: PipelineConfig,
    *,
    reference_method: str = "Proposed",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rank_col = f"{reference_method}|rank"
    for row_dict in gt_df.to_dict(orient="records"):
        paper_id = row_dict["paper_id"]
        figure_id = row_dict["figure_id"]
        item_map = load_stage2_item_map(config, paper_id)
        item = item_map.get(figure_id, {})
        all_items = list(item_map.values())

        s_direct_vals = [float(x.get("s_direct", 0) or 0) for x in all_items]
        s_link_vals = [float(x.get("s_link", x.get("s_co", 0)) or 0) for x in all_items]
        layout_vals = [float(x.get("p_layout", 0) or 0) for x in all_items]

        s_direct = float(item.get("s_direct", 0) or 0)
        s_link = float(item.get("s_link", item.get("s_co", 0)) or 0)
        p_layout = float(item.get("p_layout", 0) or 0)
        p_type = float(item.get("p_type", 1) or 1)
        caption = str(item.get("caption", row_dict.get("caption", "")))

        evidence = item.get("evidence_blocks", []) or []
        explicit_count = sum(1 for ev in evidence if ev.get("source") == "explicit_ref")

        cluster = (item.get("debug") or {}).get("cluster", {})
        cluster_label = cluster.get("cluster_top1_label")

        ref_rank = row_dict.get(rank_col)

        rows.append(
            {
                "paper_id": paper_id,
                "figure_id": figure_id,
                "caption": caption[:120],
                "figure_index": item.get("figure_index"),
                "body_order_rank": item.get("layout_index"),
                "explicit_ref_count": explicit_count,
                "p_type": round(p_type, 4),
                "type_category": _type_category(caption, p_type),
                "s_direct": round(s_direct, 4),
                "s_link": round(s_link, 4),
                "p_layout": round(p_layout, 4),
                "s_direct_pct": round(_percentile_rank(s_direct, s_direct_vals), 4),
                "s_link_pct": round(_percentile_rank(s_link, s_link_vals), 4),
                "p_layout_pct": round(_percentile_rank(p_layout, layout_vals), 4),
                "cluster_top1_label": cluster_label,
                f"{reference_method}_rank": ref_rank,
            }
        )
    return pd.DataFrame(rows)


def shared_hard_cases(
    gt_df: pd.DataFrame,
    methods: list[str],
    k: int,
    *,
    min_shared_miss: int = 3,
) -> pd.DataFrame:
    work = gt_df.copy()
    miss_cols = [f"{m}|hit@{k}" for m in methods if f"{m}|hit@{k}" in work.columns]
    work["miss_count"] = sum((~work[c].astype(bool)).astype(int) for c in miss_cols)
    work["hit_count"] = len(miss_cols) - work["miss_count"]
    shared = work[work["miss_count"] >= min_shared_miss].copy()
    shared = shared.sort_values(["miss_count", "paper_id"], ascending=[False, True])
    return shared


def bucket_miss_rates(
    gt_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    methods: list[str],
    k: int,
    bucket_col: str,
) -> pd.DataFrame:
    merged = gt_df.merge(
        profile_df[["paper_id", "figure_id", bucket_col]],
        on=["paper_id", "figure_id"],
        how="left",
    )
    rows: list[dict] = []
    for method in methods:
        hit_col = f"{method}|hit@{k}"
        if hit_col not in merged.columns:
            continue
        for bucket, sub in merged.groupby(bucket_col, dropna=False):
            n = len(sub)
            miss = int((~sub[hit_col].astype(bool)).sum())
            rows.append(
                {
                    "method": method,
                    "bucket": bucket,
                    "k": k,
                    "n_gt": n,
                    "miss": miss,
                    "miss_rate": round(miss / max(n, 1), 4),
                }
            )
    return pd.DataFrame(rows)
