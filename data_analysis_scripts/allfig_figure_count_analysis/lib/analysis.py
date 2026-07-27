from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .paths import (
    ALLFIG_GROUPS,
    FIGURE_COUNT_BIN_LABELS,
    FIGURE_COUNT_BINS,
    PRERECALL_GROUPS,
    PRERECALL_LABELS,
    SCORE_COLS,
)


def load_eval_results(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"paper_id", "group_code", "strategy", "comprehensive_score", "pool_size"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in eval CSV: {missing}")
    return df


def paper_figure_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Total candidate figures per paper (All-Figures pool_size)."""
    sub = df[df["method_name"] == "All-Figures"][["paper_id", "pool_size"]].drop_duplicates()
    sub = sub.rename(columns={"pool_size": "total_figure_count"})
    sub["total_figure_count"] = sub["total_figure_count"].astype(int)
    return sub


def _assign_figure_bin(count: int) -> str:
    for idx in range(len(FIGURE_COUNT_BIN_LABELS)):
        lo = FIGURE_COUNT_BINS[idx]
        hi = FIGURE_COUNT_BINS[idx + 1]
        if lo < count <= hi:
            return FIGURE_COUNT_BIN_LABELS[idx]
    return FIGURE_COUNT_BIN_LABELS[-1]


def build_paired_paper_table(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """One row per paper: AllFig vs each pre-recall method + best pre-recall."""
    allfig_group = ALLFIG_GROUPS[strategy]
    prerecall_groups = PRERECALL_GROUPS[strategy]
    fig_counts = paper_figure_counts(df)

    pivot = (
        df[df["group_code"].isin([allfig_group, *prerecall_groups])]
        .pivot_table(
            index="paper_id",
            columns="group_code",
            values="comprehensive_score",
            aggfunc="first",
        )
        .reset_index()
    )
    if allfig_group not in pivot.columns:
        raise ValueError(f"Missing group {allfig_group} in eval results")

    rows: list[dict[str, Any]] = []
    for _, row in pivot.iterrows():
        paper_id = row["paper_id"]
        allfig_score = float(row[allfig_group])
        prerecall_scores: dict[str, float] = {}
        for group in prerecall_groups:
            if group in pivot.columns and pd.notna(row.get(group)):
                prerecall_scores[group] = float(row[group])

        if not prerecall_scores:
            continue

        best_group = max(prerecall_scores, key=prerecall_scores.get)
        best_score = prerecall_scores[best_group]
        fc_row = fig_counts[fig_counts["paper_id"] == paper_id]
        total_figures = int(fc_row["total_figure_count"].iloc[0]) if len(fc_row) else np.nan

        entry: dict[str, Any] = {
            "paper_id": paper_id,
            "strategy": strategy,
            "allfig_group": allfig_group,
            "total_figure_count": total_figures,
            "figure_bin": _assign_figure_bin(int(total_figures)) if pd.notna(total_figures) else "",
            "allfig_score": allfig_score,
            "best_prerecall_score": best_score,
            "best_prerecall_group": best_group,
            "best_prerecall_method": PRERECALL_LABELS.get(best_group, best_group),
            "delta_allfig_minus_best": allfig_score - best_score,
            "allfig_wins": allfig_score > best_score,
            "allfig_ties": allfig_score == best_score,
        }
        for group, score in prerecall_scores.items():
            short = PRERECALL_LABELS.get(group, group)
            entry[f"{short}_score"] = score
            entry[f"delta_allfig_minus_{short}"] = allfig_score - score
            entry[f"allfig_beats_{short}"] = allfig_score > score
        rows.append(entry)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("total_figure_count").reset_index(drop=True)


def build_long_scores(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Long format for plotting: paper × method family scores."""
    paired = build_paired_paper_table(df, strategy)
    if paired.empty:
        return paired

    method_cols = {
        "AllFig": "allfig_score",
        "BestPreRecall": "best_prerecall_score",
    }
    for short in ("Proposed", "Layout", "QwenVL"):
        col = f"{short}_score"
        if col in paired.columns:
            method_cols[short] = col

    rows: list[dict[str, Any]] = []
    for _, row in paired.iterrows():
        for method, col in method_cols.items():
            if col not in row or pd.isna(row[col]):
                continue
            rows.append(
                {
                    "paper_id": row["paper_id"],
                    "strategy": row["strategy"],
                    "total_figure_count": row["total_figure_count"],
                    "figure_bin": row["figure_bin"],
                    "method_family": method,
                    "comprehensive_score": row[col],
                }
            )
    return pd.DataFrame(rows)


def bin_aggregate(paired: pd.DataFrame) -> pd.DataFrame:
    """Mean/median scores and win rates by figure-count bin."""
    if paired.empty:
        return paired

    rows: list[dict[str, Any]] = []
    for figure_bin, sub in paired.groupby("figure_bin", sort=False):
        row: dict[str, Any] = {
            "figure_bin": figure_bin,
            "paper_count": len(sub),
            "mean_total_figures": round(sub["total_figure_count"].mean(), 2),
            "allfig_mean": round(sub["allfig_score"].mean(), 4),
            "allfig_median": round(sub["allfig_score"].median(), 4),
            "best_prerecall_mean": round(sub["best_prerecall_score"].mean(), 4),
            "best_prerecall_median": round(sub["best_prerecall_score"].median(), 4),
            "delta_mean": round(sub["delta_allfig_minus_best"].mean(), 4),
            "delta_median": round(sub["delta_allfig_minus_best"].median(), 4),
            "allfig_win_rate": round(sub["allfig_wins"].mean(), 4),
            "allfig_win_count": int(sub["allfig_wins"].sum()),
        }
        for short in ("Proposed", "Layout", "QwenVL"):
            col = f"{short}_score"
            beat_col = f"allfig_beats_{short}"
            if col in sub.columns:
                row[f"{short.lower()}_mean"] = round(sub[col].mean(), 4)
            if beat_col in sub.columns:
                row[f"allfig_beats_{short.lower()}_rate"] = round(sub[beat_col].mean(), 4)
        rows.append(row)

    order = {label: idx for idx, label in enumerate(FIGURE_COUNT_BIN_LABELS)}
    result = pd.DataFrame(rows)
    result["_ord"] = result["figure_bin"].map(order)
    return result.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)


def correlation_summary(paired: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between total figure count and key quantities."""
    if paired.empty or len(paired) < 3:
        return pd.DataFrame()

    targets = {
        "allfig_score": "AllFig comprehensive",
        "best_prerecall_score": "Best pre-recall comprehensive",
        "delta_allfig_minus_best": "AllFig − Best pre-recall",
    }
    for short in ("Proposed", "Layout", "QwenVL"):
        col = f"delta_allfig_minus_{short}"
        if col in paired.columns:
            targets[col] = f"AllFig − {short}"

    rows: list[dict[str, Any]] = []
    x = paired["total_figure_count"].astype(float)
    for col, label in targets.items():
        if col not in paired.columns:
            continue
        y = paired[col].astype(float)
        rho, pval = stats.spearmanr(x, y)
        rows.append(
            {
                "target": label,
                "column": col,
                "spearman_rho": round(float(rho), 4),
                "p_value": round(float(pval), 6),
                "n": len(paired),
            }
        )
    return pd.DataFrame(rows)


def threshold_summary(paired: pd.DataFrame, threshold: int = 30) -> pd.DataFrame:
    """Compare AllFig advantage below vs at/above a figure-count threshold."""
    if paired.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for label, sub in [
        (f"≤{threshold}", paired[paired["total_figure_count"] <= threshold]),
        (f">{threshold}", paired[paired["total_figure_count"] > threshold]),
    ]:
        if sub.empty:
            continue
        rows.append(
            {
                "segment": label,
                "paper_count": len(sub),
                "figure_count_range": f"{int(sub['total_figure_count'].min())}–{int(sub['total_figure_count'].max())}",
                "allfig_mean": round(sub["allfig_score"].mean(), 4),
                "best_prerecall_mean": round(sub["best_prerecall_score"].mean(), 4),
                "delta_mean": round(sub["delta_allfig_minus_best"].mean(), 4),
                "allfig_win_rate": round(sub["allfig_wins"].mean(), 4),
            }
        )
    return pd.DataFrame(rows)


def metric_breakdown_by_bin(df: pd.DataFrame, strategy: str, metric: str) -> pd.DataFrame:
    """Image/text metric means by bin for AllFig vs best pre-recall."""
    allfig_group = ALLFIG_GROUPS[strategy]
    prerecall_groups = PRERECALL_GROUPS[strategy]
    fig_counts = paper_figure_counts(df)

    work = df[df["group_code"].isin([allfig_group, *prerecall_groups])].copy()
    work = work.merge(fig_counts, on="paper_id", how="left")
    work["figure_bin"] = work["total_figure_count"].map(lambda c: _assign_figure_bin(int(c)))

    rows: list[dict[str, Any]] = []
    for figure_bin, sub in work.groupby("figure_bin", sort=False):
        allfig_vals = sub[sub["group_code"] == allfig_group][metric].dropna()
        prerecall_vals = []
        for paper_id, grp in sub.groupby("paper_id"):
            pre = grp[grp["group_code"].isin(prerecall_groups)]
            if pre.empty or metric not in pre.columns:
                continue
            prerecall_vals.append(pre[metric].max())
        if not len(allfig_vals):
            continue
        rows.append(
            {
                "figure_bin": figure_bin,
                "metric": metric,
                "allfig_mean": round(allfig_vals.mean(), 4),
                "best_prerecall_mean": round(float(np.mean(prerecall_vals)), 4) if prerecall_vals else np.nan,
            }
        )
    order = {label: idx for idx, label in enumerate(FIGURE_COUNT_BIN_LABELS)}
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["_ord"] = result["figure_bin"].map(order)
    return result.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
