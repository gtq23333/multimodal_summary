from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import FOCUS_PAIRS, PRIMARY_METHODS


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_miss_jaccard_heatmap(
    pair_df: pd.DataFrame,
    methods: list[str],
    k: int,
    out_path: Path,
) -> None:
    n = len(methods)
    mat = np.eye(n)
    idx = {m: i for i, m in enumerate(methods)}
    sub = pair_df[pair_df["k"] == k]
    for row in sub.itertuples():
        i, j = idx.get(row.method_a), idx.get(row.method_b)
        if i is None or j is None:
            continue
        mat[i, j] = row.miss_jaccard
        mat[j, i] = row.miss_jaccard

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.7)))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlOrRd")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(methods, fontsize=8)
    ax.set_title(f"Miss-set Jaccard @ K={k}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, out_path)


def plot_rescue_heatmap(
    pair_df: pd.DataFrame,
    methods: list[str],
    k: int,
    out_path: Path,
) -> None:
    n = len(methods)
    mat = np.zeros((n, n))
    idx = {m: i for i, m in enumerate(methods)}
    sub = pair_df[pair_df["k"] == k]
    for row in sub.itertuples():
        i, j = idx.get(row.method_a), idx.get(row.method_b)
        if i is None or j is None:
            continue
        mat[i, j] = row.rescue_a_to_b
        mat[j, i] = row.rescue_b_to_a

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.7)))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(methods, fontsize=8)
    ax.set_title(f"Unique rescue count @ K={k} (row misses, col rescues)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, out_path)


def plot_union_ir_lines(union_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, sub in union_df.groupby("union_name"):
        sub = sub.sort_values("k")
        ax.plot(sub["k"], sub["ir@k"], marker="o", label=name)
    ax.set_xlabel("K")
    ax.set_ylabel("IR@K")
    ax.set_title("Union Oracle vs Single Methods")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out_path)


def plot_upset_miss_counts(
    gt_df: pd.DataFrame,
    methods: list[str],
    k: int,
    out_path: Path,
    *,
    top_n: int = 12,
) -> None:
    """Simple intersection-size bar chart for miss patterns."""
    patterns: dict[str, int] = {}
    miss_cols = [f"{m}|hit@{k}" for m in methods]
    for row in gt_df.itertuples(index=False):
        row_dict = row._asdict()
        missed = [m for m in methods if not bool(row_dict.get(f"{m}|hit@{k}", True))]
        if not missed:
            continue
        key = "&".join(missed) if missed else "none"
        patterns[key] = patterns.get(key, 0) + 1

    items = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not items:
        return
    labels, counts = zip(*items)
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.35)))
    y = np.arange(len(labels))
    ax.barh(y, counts, color="#4c78a8")
    ax.set_yticks(y)
    ax.set_yticklabels([lbl[:60] for lbl in labels], fontsize=7)
    ax.set_xlabel("GT figure count")
    ax.set_title(f"Top miss patterns @ K={k}")
    ax.invert_yaxis()
    _save(fig, out_path)


def plot_incremental_first_hit(
    first_hit_df: pd.DataFrame,
    order: list[str],
    k: int,
    out_path: Path,
) -> None:
    counts = (
        first_hit_df[first_hit_df["k"] == k]["first_hit_method"]
        .value_counts()
        .reindex(order, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(counts)), counts.values, color="#72b7b2")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("GT figures")
    ax.set_title(f"First hit step @ K={k} (incremental ablation)")
    _save(fig, out_path)


def plot_dropone_contribution(df: pd.DataFrame, k: int, out_path: Path) -> None:
    sub = df[df["k"] == k].sort_values("net", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(sub))
    ax.barh(x, sub["rescue"], color="#59a14f", label="Rescue")
    ax.barh(x, -sub["harm"], color="#e15759", label="Harm")
    ax.set_yticks(x)
    ax.set_yticklabels(sub["module"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Drop-one module contribution @ K={k}")
    ax.legend()
    _save(fig, out_path)


def plot_pool_jaccard_heatmap(
    pairwise_df: pd.DataFrame,
    methods: list[str],
    k: int,
    out_path: Path,
) -> None:
    """Pairwise candidate-pool Jaccard @ K (from union_pool_pairwise_k6)."""
    if pairwise_df.empty:
        return
    n = len(methods)
    mat = np.eye(n)
    idx = {m: i for i, m in enumerate(methods)}
    for row in pairwise_df.itertuples():
        i, j = idx.get(row.method_a), idx.get(row.method_b)
        if i is None or j is None:
            continue
        mat[i, j] = row.pool_jaccard_mean
        mat[j, i] = row.pool_jaccard_mean

    fig, ax = plt.subplots(figsize=(max(9, n * 0.85), max(7, n * 0.75)))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(methods, fontsize=7)
    ax.set_title(f"Candidate pool Jaccard @ K={k}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, out_path)


def plot_pool_union_budget_heatmap(
    pairwise_df: pd.DataFrame,
    methods: list[str],
    k: int,
    out_path: Path,
) -> None:
    """Median actual union budget for pairwise method groups."""
    if pairwise_df.empty:
        return
    n = len(methods)
    mat = np.full((n, n), np.nan)
    idx = {m: i for i, m in enumerate(methods)}
    for row in pairwise_df.itertuples():
        i, j = idx.get(row.method_a), idx.get(row.method_b)
        if i is None or j is None:
            continue
        mat[i, j] = row.actual_budget_median
        mat[j, i] = row.actual_budget_median
    for i in range(n):
        mat[i, i] = k

    fig, ax = plt.subplots(figsize=(max(9, n * 0.85), max(7, n * 0.75)))
    vmax = np.nanmax(mat) if np.any(~np.isnan(mat)) else k * 2
    im = ax.imshow(mat, vmin=k, vmax=vmax, cmap="OrRd")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(methods, fontsize=7)
    ax.set_title(f"Median union pool size (dynamic budget) @ K={k}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, out_path)


def plot_pool_size_hist(
    per_group_df: pd.DataFrame,
    nominal_budget: int,
    out_path: Path,
    *,
    title: str,
) -> None:
    if per_group_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(per_group_df["actual_budget"], bins=range(0, nominal_budget + 2), color="#4c78a8", alpha=0.85, edgecolor="white")
    ax.axvline(nominal_budget, color="#e15759", linestyle="--", linewidth=2, label=f"Nominal={nominal_budget}")
    ax.axvline(per_group_df["actual_budget"].median(), color="#59a14f", linestyle="-", linewidth=2, label=f"Median={per_group_df['actual_budget'].median():.0f}")
    ax.set_xlabel("Actual union pool size")
    ax.set_ylabel("Paper count")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_path)


def plot_pool_size_cdf(per_group_df: pd.DataFrame, out_path: Path, *, title: str) -> None:
    if per_group_df.empty:
        return
    values = np.sort(per_group_df["actual_budget"].to_numpy())
    y = np.arange(1, len(values) + 1) / len(values)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(values, y, marker=".", linewidth=2, color="#4c78a8")
    ax.set_xlabel("Actual union pool size")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    _save(fig, out_path)


def plot_overlap_vs_gain_scatter(quadrant_df: pd.DataFrame, out_path: Path) -> None:
    if quadrant_df.empty:
        return
    colors = {
        "Q1_redundant": "#bab0ac",
        "Q2_consensus_hit": "#59a14f",
        "Q3_complementary": "#4e79a7",
        "Q4_dispersed_failure": "#e15759",
    }
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for group_name, sub in quadrant_df.groupby("group_name"):
        for quadrant, pts in sub.groupby("quadrant"):
            ax.scatter(
                pts["pairwise_pool_jaccard_mean"],
                pts["union_gain"],
                label=f"{group_name[:12]}|{quadrant}",
                alpha=0.75,
                s=50,
                c=colors.get(quadrant, "#888888"),
                edgecolors="white",
                linewidths=0.5,
            )
    if not quadrant_df.empty:
        ax.axvline(quadrant_df["overlap_median_ref"].iloc[0], color="gray", linestyle=":", alpha=0.6)
        ax.axhline(quadrant_df["gain_median_ref"].iloc[0], color="gray", linestyle=":", alpha=0.6)
    ax.set_xlabel("Pairwise pool Jaccard (mean)")
    ax.set_ylabel("Union gain (pool_gt_recall - best_single)")
    ax.set_title("Overlap vs union gain @ K=6")
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
    _save(fig, out_path)


def plot_actual_budget_vs_k(dist_csv_path: Path, out_path: Path) -> None:
    dist_df = pd.read_csv(dist_csv_path)
    focus = dist_df[dist_df["group_name"].isin(["Proposed+Qwen+Layout", "PRIMARY_ALL_9"])]
    if focus.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, sub in focus.groupby("group_name"):
        sub = sub.sort_values("k")
        ax.plot(sub["k"], sub["actual_budget_median"], marker="o", linewidth=2, label=name)
    ax.set_xlabel("K")
    ax.set_ylabel("Median actual union pool size")
    ax.set_title("Dynamic budget vs K")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, out_path)
