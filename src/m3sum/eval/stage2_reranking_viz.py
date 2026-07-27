from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRIC_COLS = [
    "r_precision",
    "ip@3",
    "ir@3",
    "ir@4",
    "ir@5",
    "ir@6",
    "ir@7",
    "ir@8",
    "jaccard@3",
    "maxsim@3",
    "map",
    "mrr",
]

BAR_METRIC_COLS = ["r_precision", "ip@3", "ir@3", "jaccard@3", "maxsim@3", "map", "mrr"]

RECALL_METRIC_COLS = ["ir@3", "ir@4", "ir@5", "ir@6", "ir@7", "ir@8"]

ABLATION_CORE_METRIC_COLS = ["r_precision", "ip@3", "ir@3", "jaccard@3", "map", "mrr"]

ABLATION_RECALL_METRIC_COLS = ["ir@3", "ir@4", "ir@5", "ir@6", "ir@7"]

ABLATION_TABLE_METRIC_COLS = [
    "r_precision",
    "ip@3",
    "ir@3",
    "ir@4",
    "ir@5",
    "ir@6",
    "ir@7",
    "ir@8",
    "jaccard@3",
    "map",
    "mrr",
]

METRIC_ZH = {
    "r_precision": "R-Precision",
    "ip@3": "IP@3",
    "ir@3": "IR@3",
    "ir@4": "IR@4",
    "ir@5": "IR@5",
    "ir@6": "IR@6",
    "ir@7": "IR@7",
    "ir@8": "IR@8",
    "jaccard@3": "Jaccard@3",
    "maxsim@3": "MaxSim@3",
    "map": "MAP (AP)",
    "mrr": "MRR",
}

METHOD_ORDER = [
    "Proposed",
    "Proposed-v2",
    "Qwen3-VL-Rerank-ImgCap+Link",
    "Qwen3-VL-Rerank-ImgCap",
    "Qwen3-VL-Rerank-Img",
    "Layout-Order",
    "Caption-BM25",
    "Caption-Dense-v4",
    "Zero-shot-CLIP",
]

METHOD_COLORS = {
    "Proposed": "#e63946",
    "Proposed-v2": "#f77f00",
    "Qwen3-VL-Rerank-ImgCap+Link": "#6a1b9a",
    "Qwen3-VL-Rerank-ImgCap": "#8e44ad",
    "Qwen3-VL-Rerank-Img": "#c39bd3",
    "Layout-Order": "#457b9d",
    "Caption-BM25": "#2a9d8f",
    "Caption-Dense-v4": "#e9c46a",
    "Zero-shot-CLIP": "#9b59b6",
}

METHOD_ZH = {
    "Proposed": "Proposed（本文方法）",
    "Proposed-v2": "Proposed-v2（候选池增强）",
    "Qwen3-VL-Rerank-ImgCap+Link": "Qwen3-VL-Rerank-ImgCap+Link（S_link chunk）",
    "Qwen3-VL-Rerank-ImgCap": "Qwen3-VL-Rerank-ImgCap（强基线）",
    "Qwen3-VL-Rerank-Img": "Qwen3-VL-Rerank-Img（弱基线）",
    "Layout-Order": "Layout-Order",
    "Caption-BM25": "Caption-BM25",
    "Caption-Dense-v4": "Caption-Dense-v4",
    "Zero-shot-CLIP": "Zero-shot CLIP",
}

ABLATION_INCREMENTAL_ORDER = [
    "DirectOnly",
    "Direct+Link",
    "Direct+Link+Layout",
    "Direct+Link+Layout+Type",
    "LG-JSSF",
    "LG-JSSF+ClusterAdd",
    "LG-JSSF+ClusterMul",
]

RECALL_BAR_COLORS = {
    "ir@3": "#1d3557",
    "ir@4": "#457b9d",
    "ir@5": "#2a9d8f",
    "ir@6": "#e9c46a",
    "ir@7": "#e76f51",
    "ir@8": "#f4a261",
}


def _available_metric_cols(df: pd.DataFrame, candidates: list[str] | None = None) -> list[str]:
    """按 METRIC_COLS 顺序返回 DataFrame 中实际存在的指标列。"""
    candidates = candidates or METRIC_COLS
    return [col for col in candidates if col in df.columns]


def _setup_matplotlib_zh() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _short_paper_id(paper_id: str) -> str:
    return paper_id.split(".pdf")[0] if ".pdf" in paper_id else paper_id[:20]


def aggregate_by_method(df: pd.DataFrame, metric_cols: list[str] | None = None) -> pd.DataFrame:
    """按方法聚合：均值、标准差、样本数。"""
    metric_cols = metric_cols or _available_metric_cols(df)
    agg = df.groupby("method_name")[metric_cols].agg(["mean", "std", "count"])
    agg.columns = ["_".join(col).strip() for col in agg.columns]
    agg = agg.reset_index()
    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    agg["_order"] = agg["method_name"].map(lambda m: method_rank.get(m, 99))
    return agg.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def build_summary_table(df: pd.DataFrame, metric_cols: list[str] | None = None) -> pd.DataFrame:
    """简洁汇总表：每方法一行，指标为 mean ± std。"""
    metric_cols = metric_cols or _available_metric_cols(df)
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        sub = df[df["method_name"] == method]
        if sub.empty:
            continue
        row: dict[str, Any] = {"method_name": method, "method_zh": METHOD_ZH.get(method, method), "n_papers": len(sub)}
        for col in metric_cols:
            mean = sub[col].mean()
            std = sub[col].std()
            row[f"{col}_mean"] = round(mean, 4)
            row[f"{col}_std"] = round(std, 4)
            row[col] = f"{mean:.3f} ± {std:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def compute_win_rates(
    df: pd.DataFrame,
    reference: str = "Proposed",
    metric_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    逐样本比较：reference 相对各 baseline 在各指标上的「胜率」。
    胜率 = reference 分数严格高于 baseline 的论文比例。
    """
    metric_cols = metric_cols or _available_metric_cols(df)
    papers = df["paper_id"].unique()
    baselines = [m for m in METHOD_ORDER if m != reference]

    rows: list[dict[str, Any]] = []
    for metric in metric_cols:
        for baseline in baselines:
            wins = ties = losses = 0
            for paper in papers:
                ref_val = df[(df["paper_id"] == paper) & (df["method_name"] == reference)][metric]
                base_val = df[(df["paper_id"] == paper) & (df["method_name"] == baseline)][metric]
                if ref_val.empty or base_val.empty:
                    continue
                r, b = float(ref_val.iloc[0]), float(base_val.iloc[0])
                if r > b:
                    wins += 1
                elif r < b:
                    losses += 1
                else:
                    ties += 1
            n = wins + ties + losses
            rows.append(
                {
                    "metric": metric,
                    "metric_zh": METRIC_ZH[metric],
                    "baseline": baseline,
                    "baseline_zh": METHOD_ZH.get(baseline, baseline),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "win_rate": round(wins / n, 3) if n else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _subplot_grid(n_items: int, max_cols: int = 4) -> tuple[int, int]:
    """计算多指标子图网格 (ncols, nrows)。"""
    ncols = min(max_cols, max(1, n_items))
    nrows = (n_items + ncols - 1) // ncols
    return ncols, nrows


def _grouped_bar_offsets(n_groups: int, group_width: float = 0.82) -> tuple[float, list[float]]:
    """返回 (bar_width, offsets) 用于分组柱状图，避免条柱重叠。"""
    bar_width = group_width / max(n_groups, 1)
    offsets = [(i - (n_groups - 1) / 2) * bar_width for i in range(n_groups)]
    return bar_width * 0.92, offsets


def plot_method_bar_chart(df: pd.DataFrame, metric_cols: list[str] | None = None) -> plt.Figure:
    """分组柱状图：每个指标一个子图，对比各方法均值。"""
    metric_cols = metric_cols or _available_metric_cols(df)
    _setup_matplotlib_zh()
    summary = aggregate_by_method(df, metric_cols)
    methods = [m for m in METHOD_ORDER if m in summary["method_name"].values]
    n_metrics = len(metric_cols)
    ncols, nrows = _subplot_grid(n_metrics, max_cols=4)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.4 * ncols, 4.0 * nrows),
        sharey=False,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, metric in enumerate(metric_cols):
        ax = axes_flat[idx]
        means = [summary.loc[summary["method_name"] == m, f"{metric}_mean"].iloc[0] for m in methods]
        stds = [summary.loc[summary["method_name"] == m, f"{metric}_std"].iloc[0] for m in methods]
        colors = [METHOD_COLORS.get(m, "#888888") for m in methods]
        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="white", linewidth=0.8, alpha=0.92)
        if "Proposed" in methods:
            bars[methods.index("Proposed")].set_edgecolor("#1d3557")
            bars[methods.index("Proposed")].set_linewidth(2)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [METHOD_ZH.get(m, m).replace("（本文方法）", "\n（本文）") for m in methods],
            rotation=35,
            ha="right",
            fontsize=7,
        )
        ax.set_title(METRIC_ZH[metric], fontsize=10, fontweight="bold")
        ax.set_ylim(0, min(1.05, max(means) + max(stds) + 0.12))
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle("Stage-2 图片重排序：各方法平均指标对比", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_heatmap(df: pd.DataFrame, metric_cols: list[str] | None = None) -> plt.Figure:
    """热力图：方法 × 指标，颜色为均值。"""
    metric_cols = metric_cols or _available_metric_cols(df)
    _setup_matplotlib_zh()
    summary = aggregate_by_method(df, metric_cols)
    methods = [m for m in METHOD_ORDER if m in summary["method_name"].values]
    matrix = np.array(
        [[summary.loc[summary["method_name"] == m, f"{c}_mean"].iloc[0] for c in metric_cols] for m in methods]
    )

    fig, ax = plt.subplots(figsize=(max(11, len(metric_cols) * 1.15), 4.8))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels([METRIC_ZH[c] for c in metric_cols], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_ZH.get(m, m) for m in methods])

    for i in range(len(methods)):
        for j in range(len(metric_cols)):
            val = matrix[i, j]
            color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=10)

    ax.set_title("方法 × 指标 平均得分热力图", fontsize=12, fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="平均分")
    fig.tight_layout()
    return fig


def plot_proposed_delta(df: pd.DataFrame, metric_cols: list[str] | None = None) -> plt.Figure:
    """Proposed 相对各 baseline 的平均分差（正值=Proposed 更好）。"""
    metric_cols = metric_cols or _available_metric_cols(df)
    _setup_matplotlib_zh()
    summary = build_summary_table(df, metric_cols)
    proposed = summary[summary["method_name"] == "Proposed"]
    if proposed.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "无 Proposed 数据", ha="center")
        return fig

    baselines = [m for m in METHOD_ORDER if m != "Proposed" and m in summary["method_name"].values]
    n_metrics = len(metric_cols)
    n_baselines = len(baselines)
    bar_width, _ = _grouped_bar_offsets(n_baselines, group_width=0.78)
    fig, ax = plt.subplots(figsize=(max(12, n_metrics * 1.6), 4.8))
    x = np.arange(n_metrics)

    for i, baseline in enumerate(baselines):
        base_row = summary[summary["method_name"] == baseline].iloc[0]
        deltas = [proposed.iloc[0][f"{c}_mean"] - base_row[f"{c}_mean"] for c in metric_cols]
        offset = (i - (n_baselines - 1) / 2) * bar_width
        ax.bar(x + offset, deltas, width=bar_width, label=METHOD_ZH.get(baseline, baseline), alpha=0.85)

    ax.axhline(0, color="#333", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_ZH[c] for c in metric_cols])
    ax.set_ylabel("Proposed 平均分 − Baseline 平均分")
    ax.set_title("Proposed 相对各 Baseline 的平均优势（>0 表示 Proposed 更好）", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8, ncol=min(4, n_baselines))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.22)
    return fig


def plot_per_paper_jaccard(df: pd.DataFrame, metric: str = "jaccard@3") -> plt.Figure:
    """逐论文折线/点图：一图看清各方法在同一篇论文上的差异。"""
    _setup_matplotlib_zh()
    pivot = df.pivot(index="paper_id", columns="method_name", values=metric)
    pivot = pivot[[m for m in METHOD_ORDER if m in pivot.columns]]
    pivot.index = [_short_paper_id(p) for p in pivot.index]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(pivot))
    for method in pivot.columns:
        ax.plot(
            x,
            pivot[method].values,
            marker="o",
            linewidth=2 if method == "Proposed" else 1.2,
            markersize=6 if method == "Proposed" else 4,
            label=METHOD_ZH.get(method, method),
            color=METHOD_COLORS.get(method, None),
            alpha=0.95 if method == "Proposed" else 0.75,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(METRIC_ZH[metric])
    ax.set_title(f"逐论文 {METRIC_ZH[metric]} 对比", fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    return fig


def _ablation_chart_layout(n_methods: int, n_metrics: int) -> dict[str, float | int]:
    """按方法数与指标数自适应图表尺寸，避免柱/图例重叠。"""
    group_width = max(0.40, min(0.88, 0.90 - 0.055 * max(0, n_metrics - 3)))
    fig_w = max(13.0, n_methods * (0.72 + 0.11 * n_metrics))
    fig_h = 5.4 + 0.22 * max(0, n_metrics - 4)
    legend_rows = (n_metrics + 3) // 4
    bottom = 0.26 + 0.07 * legend_rows
    legend_y = -0.18 - 0.075 * legend_rows
    xtick_fs = 7 if n_methods >= 7 else 8
    legend_fs = 7 if n_metrics >= 5 else 8
    legend_ncol = min(4, max(2, n_metrics))
    return {
        "group_width": group_width,
        "fig_w": fig_w,
        "fig_h": fig_h,
        "bottom": bottom,
        "legend_y": legend_y,
        "xtick_fs": xtick_fs,
        "legend_fs": legend_fs,
        "legend_ncol": legend_ncol,
    }


def plot_ablation_chart(
    ablation_df: pd.DataFrame,
    title: str,
    method_filter: list[str] | None = None,
    metric_cols: list[str] | None = None,
) -> plt.Figure:
    """消融分组柱状图：多指标并列，按方法分组。"""
    _setup_matplotlib_zh()
    if method_filter:
        plot_df = ablation_df[ablation_df["method_name"].isin(method_filter)].copy()
    else:
        plot_df = ablation_df.copy()
    default_metrics = metric_cols or ABLATION_CORE_METRIC_COLS
    metrics = _available_metric_cols(plot_df, default_metrics)
    if not metrics:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "无可用消融指标", ha="center")
        return fig
    summary = plot_df.groupby("method_name")[metrics].mean().reset_index()

    if method_filter:
        order = {m: i for i, m in enumerate(method_filter)}
        summary["_order"] = summary["method_name"].map(lambda m: order.get(m, 999))
        summary = summary.sort_values("_order").drop(columns=["_order"])

    n_methods = len(summary)
    n_metrics = len(metrics)
    layout = _ablation_chart_layout(n_methods, n_metrics)
    bar_width, offsets = _grouped_bar_offsets(n_metrics, group_width=layout["group_width"])
    fig, ax = plt.subplots(figsize=(layout["fig_w"], layout["fig_h"]))
    x = np.arange(n_methods)

    for i, metric in enumerate(metrics):
        ax.bar(
            x + offsets[i],
            summary[metric],
            width=bar_width,
            label=METRIC_ZH.get(metric, metric),
            alpha=0.88,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        summary["method_name"],
        rotation=42,
        ha="right",
        fontsize=layout["xtick_fs"],
    )
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontweight="bold", pad=10, fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, layout["legend_y"]),
        ncol=int(layout["legend_ncol"]),
        fontsize=layout["legend_fs"],
        frameon=False,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=layout["bottom"])
    return fig


def _ablation_drop_one_order(ablation_df: pd.DataFrame, fusion: str) -> list[str]:
    """fusion: 'Add' | 'Mul'"""
    suffix = f"({fusion})"
    full = f"FullCluster{fusion}"
    methods = [
        m
        for m in ablation_df["method_name"].unique()
        if m == full or (m.startswith("w/o") and m.endswith(suffix))
    ]
    order = {full: 0}
    for i, m in enumerate(sorted(m for m in methods if m != full), start=1):
        order[m] = i
    return sorted(methods, key=lambda m: order.get(m, 99))


def _draw_ablation_recall_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metrics: list[str],
    title: str,
) -> None:
    n_methods = len(summary)
    n_metrics = len(metrics)
    layout = _ablation_chart_layout(n_methods, n_metrics)
    bar_width, offsets = _grouped_bar_offsets(n_metrics, group_width=layout["group_width"])
    x = np.arange(n_methods)

    for i, metric in enumerate(metrics):
        ax.bar(
            x + offsets[i],
            summary[metric],
            width=bar_width,
            label=METRIC_ZH.get(metric, metric),
            color=RECALL_BAR_COLORS.get(metric),
            alpha=0.92,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        summary["method_name"],
        rotation=42,
        ha="right",
        fontsize=layout["xtick_fs"],
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Image Recall")
    ax.set_title(title, fontweight="bold", fontsize=10, pad=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_ablation_recall_bars(ablation_df: pd.DataFrame) -> plt.Figure:
    """
    消融专用：单图三面板，仅展示 IR@3/4/5/6/7 分组条形统计。
    上：递增式消融；中：FullClusterAdd + drop-one；下：FullClusterMul + drop-one。
    """
    _setup_matplotlib_zh()
    metrics = _available_metric_cols(ablation_df, ABLATION_RECALL_METRIC_COLS)
    if not metrics:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "无可用 IR@K 消融指标", ha="center")
        return fig

    panels: list[tuple[list[str], str]] = [
        (ABLATION_INCREMENTAL_ORDER, "递增式消融"),
        (_ablation_drop_one_order(ablation_df, "Add"), "FullClusterAdd 与 Drop-one（Additive）"),
        (_ablation_drop_one_order(ablation_df, "Mul"), "FullClusterMul 与 Drop-one（Multiplicative）"),
    ]
    panels = [(methods, title) for methods, title in panels if methods]

    max_methods = max(len(m) for m, _ in panels)
    layout = _ablation_chart_layout(max_methods, len(metrics))
    fig_h = 4.2 * len(panels) + 0.6 * layout["legend_ncol"]
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(max(14.0, layout["fig_w"]), fig_h),
        squeeze=False,
    )

    for ax_row, (methods, title) in zip(axes.flatten(), panels):
        plot_df = ablation_df[ablation_df["method_name"].isin(methods)]
        summary = plot_df.groupby("method_name")[metrics].mean().reset_index()
        order = {m: i for i, m in enumerate(methods)}
        summary["_order"] = summary["method_name"].map(lambda m: order.get(m, 999))
        summary = summary.sort_values("_order").drop(columns=["_order"])
        _draw_ablation_recall_panel(ax_row, summary, metrics, title)

    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(metrics),
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(
        "消融实验 Image Recall（IR@3 / IR@4 / IR@5 / IR@6 / IR@7）",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(hspace=0.58, bottom=0.10, top=0.93)
    return fig


def plot_cluster_grid(grid_df: pd.DataFrame) -> plt.Figure:
    """ClusterPrior grid search 图：不同 tau/beta/fusion 的 MAP。"""
    _setup_matplotlib_zh()
    if grid_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "无 grid search 数据", ha="center")
        return fig

    labels = [
        f"{row.fusion_mode}\nτ={row.tau}, β={row.beta}"
        for row in grid_df.itertuples(index=False)
    ]
    values = grid_df["map"].values
    colors = ["#e76f51" if m == "additive" else "#2a9d8f" for m in grid_df["fusion_mode"]]
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.55), 4.5))
    ax.bar(np.arange(len(labels)), values, color=colors, alpha=0.9)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("MAP")
    ax.set_title("ClusterPrior Grid Search（按 MAP 对比）", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    return fig


def export_stage2_reranking_visuals(
    df: pd.DataFrame,
    output_dir: Path,
    ablation_df: pd.DataFrame | None = None,
    grid_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """
    从逐论文结果 DataFrame 生成汇总表、PNG 图表与 HTML 报告。
    返回生成文件路径字典。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = _available_metric_cols(df)
    if not metric_cols:
        raise ValueError("结果 CSV 中未找到任何已知指标列，请检查 stage2_reranking_eval_results.csv 格式。")

    bar_metric_cols = _available_metric_cols(df, BAR_METRIC_COLS)
    recall_metric_cols = _available_metric_cols(df, RECALL_METRIC_COLS)

    summary_df = build_summary_table(df, metric_cols)
    agg_df = aggregate_by_method(df, metric_cols)
    win_df = compute_win_rates(df, metric_cols=metric_cols)

    summary_csv = output_dir / "stage2_reranking_summary.csv"
    summary_zh_csv = output_dir / "stage2_reranking_summary_zh.csv"
    win_csv = output_dir / "stage2_reranking_win_rates.csv"

    summary_df[["method_name", "method_zh", "n_papers"] + metric_cols].rename(
        columns={"method_name": "方法", "method_zh": "方法名称", "n_papers": "论文数", **METRIC_ZH}
    ).to_csv(summary_zh_csv, index=False, encoding="utf-8-sig")

    export_cols = ["method_name", "n_papers"] + [f"{c}_mean" for c in metric_cols] + [f"{c}_std" for c in metric_cols]
    summary_df[export_cols].to_csv(summary_csv, index=False, encoding="utf-8-sig")
    win_df.to_csv(win_csv, index=False, encoding="utf-8-sig")

    charts = {
        "bar": plot_method_bar_chart(df, bar_metric_cols or metric_cols),
        "heatmap": plot_heatmap(df, bar_metric_cols or metric_cols),
        "delta": plot_proposed_delta(df, bar_metric_cols or metric_cols),
        "per_paper": plot_per_paper_jaccard(df),
    }
    if recall_metric_cols:
        charts["ir_recall"] = plot_method_bar_chart(
            df,
            recall_metric_cols,
        )
    ablation_df = ablation_df if ablation_df is not None else pd.DataFrame()
    grid_df = grid_df if grid_df is not None else pd.DataFrame()
    if not ablation_df.empty:
        incremental = ABLATION_INCREMENTAL_ORDER
        drop_one = [
            m
            for m in ablation_df["method_name"].unique()
            if m.startswith("w/o") or m.startswith("FullCluster")
        ]
        ablation_recall_cols = _available_metric_cols(ablation_df, ABLATION_RECALL_METRIC_COLS)
        charts["ablation_incremental"] = plot_ablation_chart(
            ablation_df,
            "递增式消融：核心模块逐步加入",
            incremental,
            metric_cols=ABLATION_CORE_METRIC_COLS,
        )
        charts["ablation_drop_one"] = plot_ablation_chart(
            ablation_df,
            "Drop-one 消融：从 FullCluster 移除单个模块",
            drop_one,
            metric_cols=ABLATION_CORE_METRIC_COLS,
        )
        if ablation_recall_cols:
            charts["ablation_recall"] = plot_ablation_recall_bars(ablation_df)
    if not grid_df.empty:
        charts["cluster_grid"] = plot_cluster_grid(grid_df)
    png_paths: dict[str, Path] = {}
    b64: dict[str, str] = {}
    for name, fig in charts.items():
        png_path = output_dir / f"stage2_reranking_{name}.png"
        fig.savefig(png_path, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        png_paths[name] = png_path
        # re-render for b64 is wasteful; read file back
        b64[name] = base64.b64encode(png_path.read_bytes()).decode("ascii")

    html_path = output_dir / "stage2_reranking_report.html"
    html_path.write_text(
        _build_html(
            summary_df,
            win_df,
            b64,
            len(df["paper_id"].unique()),
            metric_cols=metric_cols,
            ablation_df=ablation_df,
            grid_df=grid_df,
        ),
        encoding="utf-8",
    )

    return {
        "summary_csv": summary_csv,
        "summary_zh_csv": summary_zh_csv,
        "win_rates_csv": win_csv,
        "html_report": html_path,
        **{f"png_{k}": v for k, v in png_paths.items()},
    }


def _build_html(
    summary_df: pd.DataFrame,
    win_df: pd.DataFrame,
    b64: dict[str, str],
    n_papers: int,
    metric_cols: list[str] | None = None,
    ablation_df: pd.DataFrame | None = None,
    grid_df: pd.DataFrame | None = None,
) -> str:
    metric_cols = metric_cols or _available_metric_cols(summary_df, METRIC_COLS)
    summary_rows = ""
    for _, row in summary_df.iterrows():
        cells = "".join(f"<td>{row[c]}</td>" for c in metric_cols)
        summary_rows += (
            f"<tr class=\"{'proposed' if row['method_name']=='Proposed' else ''}\">"
            f"<td><b>{row['method_zh']}</b></td><td>{int(row['n_papers'])}</td>{cells}</tr>\n"
        )

    win_rows = ""
    for _, row in win_df.iterrows():
        win_rows += (
            f"<tr><td>{row['metric_zh']}</td><td>{row['baseline_zh']}</td>"
            f"<td>{row['wins']}</td><td>{row['ties']}</td><td>{row['losses']}</td>"
            f"<td><b>{row['win_rate']:.1%}</b></td></tr>\n"
        )

    ablation_sections = ""
    if ablation_df is not None and not ablation_df.empty:
        ablation_metrics = _available_metric_cols(ablation_df, ABLATION_TABLE_METRIC_COLS)
        ablation_summary = ablation_df.groupby("method_name")[ablation_metrics].mean().reset_index()
        ablation_headers = "".join(f"<th>{METRIC_ZH.get(c, c)}</th>" for c in ablation_metrics)
        ablation_rows = ""
        for _, row in ablation_summary.iterrows():
            ablation_cells = "".join(f"<td>{row[c]:.3f}</td>" for c in ablation_metrics)
            ablation_rows += f"<tr><td>{row['method_name']}</td>{ablation_cells}</tr>\n"
        incremental_recall_img = ""
        if b64.get("ablation_recall"):
            incremental_recall_img = f"""
<div class="card">
<h2>8b. 消融 Image Recall 条形图 <span class="tag">IR@3/4/5/6/7</span></h2>
<p class="note">仅展示召回率指标：上为递增式消融，中为 FullClusterAdd 及 drop-one，下为 FullClusterMul 及 drop-one。K 越大表示从更深候选池中统计 GT 覆盖率。</p>
<div class="chart-wrap"><img src="data:image/png;base64,{b64['ablation_recall']}" alt="ablation recall bars"></div>
</div>
"""
        drop_one_recall_img = ""
        ablation_sections += f"""
<div class="card">
<h2>7. ClusterPrior 与递增式消融</h2>
<p class="note">展示 Direct、Link、Layout、Type 与 ClusterPrior 逐步加入后的核心指标。</p>
<div class="chart-wrap"><img src="data:image/png;base64,{b64.get('ablation_incremental', '')}" alt="incremental ablation"></div>
</div>

<div class="card">
<h2>8. Drop-one 消融</h2>
<p class="note">从 FullCluster 中移除单个模块，观察各指标的变化。</p>
<div class="chart-wrap"><img src="data:image/png;base64,{b64.get('ablation_drop_one', '')}" alt="drop-one ablation"></div>
</div>
{incremental_recall_img}

<div class="card">
<h2>9. 消融汇总表</h2>
<p class="note">含 IR@3/4/5/6/7；列较多时可横向滚动查看。</p>
<table class="summary-table">
<tr><th>方法</th>{ablation_headers}</tr>
{ablation_rows}
</table>
</div>
"""

    if grid_df is not None and not grid_df.empty:
        best = grid_df.sort_values(["map", "mrr", "r_precision"], ascending=[False, False, False]).iloc[0]
        ablation_sections += f"""
<div class="card">
<h2>10. ClusterPrior Grid Search</h2>
<p class="note">最佳配置：fusion=<b>{best['fusion_mode']}</b>, tau=<b>{best['tau']}</b>, beta=<b>{best['beta']}</b>, MAP=<b>{best['map']:.3f}</b>, MRR=<b>{best['mrr']:.3f}</b></p>
<div class="chart-wrap"><img src="data:image/png;base64,{b64.get('cluster_grid', '')}" alt="cluster grid search"></div>
</div>
"""

    metric_headers = "".join(f"<th>{METRIC_ZH[c]}</th>" for c in metric_cols)
    ir_recall_section = ""
    if b64.get("ir_recall"):
        ir_recall_section = f"""
<div class="card">
<h2>2b. Image Recall 多档 K 对比 <span class="tag">IR@3/4/5/6/7</span></h2>
<p class="note">扩大 Top-K 窗口后的 GT 覆盖率；|GT|&gt;3 时 IR@4/5/6/7 可观察更深候选池的召回增益。</p>
<div class="chart-wrap"><img src="data:image/png;base64,{b64['ir_recall']}" alt="ir recall chart"></div>
</div>
"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Stage-2 重排序 Baseline 对比报告</title>
<style>
body {{ font-family: "Microsoft YaHei", system-ui, sans-serif; margin: 2rem; background: #f5f6fa; color: #2d3436; }}
.card {{ background: #fff; border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
h1 {{ color: #1d3557; margin-bottom: .25rem; }}
.sub {{ color: #636e72; margin-bottom: 1.5rem; }}
h2 {{ color: #1d3557; border-bottom: 2px solid #e63946; padding-bottom: .4rem; font-size: 1.1rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: .9rem; }}
th, td {{ border: 1px solid #dfe6e9; padding: .55rem .75rem; text-align: center; }}
th {{ background: #1d3557; color: #fff; }}
tr.proposed {{ background: #fff5f5; }}
tr.proposed td:first-child {{ color: #e63946; }}
img {{ max-width: 100%; height: auto; border-radius: 6px; margin: .5rem 0; }}
.chart-wrap {{ overflow-x: auto; margin: .5rem 0; padding-bottom: .25rem; }}
.chart-wrap img {{ max-width: none; width: 100%; min-width: 820px; }}
table.summary-table {{ display: block; overflow-x: auto; white-space: nowrap; font-size: .82rem; }}
table.summary-table th, table.summary-table td {{ padding: .45rem .55rem; }}
.grid {{ display: grid; grid-template-columns: 1fr; gap: 1rem; }}
.note {{ font-size: .85rem; color: #636e72; line-height: 1.6; }}
.tag {{ display: inline-block; background: #ffeef0; color: #e63946; padding: .15rem .5rem; border-radius: 4px; font-size: .8rem; }}
</style>
</head>
<body>
<h1>Stage-2 图片重排序：Proposed vs Baselines</h1>
<p class="sub">基于 {n_papers} 篇 trial 论文 · 指标为逐论文得分在全体上的 <b>均值 ± 标准差</b></p>

<div class="card">
<h2>1. 方法整体表现（汇总表）</h2>
<p class="note">数字格式：<code>均值 ± 标准差</code>。Proposed 行高亮显示。</p>
<table class="summary-table">
<tr><th>方法</th><th>论文数</th>{metric_headers}</tr>
{summary_rows}
</table>
</div>

<div class="card">
<h2>2. 平均指标柱状图 <span class="tag">推荐首看</span></h2>
<p class="note">每个子图对应一个指标；误差棒为标准差；红色为 Proposed。</p>
<div class="chart-wrap"><img src="data:image/png;base64,{b64['bar']}" alt="bar chart"></div>
</div>
{ir_recall_section}

<div class="card">
<h2>3. 方法 × 指标 热力图</h2>
<div class="chart-wrap"><img src="data:image/png;base64,{b64['heatmap']}" alt="heatmap"></div>
</div>

<div class="card">
<h2>4. Proposed 相对 Baseline 平均优势</h2>
<p class="note">柱高 &gt; 0 表示 Proposed 在该指标上平均优于对应 baseline。</p>
<div class="chart-wrap"><img src="data:image/png;base64,{b64['delta']}" alt="delta chart"></div>
</div>

<div class="card">
<h2>5. 逐论文 Jaccard@3 走势</h2>
<p class="note">同一篇论文上各方法的横向对比，便于发现 Proposed 在哪些样本上赢/输。</p>
<div class="chart-wrap"><img src="data:image/png;base64,{b64['per_paper']}" alt="per paper chart"></div>
</div>

<div class="card">
<h2>6. Proposed 逐样本胜率（相对各 Baseline）</h2>
<p class="note">胜率 = Proposed 分数严格高于 baseline 的论文占比。</p>
<table>
<tr><th>指标</th><th>Baseline</th><th>胜</th><th>平</th><th>负</th><th>胜率</th></tr>
{win_rows}
</table>
</div>

<div class="card note">
<p><b>如何解读：</b></p>
<ul>
<li><b>Image Precision (IP@3)</b>：MSMO 标准，|Top-3 命中 GT| / 3 — 推荐精确率，Stage-3 输入质量。</li>
<li><b>Image Recall (IR@K)</b>：|Top-K 命中 GT| / |GT|；IR@3 对应 Stage-3 输入槽位，IR@4/5/6/7 用于观察扩大候选窗口后的 GT 覆盖率。</li>
<li><b>R-Precision</b>：取 Top-|GT| 的 recall 式指标；|GT|&gt;3 时比 IR@3 多看更深层候选。</li>
<li><b>Exact-match 指标</b>（Jaccard@3、MAP、MRR）衡量排序与 ID 命中。</li>
<li><b>MaxSim@3</b> 是软视觉相似度，高 MaxSim + 低 Jaccard 可能意味着「视觉相似但未 exact match」。</li>
<li>若 Proposed 在 exact-match 上不占优但 MaxSim 接近，可关注布局/共现模块是否拉高了语义相关但 ID 不同的图。</li>
</ul>
</div>
{ablation_sections}
</body>
</html>"""


def load_results_and_visualize(results_csv: Path, output_dir: Path | None = None) -> dict[str, Path]:
    """从已有 CSV 生成可视化（无需重新跑评估）。"""
    import warnings

    results_csv = Path(results_csv)
    output_dir = output_dir or results_csv.parent
    df = pd.read_csv(results_csv)

    metric_cols = _available_metric_cols(df)
    missing = [c for c in METRIC_COLS if c not in metric_cols]
    if missing:
        warnings.warn(
            f"结果 CSV 缺少指标列 {missing}，图表将仅展示 {metric_cols}。"
            "若需完整指标列，请用最新配置重新运行 evaluate_stage2_reranking.py。",
            stacklevel=2,
        )

    ablation_df = pd.DataFrame()
    ablation_path = output_dir / "stage2_ablation_results.csv"
    if ablation_path.is_file():
        ablation_df = pd.read_csv(ablation_path)

    grid_df = pd.DataFrame()
    grid_path = output_dir / "stage2_cluster_grid_search.csv"
    if grid_path.is_file():
        grid_df = pd.read_csv(grid_path)

    return export_stage2_reranking_visuals(
        df,
        output_dir,
        ablation_df=ablation_df if not ablation_df.empty else None,
        grid_df=grid_df if not grid_df.empty else None,
    )
