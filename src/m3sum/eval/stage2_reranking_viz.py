from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRIC_COLS = ["r_precision", "jaccard@3", "maxsim@3", "map", "mrr"]

METRIC_ZH = {
    "r_precision": "R-Precision",
    "jaccard@3": "Jaccard@3",
    "maxsim@3": "MaxSim@3",
    "map": "MAP (AP)",
    "mrr": "MRR",
}

METHOD_ORDER = [
    "Proposed",
    "Layout-Order",
    "Caption-BM25",
    "Caption-Dense-v4",
    "Zero-shot-CLIP",
]

METHOD_COLORS = {
    "Proposed": "#e63946",
    "Layout-Order": "#457b9d",
    "Caption-BM25": "#2a9d8f",
    "Caption-Dense-v4": "#e9c46a",
    "Zero-shot-CLIP": "#9b59b6",
}

METHOD_ZH = {
    "Proposed": "Proposed（本文方法）",
    "Layout-Order": "Layout-Order",
    "Caption-BM25": "Caption-BM25",
    "Caption-Dense-v4": "Caption-Dense-v4",
    "Zero-shot-CLIP": "Zero-shot CLIP",
}


def _setup_matplotlib_zh() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _short_paper_id(paper_id: str) -> str:
    return paper_id.split(".pdf")[0] if ".pdf" in paper_id else paper_id[:20]


def aggregate_by_method(df: pd.DataFrame) -> pd.DataFrame:
    """按方法聚合：均值、标准差、样本数。"""
    agg = df.groupby("method_name")[METRIC_COLS].agg(["mean", "std", "count"])
    agg.columns = ["_".join(col).strip() for col in agg.columns]
    agg = agg.reset_index()
    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    agg["_order"] = agg["method_name"].map(lambda m: method_rank.get(m, 99))
    return agg.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """简洁汇总表：每方法一行，指标为 mean ± std。"""
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        sub = df[df["method_name"] == method]
        if sub.empty:
            continue
        row: dict[str, Any] = {"method_name": method, "method_zh": METHOD_ZH.get(method, method), "n_papers": len(sub)}
        for col in METRIC_COLS:
            mean = sub[col].mean()
            std = sub[col].std()
            row[f"{col}_mean"] = round(mean, 4)
            row[f"{col}_std"] = round(std, 4)
            row[col] = f"{mean:.3f} ± {std:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def compute_win_rates(df: pd.DataFrame, reference: str = "Proposed") -> pd.DataFrame:
    """
    逐样本比较：reference 相对各 baseline 在各指标上的「胜率」。
    胜率 = reference 分数严格高于 baseline 的论文比例。
    """
    papers = df["paper_id"].unique()
    baselines = [m for m in METHOD_ORDER if m != reference]

    rows: list[dict[str, Any]] = []
    for metric in METRIC_COLS:
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


def plot_method_bar_chart(df: pd.DataFrame) -> plt.Figure:
    """分组柱状图：每个指标一个子图，对比各方法均值。"""
    _setup_matplotlib_zh()
    summary = aggregate_by_method(df)
    methods = [m for m in METHOD_ORDER if m in summary["method_name"].values]
    n_metrics = len(METRIC_COLS)

    fig, axes = plt.subplots(1, n_metrics, figsize=(3.2 * n_metrics, 4.5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, METRIC_COLS):
        means = [summary.loc[summary["method_name"] == m, f"{metric}_mean"].iloc[0] for m in methods]
        stds = [summary.loc[summary["method_name"] == m, f"{metric}_std"].iloc[0] for m in methods]
        colors = [METHOD_COLORS.get(m, "#888888") for m in methods]
        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="white", linewidth=0.8, alpha=0.92)
        if "Proposed" in methods:
            bars[methods.index("Proposed")].set_edgecolor("#1d3557")
            bars[methods.index("Proposed")].set_linewidth(2)

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_ZH.get(m, m).replace("（本文方法）", "\n（本文）") for m in methods], rotation=35, ha="right", fontsize=8)
        ax.set_title(METRIC_ZH[metric], fontsize=11, fontweight="bold")
        ax.set_ylim(0, min(1.05, max(means) + max(stds) + 0.12))
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Stage-2 图片重排序：各方法平均指标对比（10 篇 trial）", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_heatmap(df: pd.DataFrame) -> plt.Figure:
    """热力图：方法 × 指标，颜色为均值。"""
    _setup_matplotlib_zh()
    summary = aggregate_by_method(df)
    methods = [m for m in METHOD_ORDER if m in summary["method_name"].values]
    matrix = np.array(
        [[summary.loc[summary["method_name"] == m, f"{c}_mean"].iloc[0] for c in METRIC_COLS] for m in methods]
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(METRIC_COLS)))
    ax.set_xticklabels([METRIC_ZH[c] for c in METRIC_COLS], rotation=30, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_ZH.get(m, m) for m in methods])

    for i in range(len(methods)):
        for j in range(len(METRIC_COLS)):
            val = matrix[i, j]
            color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=10)

    ax.set_title("方法 × 指标 平均得分热力图", fontsize=12, fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="平均分")
    fig.tight_layout()
    return fig


def plot_proposed_delta(df: pd.DataFrame) -> plt.Figure:
    """Proposed 相对各 baseline 的平均分差（正值=Proposed 更好）。"""
    _setup_matplotlib_zh()
    summary = build_summary_table(df)
    proposed = summary[summary["method_name"] == "Proposed"]
    if proposed.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "无 Proposed 数据", ha="center")
        return fig

    baselines = [m for m in METHOD_ORDER if m != "Proposed" and m in summary["method_name"].values]
    fig, ax = plt.subplots(figsize=(9, 4))
    bar_width = 0.18
    x = np.arange(len(METRIC_COLS))

    for i, baseline in enumerate(baselines):
        base_row = summary[summary["method_name"] == baseline].iloc[0]
        deltas = [proposed.iloc[0][f"{c}_mean"] - base_row[f"{c}_mean"] for c in METRIC_COLS]
        offset = (i - len(baselines) / 2 + 0.5) * bar_width
        ax.bar(x + offset, deltas, width=bar_width, label=METHOD_ZH.get(baseline, baseline), alpha=0.85)

    ax.axhline(0, color="#333", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_ZH[c] for c in METRIC_COLS])
    ax.set_ylabel("Proposed 平均分 − Baseline 平均分")
    ax.set_title("Proposed 相对各 Baseline 的平均优势（>0 表示 Proposed 更好）", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
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


def plot_ablation_chart(
    ablation_df: pd.DataFrame,
    title: str,
    method_filter: list[str] | None = None,
) -> plt.Figure:
    """消融柱状图：展示 R-Precision、MAP、MRR。"""
    _setup_matplotlib_zh()
    if method_filter:
        plot_df = ablation_df[ablation_df["method_name"].isin(method_filter)].copy()
    else:
        plot_df = ablation_df.copy()
    metrics = ["r_precision", "map", "mrr"]
    summary = plot_df.groupby("method_name")[metrics].mean().reset_index()

    if method_filter:
        order = {m: i for i, m in enumerate(method_filter)}
        summary["_order"] = summary["method_name"].map(lambda m: order.get(m, 999))
        summary = summary.sort_values("_order").drop(columns=["_order"])

    fig, ax = plt.subplots(figsize=(max(8, len(summary) * 0.75), 4.5))
    x = np.arange(len(summary))
    width = 0.24
    for i, metric in enumerate(metrics):
        ax.bar(
            x + (i - 1) * width,
            summary[metric],
            width=width,
            label=METRIC_ZH[metric],
            alpha=0.88,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method_name"], rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
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

    summary_df = build_summary_table(df)
    agg_df = aggregate_by_method(df)
    win_df = compute_win_rates(df)

    summary_csv = output_dir / "stage2_reranking_summary.csv"
    summary_zh_csv = output_dir / "stage2_reranking_summary_zh.csv"
    win_csv = output_dir / "stage2_reranking_win_rates.csv"

    summary_df[["method_name", "method_zh", "n_papers"] + METRIC_COLS].rename(
        columns={"method_name": "方法", "method_zh": "方法名称", "n_papers": "论文数", **METRIC_ZH}
    ).to_csv(summary_zh_csv, index=False, encoding="utf-8-sig")

    export_cols = ["method_name", "n_papers"] + [f"{c}_mean" for c in METRIC_COLS] + [f"{c}_std" for c in METRIC_COLS]
    summary_df[export_cols].to_csv(summary_csv, index=False, encoding="utf-8-sig")
    win_df.to_csv(win_csv, index=False, encoding="utf-8-sig")

    charts = {
        "bar": plot_method_bar_chart(df),
        "heatmap": plot_heatmap(df),
        "delta": plot_proposed_delta(df),
        "per_paper": plot_per_paper_jaccard(df),
    }
    ablation_df = ablation_df if ablation_df is not None else pd.DataFrame()
    grid_df = grid_df if grid_df is not None else pd.DataFrame()
    if not ablation_df.empty:
        incremental = [
            "DirectOnly",
            "Direct+Link",
            "Direct+Link+Layout",
            "Direct+Link+Layout+Type",
            "LG-JSSF",
            "LG-JSSF+ClusterAdd",
            "LG-JSSF+ClusterMul",
        ]
        drop_one = [m for m in ablation_df["method_name"].unique() if m.startswith("w/o") or m.startswith("FullCluster")]
        charts["ablation_incremental"] = plot_ablation_chart(
            ablation_df,
            "递增式消融：核心模块逐步加入",
            incremental,
        )
        charts["ablation_drop_one"] = plot_ablation_chart(
            ablation_df,
            "Drop-one 消融：从 FullCluster 移除单个模块",
            drop_one,
        )
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
    ablation_df: pd.DataFrame | None = None,
    grid_df: pd.DataFrame | None = None,
) -> str:
    summary_rows = ""
    for _, row in summary_df.iterrows():
        cells = "".join(f"<td>{row[c]}</td>" for c in METRIC_COLS)
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
        ablation_summary = ablation_df.groupby("method_name")[["r_precision", "map", "mrr"]].mean().reset_index()
        ablation_rows = ""
        for _, row in ablation_summary.iterrows():
            ablation_rows += (
                f"<tr><td>{row['method_name']}</td>"
                f"<td>{row['r_precision']:.3f}</td>"
                f"<td>{row['map']:.3f}</td>"
                f"<td>{row['mrr']:.3f}</td></tr>\n"
            )
        ablation_sections += f"""
<div class="card">
<h2>7. ClusterPrior 与递增式消融</h2>
<p class="note">展示 Direct、Link、Layout、Type 与 ClusterPrior 逐步加入后的效果。</p>
<img src="data:image/png;base64,{b64.get('ablation_incremental', '')}" alt="incremental ablation">
</div>

<div class="card">
<h2>8. Drop-one 消融</h2>
<p class="note">从 FullCluster 中移除单个模块，观察 R-Precision / MAP / MRR 的变化。</p>
<img src="data:image/png;base64,{b64.get('ablation_drop_one', '')}" alt="drop-one ablation">
</div>

<div class="card">
<h2>9. 消融汇总表</h2>
<table>
<tr><th>方法</th><th>R-Precision</th><th>MAP</th><th>MRR</th></tr>
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
<img src="data:image/png;base64,{b64.get('cluster_grid', '')}" alt="cluster grid search">
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
<table>
<tr><th>方法</th><th>论文数</th><th>R-Precision</th><th>Jaccard@3</th><th>MaxSim@3</th><th>MAP</th><th>MRR</th></tr>
{summary_rows}
</table>
</div>

<div class="card">
<h2>2. 平均指标柱状图 <span class="tag">推荐首看</span></h2>
<p class="note">每个子图对应一个指标；误差棒为标准差；红色为 Proposed。</p>
<img src="data:image/png;base64,{b64['bar']}" alt="bar chart">
</div>

<div class="card">
<h2>3. 方法 × 指标 热力图</h2>
<img src="data:image/png;base64,{b64['heatmap']}" alt="heatmap">
</div>

<div class="card">
<h2>4. Proposed 相对 Baseline 平均优势</h2>
<p class="note">柱高 &gt; 0 表示 Proposed 在该指标上平均优于对应 baseline。</p>
<img src="data:image/png;base64,{b64['delta']}" alt="delta chart">
</div>

<div class="card">
<h2>5. 逐论文 Jaccard@3 走势</h2>
<p class="note">同一篇论文上各方法的横向对比，便于发现 Proposed 在哪些样本上赢/输。</p>
<img src="data:image/png;base64,{b64['per_paper']}" alt="per paper chart">
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
<li><b>Exact-match 指标</b>（R-Precision、Jaccard@3、MAP、MRR）衡量是否命中 GT 图片 ID。</li>
<li><b>MaxSim@3</b> 是软视觉相似度，高 MaxSim + 低 Jaccard 可能意味着「视觉相似但未 exact match」。</li>
<li>若 Proposed 在 exact-match 上不占优但 MaxSim 接近，可关注布局/共现模块是否拉高了语义相关但 ID 不同的图。</li>
</ul>
</div>
{ablation_sections}
</body>
</html>"""


def load_results_and_visualize(results_csv: Path, output_dir: Path | None = None) -> dict[str, Path]:
    """从已有 CSV 生成可视化（无需重新跑评估）。"""
    results_csv = Path(results_csv)
    output_dir = output_dir or results_csv.parent
    df = pd.read_csv(results_csv)
    return export_stage2_reranking_visuals(df, output_dir)
