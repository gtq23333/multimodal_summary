from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRIC_COLS = ["r_precision", "ip@3", "ir@3", "ir@4", "ir@5", "ir@6", "ir@7", "jaccard@3", "map", "mrr"]
BAR_METRIC_COLS = ["r_precision", "ip@3", "map", "mrr"]
RECALL_METRIC_COLS = ["ir@3", "ir@4", "ir@5", "ir@6", "ir@7"]

METRIC_ZH = {
    "r_precision": "R-Precision",
    "ip@3": "IP@3",
    "ir@3": "IR@3",
    "ir@4": "IR@4",
    "ir@5": "IR@5",
    "ir@6": "IR@6",
    "ir@7": "IR@7",
    "jaccard@3": "Jaccard@3",
    "map": "MAP",
    "mrr": "MRR",
}

METHOD_ORDER = [
    "Proposed",
    "Proposed-v2",
    "Qwen3-VL-Rerank-ImgCap+Link",
    "Layout-Order",
    "Fusion-RRF-PQL",
    "Fusion-Borda-PQL",
    "Fusion-Weighted-PQL",
    "Fusion-Cascade-PQL",
    "Fusion-UnionRRF-PQL",
]

METHOD_COLORS = {
    "Proposed": "#e63946",
    "Proposed-v2": "#f77f00",
    "Qwen3-VL-Rerank-ImgCap+Link": "#6a1b9a",
    "Layout-Order": "#2a9d8f",
    "Fusion-RRF-PQL": "#1d3557",
    "Fusion-Borda-PQL": "#457b9d",
    "Fusion-Weighted-PQL": "#264653",
    "Fusion-Cascade-PQL": "#a8dadc",
    "Fusion-UnionRRF-PQL": "#e9c46a",
    "Union-Oracle-PQL": "#9b2226",
    "Pool-Union-PQL": "#94d2bd",
}

METHOD_ZH = {
    "Proposed": "Proposed",
    "Proposed-v2": "Proposed-v2",
    "Qwen3-VL-Rerank-ImgCap+Link": "Qwen Link",
    "Layout-Order": "Layout-Order",
    "Fusion-RRF-PQL": "Fusion-RRF",
    "Fusion-Borda-PQL": "Fusion-Borda",
    "Fusion-Weighted-PQL": "Fusion-Weighted",
    "Fusion-Cascade-PQL": "Fusion-Cascade",
    "Fusion-UnionRRF-PQL": "Fusion-UnionRRF",
    "Union-Oracle-PQL": "Union Oracle",
    "Pool-Union-PQL": "Pool Union",
}


def _setup_matplotlib_zh() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _ordered_methods(df: pd.DataFrame) -> list[str]:
    present = set(df["method_name"].unique())
    ordered = [m for m in METHOD_ORDER if m in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def plot_main_metrics_bar(fixed_summary: pd.DataFrame) -> plt.Figure:
    _setup_matplotlib_zh()
    metrics = [c for c in BAR_METRIC_COLS if c in fixed_summary.columns]
    methods = _ordered_methods(fixed_summary)
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 4.5), squeeze=False)
    axes_flat = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        values = [
            fixed_summary.loc[fixed_summary["method_name"] == m, metric].iloc[0]
            for m in methods
        ]
        colors = [METHOD_COLORS.get(m, "#888888") for m in methods]
        x = np.arange(len(methods))
        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8, alpha=0.92)
        if "Proposed" in methods:
            bars[methods.index("Proposed")].set_edgecolor("#1d3557")
            bars[methods.index("Proposed")].set_linewidth(2)
        if "Fusion-RRF-PQL" in methods:
            bars[methods.index("Fusion-RRF-PQL")].set_edgecolor("#e63946")
            bars[methods.index("Fusion-RRF-PQL")].set_linewidth(2)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [METHOD_ZH.get(m, m) for m in methods],
            rotation=40,
            ha="right",
            fontsize=7,
        )
        ax.set_title(METRIC_ZH.get(metric, metric), fontsize=10, fontweight="bold")
        ax.set_ylim(0, min(1.05, max(values) + 0.12))
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("多路融合 vs 单方法：主指标对比（固定输出预算）", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_ir_recall_lines(fixed_summary: pd.DataFrame) -> plt.Figure:
    _setup_matplotlib_zh()
    metrics = [c for c in RECALL_METRIC_COLS if c in fixed_summary.columns]
    if not metrics:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "无 IR@K 数据", ha="center")
        return fig

    ks = [int(c.split("@")[1]) for c in metrics]
    methods = _ordered_methods(fixed_summary)
    fig, ax = plt.subplots(figsize=(9, 5))

    for method in methods:
        row = fixed_summary[fixed_summary["method_name"] == method].iloc[0]
        ys = [row[c] for c in metrics]
        color = METHOD_COLORS.get(method, "#888888")
        lw = 2.5 if method.startswith("Fusion") else 1.5
        ls = "-" if not method.startswith("Fusion") else "--"
        ax.plot(ks, ys, marker="o", label=METHOD_ZH.get(method, method), color=color, linewidth=lw, linestyle=ls)

    ax.set_xlabel("K")
    ax.set_ylabel("Image Recall (IR@K)")
    ax.set_title("IR@K 曲线：固定预算融合 vs 单方法", fontsize=12, fontweight="bold")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()
    return fig


def plot_dual_track(pool_df: pd.DataFrame, fixed_summary: pd.DataFrame) -> plt.Figure:
    _setup_matplotlib_zh()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: fixed budget IR@5 / IR@6 / IR@7
    ax = axes[0]
    methods = _ordered_methods(fixed_summary)
    focus = [m for m in methods if m in fixed_summary["method_name"].values]
    x = np.arange(len(focus))
    width = 0.25
    ir_cols = [("ir@5", "IR@5", "#457b9d"), ("ir@6", "IR@6", "#2a9d8f"), ("ir@7", "IR@7", "#e63946")]
    offsets = [-width, 0.0, width]
    for (col, label, color), offset in zip(ir_cols, offsets):
        if col not in fixed_summary.columns:
            continue
        values = [
            fixed_summary.loc[fixed_summary["method_name"] == m, col].iloc[0]
            for m in focus
        ]
        ax.bar(x + offset, values, width, label=label, color=color, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_ZH.get(m, m) for m in focus], rotation=35, ha="right", fontsize=7)
    ax.set_title("Track A：固定输出预算", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Right: pool / oracle recall
    ax = axes[1]
    oracle = pool_df[pool_df["metric_type"] == "union_oracle_ir"].sort_values("k")
    pool_cov = pool_df[pool_df["metric_type"] == "pool_coverage"]
    fusion_pool = pool_df[pool_df["metric_type"] == "fusion_pool_ir"].sort_values("k")

    if not oracle.empty:
        ax.plot(
            oracle["k"],
            oracle["ir@k"],
            marker="s",
            color=METHOD_COLORS["Union-Oracle-PQL"],
            linewidth=2.5,
            label="Union Oracle IR@K",
        )
    if not fusion_pool.empty:
        ax.plot(
            fusion_pool["k"],
            fusion_pool["ir@k"],
            marker="o",
            color=METHOD_COLORS["Fusion-UnionRRF-PQL"],
            linewidth=2,
            linestyle="--",
            label="Fusion-UnionRRF IR@K",
        )
    if not pool_cov.empty:
        cov = pool_cov.iloc[0]["pool_gt_coverage"]
        ax.axhline(
            cov,
            color=METHOD_COLORS["Pool-Union-PQL"],
            linestyle=":",
            linewidth=2,
            label=f"Pool Union GT覆盖率 (pool_k={pool_cov.iloc[0]['pool_k']})",
        )

    ax.set_xlabel("K")
    ax.set_ylabel("Recall / Coverage")
    ax.set_title("Track B：候选池召回", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3, linestyle="--")

    fig.suptitle("双轨评估对比", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def _summary_table_html(df: pd.DataFrame, title: str) -> str:
    if df.empty:
        return f"<h3>{title}</h3><p>无数据</p>"
    cols = list(df.columns)
    header = "".join(f"<th>{c}</th>" for c in cols)
    rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{row[c]}</td>" for c in cols)
        cls = ' class="fusion"' if str(row.get("method_name", "")).startswith("Fusion") else ""
        rows.append(f"<tr{cls}>{cells}</tr>")
    return f"""
<h3>{title}</h3>
<table>
<tr>{header}</tr>
{"".join(rows)}
</table>
"""


def _interpretation_lines(fixed_summary: pd.DataFrame, pool_df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    proposed = fixed_summary[fixed_summary["method_name"] == "Proposed"]
    rrf = fixed_summary[fixed_summary["method_name"] == "Fusion-RRF-PQL"]
    qwen = fixed_summary[fixed_summary["method_name"] == "Qwen3-VL-Rerank-ImgCap+Link"]
    oracle = pool_df[pool_df["metric_type"] == "union_oracle_ir"]

    if not proposed.empty and not rrf.empty:
        for k in (5, 6, 7):
            col = f"ir@{k}"
            if col not in proposed.columns or col not in rrf.columns:
                continue
            p = proposed.iloc[0].get(col, float("nan"))
            r = rrf.iloc[0].get(col, float("nan"))
            if p == p and r == r:
                lines.append(f"Fusion-RRF 相对 Proposed 的 {col.upper()} 提升约 {r - p:+.3f}。")

    if not rrf.empty and not oracle.empty:
        for k in (5, 6, 7):
            col = f"ir@{k}"
            if col not in rrf.columns:
                continue
            r_val = rrf.iloc[0].get(col, float("nan"))
            o_row = oracle[oracle["k"] == k]
            if not o_row.empty and r_val == r_val:
                o_val = o_row.iloc[0]["ir@k"]
                lines.append(
                    f"Fusion-RRF {col.upper()} ({r_val:.3f}) 仍低于 Union Oracle ({o_val:.3f})，差距约 {o_val - r_val:.3f}。"
                )

    if not qwen.empty and not rrf.empty:
        q3 = qwen.iloc[0].get("ip@3", float("nan"))
        r3 = rrf.iloc[0].get("ip@3", float("nan"))
        if q3 == q3 and r3 == r3:
            lines.append(f"Fusion-RRF 的 IP@3 ({r3:.3f}) vs Qwen ({q3:.3f})：融合可能以精度换召回。")

    lines.append(
        "口径说明：Track A 为固定输出预算（与单方法公平对比）；"
        "Track B 的 Union Oracle 为不等预算上限，适合作为 proposal 阶段召回潜力证据。"
    )
    return lines


def _build_interpretation(fixed_summary: pd.DataFrame, pool_df: pd.DataFrame) -> str:
    lines = _interpretation_lines(fixed_summary, pool_df)
    return "<ul>" + "".join(f"<li><b>{line}</b></li>" if line.startswith("口径") else f"<li>{line}</li>" for line in lines) + "</ul>"


def export_fusion_visuals(
    fixed_summary: pd.DataFrame,
    pool_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_bar = plot_main_metrics_bar(fixed_summary)
    fig_ir = plot_ir_recall_lines(fixed_summary)
    fig_dual = plot_dual_track(pool_df, fixed_summary)

    paths: dict[str, Path] = {}
    for key, fig in [("bar", fig_bar), ("ir_lines", fig_ir), ("dual_track", fig_dual)]:
        path = output_dir / f"fusion_{key}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths[key] = path

    b64 = {
        key: base64.b64encode(path.read_bytes()).decode("ascii")
        for key, path in paths.items()
        if path.suffix == ".png"
    }

    html_path = output_dir / "fusion_report.html"
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>多路融合方法评估报告</title>
<style>
body {{ font-family: "Microsoft YaHei", system-ui, sans-serif; margin: 2rem; background: #f5f6fa; color: #2d3436; }}
.card {{ background: #fff; border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
h1 {{ color: #1d3557; }}
h2 {{ color: #1d3557; border-bottom: 2px solid #e63946; padding-bottom: .4rem; font-size: 1.1rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: .85rem; margin: .5rem 0; }}
th, td {{ border: 1px solid #dfe6e9; padding: .45rem .6rem; text-align: center; }}
th {{ background: #1d3557; color: #fff; }}
tr.fusion {{ background: #f0f7ff; }}
img {{ max-width: 100%; height: auto; border-radius: 6px; }}
.note {{ font-size: .85rem; color: #636e72; line-height: 1.6; }}
</style>
</head>
<body>
<h1>多路融合方法评估（Proposed + Qwen Link + Layout）</h1>
<p class="note">默认融合来源：Proposed、Qwen3-VL-Rerank-ImgCap+Link、Layout-Order</p>

<div class="card">
<h2>1. 固定预算主指标</h2>
<img src="data:image/png;base64,{b64['bar']}" alt="bar">
{_summary_table_html(fixed_summary, "固定预算汇总（micro mean）")}
</div>

<div class="card">
<h2>2. IR@K 曲线</h2>
<img src="data:image/png;base64,{b64['ir_lines']}" alt="ir lines">
</div>

<div class="card">
<h2>3. 双轨对比</h2>
<img src="data:image/png;base64,{b64['dual_track']}" alt="dual track">
{_summary_table_html(pool_df, "候选池召回（Track B）")}
</div>

<div class="card note">
<h2>4. 判读要点</h2>
{_build_interpretation(fixed_summary, pool_df)}
</div>
</body>
</html>"""
    html_path.write_text(html, encoding="utf-8")
    paths["html"] = html_path

    md_path = output_dir / "fusion_summary.md"
    md_path.write_text(
        "# 多路融合方法判读摘要\n\n"
        + "\n".join(f"- {line}" for line in _interpretation_lines(fixed_summary, pool_df))
        + "\n\n## 固定预算\n\n"
        + fixed_summary.to_markdown(index=False)
        + "\n\n## 候选池召回\n\n"
        + pool_df.to_markdown(index=False),
        encoding="utf-8",
    )
    paths["md"] = md_path

    return paths


def load_and_visualize(report_dir: Path) -> dict[str, Path]:
    report_dir = Path(report_dir)
    fixed_path = report_dir / "fusion_eval_fixed_budget.csv"
    pool_path = report_dir / "fusion_eval_pool_recall.csv"
    if not fixed_path.is_file():
        raise FileNotFoundError(f"Missing {fixed_path}; run evaluate_fusion_methods.py first")
    fixed_summary = pd.read_csv(fixed_path)
    pool_df = pd.read_csv(pool_path) if pool_path.is_file() else pd.DataFrame()
    return export_fusion_visuals(fixed_summary, pool_df, report_dir)
