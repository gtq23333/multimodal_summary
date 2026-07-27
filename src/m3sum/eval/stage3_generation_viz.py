from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCORE_COLS = ["cr", "icn", "ocdu", "overall"]
SCORE_LABELS = {
    "cr": "CR",
    "icn": "ICN",
    "ocdu": "OCDU",
    "overall": "Overall",
}


def _setup_matplotlib_zh() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _method_strategy_label(row: pd.Series) -> str:
    method = str(row.get("method_name", ""))
    strategy = str(row.get("strategy", ""))
    if method == "Reference-Oracle":
        return method
    return f"{method}\n{strategy}"


def plot_score_bars(df: pd.DataFrame) -> plt.Figure:
    _setup_matplotlib_zh()
    work = df.copy()
    work["group"] = work.apply(_method_strategy_label, axis=1)
    summary = work.groupby("group")[SCORE_COLS].mean().sort_values("overall", ascending=False)
    fig, ax = plt.subplots(figsize=(max(8, len(summary) * 1.2), 4.8))
    x = np.arange(len(summary))
    width = 0.18
    for idx, col in enumerate(SCORE_COLS):
        ax.bar(x + (idx - 1.5) * width, summary[col], width=width, label=SCORE_LABELS[col])
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Likert Score")
    ax.set_ylim(0, 5.2)
    ax.set_title("Stage3 多模态摘要评分均值")
    ax.legend(ncols=4, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    return fig


def plot_pool_size_trend(df: pd.DataFrame) -> plt.Figure:
    _setup_matplotlib_zh()
    work = df[df["method_name"] != "Reference-Oracle"].copy()
    if work.empty:
        work = df.copy()
    work["group"] = work.apply(_method_strategy_label, axis=1)
    summary = work.groupby(["group", "pool_size"])["overall"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for group, sub in summary.groupby("group"):
        sub = sub.sort_values("pool_size")
        ax.plot(sub["pool_size"], sub["overall"], marker="o", label=group)
    ax.set_xlabel("Candidate Pool Size")
    ax.set_ylabel("Overall")
    ax.set_ylim(0, 5.2)
    ax.set_title("候选池大小与摘要总体评分")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    return fig


def plot_recall_scatter(df: pd.DataFrame) -> plt.Figure:
    _setup_matplotlib_zh()
    work = df.copy()
    recall_col = _best_recall_col(work)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    if recall_col is None:
        ax.text(0.5, 0.5, "未发现 IR@K 列", ha="center", va="center")
        ax.set_axis_off()
        return fig
    ax.scatter(work[recall_col], work["overall"], alpha=0.72)
    if len(work) >= 2:
        corr = work[[recall_col, "overall"]].corr().iloc[0, 1]
        ax.set_title(f"{recall_col} 与 Overall 相关性 r={corr:.3f}")
    else:
        ax.set_title(f"{recall_col} 与 Overall")
    ax.set_xlabel(recall_col)
    ax.set_ylabel("Overall")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 5.2)
    ax.grid(alpha=0.25)
    return fig


def plot_winner_heatmap(df: pd.DataFrame) -> plt.Figure:
    _setup_matplotlib_zh()
    work = df.copy()
    work["group"] = work.apply(_method_strategy_label, axis=1)
    pivot = work.pivot_table(index="paper_id", columns="group", values="overall", aggfunc="max")
    if pivot.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "无可视化数据", ha="center", va="center")
        ax.set_axis_off()
        return fig
    winners = pivot.eq(pivot.max(axis=1), axis=0).astype(float)
    fig, ax = plt.subplots(figsize=(max(7, winners.shape[1] * 1.1), max(5, winners.shape[0] * 0.25)))
    im = ax.imshow(winners.values, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(winners.shape[1]))
    ax.set_xticklabels(winners.columns, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(np.arange(winners.shape[0]))
    ax.set_yticklabels([_short_paper_id(p) for p in winners.index], fontsize=6)
    ax.set_title("逐论文 Overall Winner")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    return fig


def export_stage3_generation_visuals(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        raise ValueError("Stage3 eval dataframe is empty")

    summary = (
        df.groupby(["method_name", "strategy", "model", "pool_size"])[SCORE_COLS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary_path = out_dir / "stage3_generation_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    figures: dict[str, plt.Figure] = {
        "score_bars": plot_score_bars(df),
        "pool_size_trend": plot_pool_size_trend(df),
        "recall_scatter": plot_recall_scatter(df),
        "winner_heatmap": plot_winner_heatmap(df),
    }
    image_paths: dict[str, Path] = {}
    encoded: dict[str, str] = {}
    for name, fig in figures.items():
        image_path = out_dir / f"stage3_generation_{name}.png"
        fig.savefig(image_path, dpi=140, bbox_inches="tight", facecolor="white")
        encoded[name] = _fig_to_base64(fig)
        image_paths[name] = image_path

    html_path = out_dir / "stage3_generation_report.html"
    html_path.write_text(_build_html_report(df, encoded), encoding="utf-8")
    return {"summary_csv": summary_path, "html_report": html_path, **image_paths}


def load_results_and_visualize(csv_path: Path, out_dir: Path | None = None) -> dict[str, Path]:
    df = pd.read_csv(csv_path)
    return export_stage3_generation_visuals(df, out_dir or csv_path.parent)


def _build_html_report(df: pd.DataFrame, encoded: dict[str, str]) -> str:
    top = (
        df.groupby(["method_name", "strategy", "model", "pool_size"])[SCORE_COLS]
        .mean()
        .sort_values("overall", ascending=False)
        .head(20)
        .round(3)
        .reset_index()
    )
    table_html = top.to_html(index=False, escape=False)
    imgs = "\n".join(
        f'<section><h2>{name}</h2><img src="data:image/png;base64,{b64}" /></section>'
        for name, b64 in encoded.items()
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Stage3 Generation Report</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 24px; color: #222; }}
    img {{ max-width: 100%; border: 1px solid #ddd; margin: 8px 0 24px; }}
    table {{ border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; }}
    th {{ background: #f6f6f6; }}
  </style>
</head>
<body>
  <h1>Stage3 多模态摘要生成评估报告</h1>
  <p>样本行数：{len(df)}；论文数：{df["paper_id"].nunique()}；实验组数：{df["experiment_id"].nunique()}。</p>
  <h2>Top Groups</h2>
  {table_html}
  {imgs}
</body>
</html>"""


def _best_recall_col(df: pd.DataFrame) -> str | None:
    candidates = sorted(
        [c for c in df.columns if c.startswith("ir@")],
        key=lambda c: int(c.split("@", 1)[1]) if c.split("@", 1)[1].isdigit() else 0,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _short_paper_id(paper_id: str) -> str:
    return paper_id.split(".pdf")[0] if ".pdf" in paper_id else paper_id[:18]
