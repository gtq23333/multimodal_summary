from __future__ import annotations

import base64
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import FIGURE_COUNT_BIN_LABELS


def _setup_zh() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _short_paper_id(paper_id: str) -> str:
    return paper_id.split(".pdf")[0] if ".pdf" in paper_id else paper_id[:20]


def plot_score_vs_figure_count(long_df: pd.DataFrame, strategy: str, out_path: Path) -> None:
    _setup_zh()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {
        "AllFig": "#e76f51",
        "BestPreRecall": "#457b9d",
        "Proposed": "#2a9d8f",
        "Layout": "#6a4c93",
        "QwenVL": "#f4a261",
    }
    order = ["AllFig", "BestPreRecall", "Proposed", "Layout", "QwenVL"]
    for method in order:
        sub = long_df[long_df["method_family"] == method]
        if sub.empty:
            continue
        sub = sub.sort_values("total_figure_count")
        ax.scatter(
            sub["total_figure_count"],
            sub["comprehensive_score"],
            alpha=0.65,
            s=42,
            label=method,
            color=colors.get(method, "#333"),
            edgecolors="white",
            linewidths=0.4,
        )
        if len(sub) >= 3:
            z = np.polyfit(sub["total_figure_count"], sub["comprehensive_score"], 1)
            xs = np.linspace(sub["total_figure_count"].min(), sub["total_figure_count"].max(), 50)
            ax.plot(xs, np.poly1d(z)(xs), color=colors.get(method, "#333"), alpha=0.55, linewidth=1.5)

    ax.set_xlabel("正文图片总数")
    ax.set_ylabel("Comprehensive Score")
    ax.set_title(f"Comprehensive vs 图片数 — {strategy}")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    _save(fig, out_path)


def plot_delta_vs_figure_count(paired: pd.DataFrame, strategy: str, out_path: Path) -> None:
    _setup_zh()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sub = paired.sort_values("total_figure_count")
    colors = np.where(sub["delta_allfig_minus_best"] >= 0, "#2a9d8f", "#e76f51")
    ax.bar(
        range(len(sub)),
        sub["delta_allfig_minus_best"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axhline(0, color="#333", linewidth=1)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels([_short_paper_id(p) for p in sub["paper_id"]], rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("AllFig − Best PreRecall")
    ax.set_title(f"逐论文优势（绿=AllFig 胜）— {strategy}")
    ax.grid(axis="y", alpha=0.25)
    fig.subplots_adjust(bottom=0.32)
    _save(fig, out_path)


def plot_delta_scatter(paired: pd.DataFrame, strategy: str, out_path: Path) -> None:
    _setup_zh()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = np.where(paired["delta_allfig_minus_best"] >= 0, "#2a9d8f", "#e76f51")
    ax.scatter(
        paired["total_figure_count"],
        paired["delta_allfig_minus_best"],
        c=colors,
        s=55,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.5,
    )
    ax.axhline(0, color="#333", linewidth=1, linestyle="--")
    if len(paired) >= 3:
        z = np.polyfit(paired["total_figure_count"], paired["delta_allfig_minus_best"], 1)
        xs = np.linspace(paired["total_figure_count"].min(), paired["total_figure_count"].max(), 50)
        ax.plot(xs, np.poly1d(z)(xs), color="#457b9d", linewidth=1.8, label="线性趋势")
    ax.set_xlabel("正文图片总数")
    ax.set_ylabel("AllFig − Best PreRecall")
    ax.set_title(f"优势随图片数变化 — {strategy}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    _save(fig, out_path)


def plot_binned_means(bin_df: pd.DataFrame, strategy: str, out_path: Path) -> None:
    _setup_zh()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(bin_df))
    width = 0.35
    ax.bar(x - width / 2, bin_df["allfig_mean"], width, label="AllFig", color="#e76f51")
    ax.bar(x + width / 2, bin_df["best_prerecall_mean"], width, label="Best PreRecall", color="#457b9d")
    for idx, row in bin_df.iterrows():
        ax.text(x[idx] - width / 2, row["allfig_mean"] + 0.02, f"{row['allfig_mean']:.2f}", ha="center", fontsize=8)
        ax.text(x[idx] + width / 2, row["best_prerecall_mean"] + 0.02, f"{row['best_prerecall_mean']:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_df["figure_bin"])
    ax.set_xlabel("图片数分段")
    ax.set_ylabel("Comprehensive Score 均值")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"分段均值对比 — {strategy}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out_path)


def plot_win_rate_by_bin(bin_df: pd.DataFrame, strategy: str, out_path: Path) -> None:
    _setup_zh()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(bin_df))
    win = bin_df["allfig_win_rate"]
    lose = 1 - win
    ax.bar(x, win, label="AllFig 胜", color="#2a9d8f")
    ax.bar(x, lose, bottom=win, label="PreRecall 胜/平", color="#e76f51", alpha=0.85)
    for idx, row in bin_df.iterrows():
        ax.text(x[idx], row["allfig_win_rate"] / 2, f"{row['allfig_win_count']}/{row['paper_count']}", ha="center", va="center", fontsize=9, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_df["figure_bin"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("占比")
    ax.set_title(f"AllFig 胜率（相对 Best PreRecall）— {strategy}")
    ax.legend(loc="upper right")
    _save(fig, out_path)


def plot_prerecall_breakdown(bin_df: pd.DataFrame, strategy: str, out_path: Path) -> None:
    _setup_zh()
    cols = [c for c in ("proposed_mean", "layout_mean", "qwenvl_mean") if c in bin_df.columns]
    if not cols:
        return
    labels = {"proposed_mean": "Proposed", "layout_mean": "Layout", "qwenvl_mean": "QwenVL"}
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(bin_df))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(cols) + 1)
    ax.bar(x + offsets[0], bin_df["allfig_mean"], width, label="AllFig", color="#e76f51")
    for idx, col in enumerate(cols):
        ax.bar(x + offsets[idx + 1], bin_df[col], width, label=labels.get(col, col), alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_df["figure_bin"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Comprehensive 均值")
    ax.set_title(f"分段：AllFig vs 各 PreRecall — {strategy}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out_path)


def plot_metric_by_bin(metric_df: pd.DataFrame, metric: str, strategy: str, out_path: Path) -> None:
    _setup_zh()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(metric_df))
    width = 0.35
    ax.bar(x - width / 2, metric_df["allfig_mean"], width, label="AllFig", color="#e76f51")
    ax.bar(x + width / 2, metric_df["best_prerecall_mean"], width, label="Best PreRecall", color="#457b9d")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_df["figure_bin"])
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{metric} 分段均值 — {strategy}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out_path)


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")
