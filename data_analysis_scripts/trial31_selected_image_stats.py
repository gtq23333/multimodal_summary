#!/usr/bin/env python3
"""统计并可视化 trial_31 标注样本中被选中图片数量的分布。"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "trial_31" / "manifest.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "trial_31"


def _setup_matplotlib_zh() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _short_paper_id(paper_id: str) -> str:
    return paper_id.split(".pdf")[0] if ".pdf" in paper_id else paper_id[:24]


def load_selected_image_counts(manifest_path: Path) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    for sample in manifest.get("samples", []):
        ann_path = REPO_ROOT / Path(sample["annotation_path"])
        if not ann_path.is_file():
            raise FileNotFoundError(f"标注文件不存在: {ann_path}")

        annotation = json.loads(ann_path.read_text(encoding="utf-8"))
        insertions = annotation.get("insertions", [])
        selected_count = len(insertions)
        source_types = [ins.get("source_type", "image") for ins in insertions]
        image_count = sum(1 for t in source_types if t == "image")
        table_count = sum(1 for t in source_types if t == "table")

        rows.append(
            {
                "paper_id": sample["paper_id"],
                "problem_key": sample.get("problem_key", ""),
                "selected_count": selected_count,
                "image_count": image_count,
                "table_count": table_count,
                "text_modified": annotation.get("abstract", {}).get("text_modified", False),
            }
        )

    return pd.DataFrame(rows).sort_values("selected_count", ascending=False).reset_index(drop=True)


def compute_summary_metrics(df: pd.DataFrame) -> dict[str, Any]:
    counts = df["selected_count"].astype(float)
    distribution = (
        df["selected_count"]
        .value_counts()
        .sort_index()
        .rename_axis("selected_count")
        .reset_index(name="paper_count")
    )
    distribution["ratio"] = distribution["paper_count"] / len(df)

    q1, median, q3 = np.percentile(counts, [25, 50, 75])
    mode_row = distribution.loc[distribution["paper_count"].idxmax()]

    return {
        "sample_count": int(len(df)),
        "total_selected_images": int(counts.sum()),
        "mean": float(counts.mean()),
        "std": float(counts.std(ddof=1)) if len(counts) > 1 else 0.0,
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mode_count": int(mode_row["selected_count"]),
        "mode_frequency": int(mode_row["paper_count"]),
        "image_total": int(df["image_count"].sum()),
        "table_total": int(df["table_count"].sum()),
        "text_modified_papers": int(df["text_modified"].sum()),
        "distribution": distribution,
    }


def plot_distribution(summary: dict[str, Any], output_path: Path) -> plt.Figure:
    _setup_matplotlib_zh()
    dist = summary["distribution"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(
        dist["selected_count"].astype(str),
        dist["paper_count"],
        color="#457b9d",
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_xlabel("每篇选中图片数量")
    ax.set_ylabel("论文篇数")
    ax.set_title("Trial 31 标注样本：选中图片数量分布")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    for bar, ratio in zip(bars, dist["ratio"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.05,
            f"{int(height)}\n({ratio:.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    stats_text = (
        f"n={summary['sample_count']}  "
        f"均值={summary['mean']:.2f}  "
        f"中位数={summary['median']:.1f}  "
        f"标准差={summary['std']:.2f}  "
        f"范围=[{summary['min']}, {summary['max']}]"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig


def plot_per_paper(df: pd.DataFrame, output_path: Path) -> plt.Figure:
    _setup_matplotlib_zh()
    plot_df = df.sort_values("selected_count", ascending=True).reset_index(drop=True)
    labels = [_short_paper_id(pid) for pid in plot_df["paper_id"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.22)))
    colors = ["#e63946" if c >= 5 else "#457b9d" for c in plot_df["selected_count"]]
    ax.barh(labels, plot_df["selected_count"], color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("选中图片数量")
    ax.set_ylabel("论文")
    ax.set_title("Trial 31 标注样本：各论文选中图片数量")
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    for idx, count in enumerate(plot_df["selected_count"]):
        ax.text(count + 0.05, idx, str(count), va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig


def plot_cumulative(summary: dict[str, Any], output_path: Path) -> plt.Figure:
    _setup_matplotlib_zh()
    dist = summary["distribution"].copy()
    dist["cumulative_ratio"] = dist["ratio"].cumsum()

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(
        dist["selected_count"],
        dist["cumulative_ratio"],
        marker="o",
        color="#2a9d8f",
        linewidth=2,
    )
    ax.set_xlabel("选中图片数量（阈值 k：≤ k 篇的累计占比）")
    ax.set_ylabel("累计论文占比")
    ax.set_title("Trial 31 标注样本：选中图片数量累计分布")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25, linestyle="--")

    for _, row in dist.iterrows():
        ax.annotate(
            f"{row['cumulative_ratio']:.0%}",
            (row["selected_count"], row["cumulative_ratio"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig


def write_html_report(
    summary: dict[str, Any],
    df: pd.DataFrame,
    image_paths: dict[str, Path],
    output_path: Path,
) -> None:
    dist = summary["distribution"]
    dist_rows = "".join(
        f"<tr><td>{int(r.selected_count)}</td>"
        f"<td>{int(r.paper_count)}</td>"
        f"<td>{r.ratio:.1%}</td></tr>"
        for r in dist.itertuples(index=False)
    )
    per_paper_rows = "".join(
        f"<tr><td>{row.paper_id}</td><td>{row.problem_key}</td>"
        f"<td>{row.selected_count}</td><td>{row.image_count}</td>"
        f"<td>{row.table_count}</td></tr>"
        for row in df.itertuples(index=False)
    )

    embedded_images = {
        key: base64.b64encode(path.read_bytes()).decode("ascii")
        for key, path in image_paths.items()
    }

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>Trial 31 选中图片数量统计</title>
  <style>
    body {{ font-family: "Microsoft YaHei", sans-serif; margin: 24px; color: #222; }}
    h1, h2 {{ color: #1d3557; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: center; }}
    th {{ background: #f1f5f9; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .metric-card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; }}
    .metric-card .value {{ font-size: 1.4rem; font-weight: 700; color: #457b9d; }}
    img {{ max-width: 100%; margin: 12px 0 24px; border: 1px solid #e2e8f0; }}
  </style>
</head>
<body>
  <h1>Trial 31 标注样本：选中图片数量统计</h1>

  <h2>汇总指标</h2>
  <div class="metrics">
    <div class="metric-card"><div>样本数</div><div class="value">{summary['sample_count']}</div></div>
    <div class="metric-card"><div>选中图片总数</div><div class="value">{summary['total_selected_images']}</div></div>
    <div class="metric-card"><div>均值</div><div class="value">{summary['mean']:.2f}</div></div>
    <div class="metric-card"><div>中位数</div><div class="value">{summary['median']:.1f}</div></div>
    <div class="metric-card"><div>标准差</div><div class="value">{summary['std']:.2f}</div></div>
    <div class="metric-card"><div>最小值 / 最大值</div><div class="value">{summary['min']} / {summary['max']}</div></div>
    <div class="metric-card"><div>Q1 / Q3</div><div class="value">{summary['q1']:.1f} / {summary['q3']:.1f}</div></div>
    <div class="metric-card"><div>众数</div><div class="value">{summary['mode_count']}（{summary['mode_frequency']} 篇）</div></div>
    <div class="metric-card"><div>图 / 表</div><div class="value">{summary['image_total']} / {summary['table_total']}</div></div>
  </div>

  <h2>数量分布</h2>
  <table>
    <tr><th>选中数量</th><th>论文篇数</th><th>占比</th></tr>
    {dist_rows}
  </table>

  <h2>可视化</h2>
  <h3>分布直方图</h3>
  <img src="data:image/png;base64,{embedded_images['distribution']}" alt="distribution"/>
  <h3>各论文明细</h3>
  <img src="data:image/png;base64,{embedded_images['per_paper']}" alt="per paper"/>
  <h3>累计分布</h3>
  <img src="data:image/png;base64,{embedded_images['cumulative']}" alt="cumulative"/>

  <h2>逐篇明细</h2>
  <table>
    <tr><th>paper_id</th><th>problem_key</th><th>选中数</th><th>图</th><th>表</th></tr>
    {per_paper_rows}
  </table>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def export_tables(
    df: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    per_paper_path = output_dir / "selected_image_counts.csv"
    summary_path = output_dir / "selected_image_summary.csv"
    distribution_path = output_dir / "selected_image_distribution.csv"

    df.to_csv(per_paper_path, index=False, encoding="utf-8-sig")

    summary_row = {k: v for k, v in summary.items() if k != "distribution"}
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    summary["distribution"].to_csv(distribution_path, index=False, encoding="utf-8-sig")

    return {
        "per_paper": per_paper_path,
        "summary": summary_path,
        "distribution": distribution_path,
    }


def run(manifest_path: Path, output_dir: Path) -> dict[str, Path]:
    df = load_selected_image_counts(manifest_path)
    summary = compute_summary_metrics(df)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = export_tables(df, summary, output_dir)

    dist_fig = plot_distribution(summary, output_dir / "selected_image_distribution.png")
    plt.close(dist_fig)
    per_paper_fig = plot_per_paper(df, output_dir / "selected_image_per_paper.png")
    plt.close(per_paper_fig)
    cum_fig = plot_cumulative(summary, output_dir / "selected_image_cumulative.png")
    plt.close(cum_fig)

    image_paths = {
        "distribution_png": output_dir / "selected_image_distribution.png",
        "per_paper_png": output_dir / "selected_image_per_paper.png",
        "cumulative_png": output_dir / "selected_image_cumulative.png",
    }
    paths.update(image_paths)

    report_path = output_dir / "selected_image_report.html"
    write_html_report(
        summary,
        df,
        {
            "distribution": image_paths["distribution_png"],
            "per_paper": image_paths["per_paper_png"],
            "cumulative": image_paths["cumulative_png"],
        },
        report_path,
    )
    paths["report"] = report_path

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 trial_31 标注样本中被选中图片数量分布")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"trial manifest 路径（默认: {DEFAULT_MANIFEST.relative_to(REPO_ROOT)}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认: {DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)}）",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"未找到 manifest: {manifest_path}")

    paths = run(manifest_path, args.output_dir.resolve())

    print("Trial 31 选中图片数量统计已完成：")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
