#!/usr/bin/env python3
"""AllFig vs pre-recall analysis stratified by paper figure count (trial_31)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.analysis import (  # noqa: E402
    bin_aggregate,
    build_long_scores,
    build_paired_paper_table,
    correlation_summary,
    load_eval_results,
    metric_breakdown_by_bin,
    paper_figure_counts,
    threshold_summary,
)
from lib.paths import DEFAULT_OUTPUT_DIR, DEFAULT_REPORT_DIR, eval_csv_path, report_dir
from lib.report import build_html_report  # noqa: E402
from lib.visualize import (  # noqa: E402
    plot_binned_means,
    plot_delta_scatter,
    plot_delta_vs_figure_count,
    plot_metric_by_bin,
    plot_prerecall_breakdown,
    plot_score_vs_figure_count,
    plot_win_rate_by_bin,
)


def _run_strategy(
    df: pd.DataFrame,
    strategy: str,
    fig_dir: Path,
    prefix: str,
) -> dict[str, pd.DataFrame]:
    paired = build_paired_paper_table(df, strategy)
    long_df = build_long_scores(df, strategy)
    bin_df = bin_aggregate(paired)
    corr_df = correlation_summary(paired)
    thresh_df = threshold_summary(paired, threshold=30)

    figures: dict[str, Path] = {}
    if not paired.empty:
        plot_score_vs_figure_count(long_df, strategy, fig_dir / f"{prefix}_score_vs_figures.png")
        figures[f"{prefix}_score_vs_figures"] = fig_dir / f"{prefix}_score_vs_figures.png"

        plot_delta_scatter(paired, strategy, fig_dir / f"{prefix}_delta_scatter.png")
        figures[f"{prefix}_delta_scatter"] = fig_dir / f"{prefix}_delta_scatter.png"

        plot_delta_vs_figure_count(paired, strategy, fig_dir / f"{prefix}_delta_bars.png")
        figures[f"{prefix}_delta_bars"] = fig_dir / f"{prefix}_delta_bars.png"

        if not bin_df.empty:
            plot_binned_means(bin_df, strategy, fig_dir / f"{prefix}_binned_means.png")
            figures[f"{prefix}_binned_means"] = fig_dir / f"{prefix}_binned_means.png"

            plot_win_rate_by_bin(bin_df, strategy, fig_dir / f"{prefix}_win_rate.png")
            figures[f"{prefix}_win_rate"] = fig_dir / f"{prefix}_win_rate.png"

            plot_prerecall_breakdown(bin_df, strategy, fig_dir / f"{prefix}_prerecall_breakdown.png")
            figures[f"{prefix}_prerecall_breakdown"] = fig_dir / f"{prefix}_prerecall_breakdown.png"

        for metric, suffix in (("image_f1", "image_f1_bins"), ("rouge_l", "rouge_l_bins")):
            mdf = metric_breakdown_by_bin(df, strategy, metric)
            if not mdf.empty:
                plot_metric_by_bin(mdf, metric, strategy, fig_dir / f"{prefix}_{suffix}.png")
                figures[f"{prefix}_{suffix}"] = fig_dir / f"{prefix}_{suffix}.png"

    return {
        "paired": paired,
        "bin": bin_df,
        "corr": corr_df,
        "thresh": thresh_df,
        "figures": figures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AllFig vs PreRecall × 图片数专题分析")
    parser.add_argument("--trial", default="trial_31")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    out_report = (args.report_dir or report_dir(args.trial)).resolve()
    out_report.mkdir(parents=True, exist_ok=True)
    fig_dir = out_report / "figures"
    csv_path = eval_csv_path(output_dir)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing eval CSV: {csv_path}")

    df = load_eval_results(csv_path)
    fig_counts = paper_figure_counts(df)
    fig_counts.to_csv(out_report / "paper_figure_counts.csv", index=False, encoding="utf-8-sig")

    e2e = _run_strategy(df, "end_to_end_vlm", fig_dir, "e2e")
    rag = _run_strategy(df, "text_rag_then_rewrite", fig_dir, "rag")

    e2e["paired"].to_csv(out_report / "paired_e2e.csv", index=False, encoding="utf-8-sig")
    rag["paired"].to_csv(out_report / "paired_rag.csv", index=False, encoding="utf-8-sig")
    e2e["bin"].to_csv(out_report / "bin_summary_e2e.csv", index=False, encoding="utf-8-sig")
    rag["bin"].to_csv(out_report / "bin_summary_rag.csv", index=False, encoding="utf-8-sig")
    e2e["corr"].to_csv(out_report / "correlation_e2e.csv", index=False, encoding="utf-8-sig")
    rag["corr"].to_csv(out_report / "correlation_rag.csv", index=False, encoding="utf-8-sig")

    all_figures = {**e2e["figures"], **rag["figures"]}
    build_html_report(
        out_report / "report.html",
        paired_e2e=e2e["paired"],
        paired_rag=rag["paired"],
        bin_e2e=e2e["bin"],
        bin_rag=rag["bin"],
        corr_e2e=e2e["corr"],
        corr_rag=rag["corr"],
        thresh_e2e=e2e["thresh"],
        thresh_rag=rag["thresh"],
        figures=all_figures,
    )

    _write_summary_md(out_report / "summary.md", e2e, rag, fig_counts)
    print(f"Report: {out_report / 'report.html'}")


def _write_summary_md(
    path: Path,
    e2e: dict,
    rag: dict,
    fig_counts: pd.DataFrame,
) -> None:
    lines = [
        "# AllFig vs PreRecall × 图片数\n",
        f"图片数范围：{int(fig_counts['total_figure_count'].min())}–{int(fig_counts['total_figure_count'].max())}，"
        f"中位数 {int(fig_counts['total_figure_count'].median())}。\n",
    ]
    for label, result in [("E2E", e2e), ("RAG", rag)]:
        paired: pd.DataFrame = result["paired"]
        if paired.empty:
            continue
        lines.append(f"## {label}\n")
        lines.append(f"- AllFig 胜率：{paired['allfig_wins'].mean():.1%}\n")
        lines.append(f"- 平均 Δ：{paired['delta_allfig_minus_best'].mean():+.4f}\n")
        corr: pd.DataFrame = result["corr"]
        if not corr.empty:
            d = corr[corr["column"] == "delta_allfig_minus_best"]
            if not d.empty:
                lines.append(
                    f"- Spearman(图片数, Δ)：ρ={d.iloc[0]['spearman_rho']:.3f}, p={d.iloc[0]['p_value']:.4f}\n"
                )
        lines.append("\n")
    path.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
