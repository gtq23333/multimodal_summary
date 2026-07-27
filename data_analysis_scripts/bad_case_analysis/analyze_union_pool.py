#!/usr/bin/env python3
"""Union candidate-pool size, overlap, and GT contribution analysis."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib import paths as _paths  # noqa: F401
from lib.io import load_rankings_jsonl, rankings_to_frames
from lib.paths import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    PRIMARY_METHODS,
    RECALL_KS,
    artifacts_dir,
    load_pipeline_config,
    report_dir,
)
from lib.union_pool_stats import (
    build_method_groups,
    build_pairwise_summary,
    build_per_paper_stats,
    build_quadrant_labels,
    aggregate_distribution,
)
from lib.visualize import (
    plot_actual_budget_vs_k,
    plot_overlap_vs_gain_scatter,
    plot_pool_jaccard_heatmap,
    plot_pool_size_cdf,
    plot_pool_size_hist,
    plot_pool_union_budget_heatmap,
)
from m3sum.stage2_rerank.sample_loader import load_all_stage2_samples

logger = logging.getLogger(__name__)

FOCUS_GROUPS_K6 = ["Proposed+Qwen+Layout", "PRIMARY_ALL_9"]
K_FOCUS = 6


def _write_summary_md(
    out_path: Path,
    *,
    dist_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    quadrant_df: pd.DataFrame,
    per_paper_df: pd.DataFrame,
) -> None:
    lines = ["# Union Pool Overlap Summary\n"]
    lines.append("## Dynamic budget @ K=6\n")

    for group in ["Proposed+Qwen+Layout", "PRIMARY_ALL_9"]:
        row = dist_df[(dist_df["group_name"] == group) & (dist_df["k"] == K_FOCUS)]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(
            f"- **{group}**: actual_budget median={r.actual_budget_median}, "
            f"mean={r.actual_budget_mean}, range=[{int(r.actual_budget_min)}, {int(r.actual_budget_max)}], "
            f"compression={r.compression_ratio_median:.3f}, "
            f"pool_gt_recall={r.pool_gt_recall_micro:.3f}, "
            f"mean_pool_jaccard={r.pairwise_pool_jaccard_mean:.3f}"
        )

    lines.append("\n## Quadrant counts @ K=6\n")
    if not quadrant_df.empty:
        counts = quadrant_df.groupby(["group_name", "quadrant"]).size().reset_index(name="n")
        for group in FOCUS_GROUPS_K6:
            sub = counts[counts["group_name"] == group]
            if sub.empty:
                continue
            parts = ", ".join(f"{r.quadrant}={int(r.n)}" for r in sub.itertuples())
            lines.append(f"- **{group}**: {parts}")

    lines.append("\n## Representative papers (Q3 complementary / Q4 dispersed failure)\n")
    if not quadrant_df.empty:
        for group in FOCUS_GROUPS_K6:
            for label in ("Q3_complementary", "Q4_dispersed_failure"):
                sub = quadrant_df[
                    (quadrant_df["group_name"] == group)
                    & (quadrant_df["quadrant"] == label)
                ].sort_values("union_gain", ascending=(label == "Q4_dispersed_failure"))
                if sub.empty:
                    continue
                lines.append(f"\n### {group} — {label}\n")
                for r in sub.head(4).itertuples():
                    lines.append(
                        f"- {r.paper_id[:28]}... | overlap={r.pairwise_pool_jaccard_mean:.3f}, "
                        f"gain={r.union_gain:.3f}, actual_budget={int(r.actual_budget)}, "
                        f"pool_recall={r.pool_gt_recall:.3f}"
                    )

    lines.append("\n## Top pairwise pool Jaccard @ K=6 (high overlap)\n")
    if not pairwise_df.empty:
        top = pairwise_df.nlargest(5, "pool_jaccard_mean")
        for r in top.itertuples():
            lines.append(
                f"- {r.method_a} vs {r.method_b}: jaccard={r.pool_jaccard_mean:.3f}, "
                f"median_budget={r.actual_budget_median:.1f}, union_gain={r.union_gain_mean:.3f}"
            )

    lines.append("\n## Lowest pairwise pool Jaccard @ K=6 (dispersed candidates)\n")
    if not pairwise_df.empty:
        low = pairwise_df.nsmallest(5, "pool_jaccard_mean")
        for r in low.itertuples():
            lines.append(
                f"- {r.method_a} vs {r.method_b}: jaccard={r.pool_jaccard_mean:.3f}, "
                f"median_budget={r.actual_budget_median:.1f}, union_gain={r.union_gain_mean:.3f}"
            )

    lines.append("\n## Interpretation guide\n")
    lines.append("- **Q1 redundant**: high pool overlap, low union gain — fusion adds little beyond one strong method.")
    lines.append("- **Q3 complementary**: low overlap, high gain — union/dynamic budget is most valuable.")
    lines.append("- **Q4 dispersed failure**: low overlap, low gain — methods disagree on wrong candidates; hard to rescue by union alone.")

    pql_k6 = per_paper_df[
        (per_paper_df["group_name"] == "Proposed+Qwen+Layout") & (per_paper_df["k"] == K_FOCUS)
    ]
    if not pql_k6.empty:
        micro = pql_k6["gt_in_pool"].sum() / max(pql_k6["n_gt"].sum(), 1)
        lines.append(f"\n**Sanity**: PQL micro pool_gt_recall @ K=6 = {micro:.4f}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Union pool overlap analysis")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trial", default="trial_31")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_pipeline_config(args.config, output_dir=args.output_dir)
    samples = load_all_stage2_samples(config)

    rankings_path = artifacts_dir(args.trial) / "rankings.jsonl"
    records = load_rankings_jsonl(rankings_path)
    if not records:
        raise SystemExit(f"Missing rankings cache: {rankings_path}. Run export_rankings.py first.")

    rankings_by_method = rankings_to_frames(records)
    methods = [m for m in PRIMARY_METHODS if m in rankings_by_method]
    if not methods:
        methods = sorted(rankings_by_method.keys())

    groups = build_method_groups(methods)
    per_paper_df = build_per_paper_stats(rankings_by_method, samples, groups, RECALL_KS)
    dist_df = aggregate_distribution(per_paper_df)
    pairwise_df = build_pairwise_summary(per_paper_df, K_FOCUS)
    quadrant_df = build_quadrant_labels(per_paper_df, FOCUS_GROUPS_K6, K_FOCUS)

    out_root = report_dir(args.trial) / "union_pool"
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)

    per_paper_df.to_csv(out_root / "union_pool_per_paper.csv", index=False, encoding="utf-8-sig")
    dist_df.to_csv(out_root / "union_pool_distribution.csv", index=False, encoding="utf-8-sig")
    quadrant_df.to_csv(out_root / "union_pool_quadrant_k6.csv", index=False, encoding="utf-8-sig")
    pairwise_df.to_csv(out_root / "union_pool_pairwise_k6.csv", index=False, encoding="utf-8-sig")

    plot_pool_jaccard_heatmap(pairwise_df, methods, K_FOCUS, fig_dir / "pool_jaccard_k6.png")
    plot_pool_union_budget_heatmap(pairwise_df, methods, K_FOCUS, fig_dir / "pool_actual_budget_k6.png")

    pql = per_paper_df[
        (per_paper_df["group_name"] == "Proposed+Qwen+Layout") & (per_paper_df["k"] == K_FOCUS)
    ]
    if not pql.empty:
        plot_pool_size_hist(
            pql,
            nominal_budget=int(pql.iloc[0]["nominal_budget"]),
            out_path=fig_dir / "pool_size_hist_pql_k6.png",
            title="PQL union pool size @ K=6",
        )

    primary9 = per_paper_df[
        (per_paper_df["group_name"] == "PRIMARY_ALL_9") & (per_paper_df["k"] == K_FOCUS)
    ]
    if not primary9.empty:
        plot_pool_size_cdf(
            primary9,
            out_path=fig_dir / "pool_size_cdf_primary9_k6.png",
            title="PRIMARY_ALL_9 union pool size CDF @ K=6",
        )

    if not quadrant_df.empty:
        plot_overlap_vs_gain_scatter(
            quadrant_df,
            out_path=fig_dir / "overlap_vs_gain_scatter_k6.png",
        )

    plot_actual_budget_vs_k(
        out_root / "union_pool_distribution.csv",
        fig_dir / "actual_budget_vs_k_lines.png",
    )

    _write_summary_md(
        out_root / "union_pool_summary.md",
        dist_df=dist_df,
        pairwise_df=pairwise_df,
        quadrant_df=quadrant_df,
        per_paper_df=per_paper_df,
    )

    logger.info("Union pool analysis written to %s", out_root)


if __name__ == "__main__":
    main()
