#!/usr/bin/env python3
"""Cross-method bad-case complementarity analysis."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib import paths as _paths  # noqa: F401
from lib.failure_sets import (  # noqa: E402
    aggregate_paper_metrics,
    build_gt_outcome_matrix,
)
from lib.figure_profile import (
    bucket_miss_rates,
    enrich_gt_profiles,
    shared_hard_cases,
)
from lib.io import load_rankings_jsonl, load_summary_csv, rankings_to_frames
from lib.overlap_metrics import method_pair_stats, single_method_ir, union_oracle_ir
from lib.paths import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    FOCUS_PAIRS,
    PRIMARY_METHODS,
    RECALL_KS,
    artifacts_dir,
    load_pipeline_config,
    report_dir,
)
from lib.visualize import (
    plot_miss_jaccard_heatmap,
    plot_rescue_heatmap,
    plot_union_ir_lines,
    plot_upset_miss_counts,
)
from m3sum.stage2_rerank.sample_loader import load_all_stage2_samples

logger = logging.getLogger(__name__)


def _write_summary_md(
    out_path: Path,
    *,
    pair_df: pd.DataFrame,
    union_df: pd.DataFrame,
    shared_df: pd.DataFrame,
    focus_pairs: list[tuple[str, str]],
) -> None:
    lines = ["# Complementarity Summary\n"]

    k_show = 5
    sub = pair_df[pair_df["k"] == k_show]
    lines.append(f"## Focus pairs @ K={k_show}\n")
    for a, b in focus_pairs:
        row = sub[(sub.method_a == a) & (sub.method_b == b)]
        if row.empty:
            row = sub[(sub.method_a == b) & (sub.method_b == a)]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(
            f"- **{a} vs {b}**: miss_jaccard={r.miss_jaccard}, "
            f"rescue {a}->{b}={int(r.rescue_a_to_b)}, rescue {b}->{a}={int(r.rescue_b_to_a)}, "
            f"kappa={r.cohen_kappa}"
        )

    lines.append(f"\n## Union IR\n")
    for name in ["Proposed", "Qwen3-VL-Rerank-ImgCap+Link", "Proposed+QwenLink"]:
        part = union_df[union_df.union_name == name].sort_values("k")
        if part.empty:
            continue
        ir_line = ", ".join(
            f"IR@{int(row['k'])}={row['ir@k']:.3f}" for _, row in part.iterrows()
        )
        lines.append(f"- **{name}**: {ir_line}")

    lines.append(f"\n## Shared hard cases @ K={k_show} (miss>=3 primary methods)\n")
    lines.append(f"- Count: {len(shared_df)}")
    if not shared_df.empty:
        for r in shared_df.head(8).itertuples():
            lines.append(f"  - {r.paper_id[:20]}... | {r.caption[:50]} | miss={int(r.miss_count)}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze method complementarity")
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

    gt_df = build_gt_outcome_matrix(rankings_by_method, samples, ks=RECALL_KS)
    out_root = report_dir(args.trial)
    out_root.mkdir(parents=True, exist_ok=True)
    gt_df.to_csv(out_root / "gt_outcome_matrix.csv", index=False, encoding="utf-8-sig")

    pair_frames = [method_pair_stats(gt_df, methods, k) for k in RECALL_KS]
    pair_df = pd.concat(pair_frames, ignore_index=True)
    pair_df.to_csv(out_root / "method_pair_overlap.csv", index=False, encoding="utf-8-sig")
    for k in RECALL_KS:
        pair_df[pair_df["k"] == k].to_csv(
            out_root / f"method_pair_overlap_k{k}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    union_groups = {
        "Proposed": ["Proposed"],
        "Proposed-v2": ["Proposed-v2"],
        "Qwen3-VL-Rerank-ImgCap+Link": ["Qwen3-VL-Rerank-ImgCap+Link"],
        "Layout-Order": ["Layout-Order"],
        "Proposed+Proposed-v2": ["Proposed", "Proposed-v2"],
        "Proposed+QwenLink": ["Proposed", "Qwen3-VL-Rerank-ImgCap+Link"],
        "Proposed+Layout": ["Proposed", "Layout-Order"],
        "Proposed+Qwen+Layout": [
            "Proposed",
            "Qwen3-VL-Rerank-ImgCap+Link",
            "Layout-Order",
        ],
    }
    union_df = union_oracle_ir(rankings_by_method, samples, union_groups, RECALL_KS)
    union_df.to_csv(out_root / "union_oracle_ir.csv", index=False, encoding="utf-8-sig")

    paper_df = aggregate_paper_metrics(rankings_by_method, samples, ks=RECALL_KS)
    paper_df.to_csv(out_root / "paper_metrics_recomputed.csv", index=False, encoding="utf-8-sig")

    summary_ref = load_summary_csv(config)
    if not summary_ref.empty and not paper_df.empty:
        recomputed = (
            paper_df.groupby("method_name")[["ir@3", "ir@5", "ir@7", "map", "mrr"]]
            .mean()
            .reset_index()
        )
        ref = summary_ref.set_index("method_name")
        rows = []
        for _, r in recomputed.iterrows():
            m = r["method_name"]
            if m not in ref.index:
                continue
            rows.append(
                {
                    "method_name": m,
                    "ir@3_recomputed": r["ir@3"],
                    "ir@3_eval": ref.loc[m].get("ir@3_mean"),
                    "ir@5_recomputed": r["ir@5"],
                    "ir@7_recomputed": r["ir@7"],
                    "delta_ir@3": round(r["ir@3"] - float(ref.loc[m].get("ir@3_mean", 0)), 4),
                }
            )
        pd.DataFrame(rows).to_csv(
            out_root / "metric_validation.csv", index=False, encoding="utf-8-sig"
        )

    profile_df = enrich_gt_profiles(gt_df, config)
    profile_df.to_csv(out_root / "gt_figure_profiles.csv", index=False, encoding="utf-8-sig")

    shared_df = shared_hard_cases(gt_df, methods[:5], k=5, min_shared_miss=3)
    if not profile_df.empty:
        shared_enriched = shared_df.merge(
            profile_df,
            on=["paper_id", "figure_id", "caption"],
            how="left",
            suffixes=("", "_prof"),
        )
    else:
        shared_enriched = shared_df
    shared_enriched.to_csv(out_root / "shared_hard_cases_k5.csv", index=False, encoding="utf-8-sig")

    bucket_df = bucket_miss_rates(
        gt_df, profile_df, methods[:5], k=5, bucket_col="type_category"
    )
    bucket_df.to_csv(out_root / "miss_rate_by_type_k5.csv", index=False, encoding="utf-8-sig")

    fig_dir = out_root / "figures"
    plot_miss_jaccard_heatmap(pair_df, methods[:6], 5, fig_dir / "miss_jaccard_k5.png")
    plot_rescue_heatmap(pair_df, methods[:6], 5, fig_dir / "rescue_matrix_k5.png")
    plot_union_ir_lines(union_df, fig_dir / "union_ir_lines.png")
    plot_upset_miss_counts(gt_df, methods[:5], 5, fig_dir / "upset_miss_k5.png")

    _write_summary_md(
        out_root / "complementarity_summary.md",
        pair_df=pair_df,
        union_df=union_df,
        shared_df=shared_enriched,
        focus_pairs=FOCUS_PAIRS,
    )

    logger.info("Complementarity analysis written to %s", out_root)


if __name__ == "__main__":
    main()
