#!/usr/bin/env python3
"""Ablation module contribution and bad-case overlap analysis."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib import paths as _paths  # noqa: F401
from lib.failure_sets import build_gt_outcome_matrix, hit_set, miss_set
from lib.io import load_rankings_jsonl, rankings_to_frames
from lib.paths import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    DROP_ONE_MODULES,
    INCREMENTAL_ABLATION_ORDER,
    RECALL_KS,
    artifacts_dir,
    load_pipeline_config,
    report_dir,
)
from lib.visualize import plot_dropone_contribution, plot_incremental_first_hit
from m3sum.stage2_rerank.sample_loader import load_all_stage2_samples

logger = logging.getLogger(__name__)

FULL_METHOD = "FullClusterAdd"
QWEN_METHOD = "Qwen3-VL-Rerank-ImgCap+Link"


def dropone_contribution(
    gt_df: pd.DataFrame,
    full_method: str,
    drop_methods: dict[str, str],
    ks: list[int],
) -> pd.DataFrame:
    rows: list[dict] = []
    for k in ks:
        full_hit = hit_set(gt_df, full_method, k)
        full_miss = miss_set(gt_df, full_method, k)
        for drop_name, module in drop_methods.items():
            drop_hit = hit_set(gt_df, drop_name, k)
            drop_miss = miss_set(gt_df, drop_name, k)
            rescue = full_hit & drop_miss
            harm = full_miss & drop_hit
            rows.append(
                {
                    "k": k,
                    "module": module,
                    "drop_method": drop_name,
                    "rescue": len(rescue),
                    "harm": len(harm),
                    "net": len(rescue) - len(harm),
                    "rescue_ids": ";".join(f"{p}:{f[:8]}" for p, f in sorted(rescue)),
                    "harm_ids": ";".join(f"{p}:{f[:8]}" for p, f in sorted(harm)),
                }
            )
    return pd.DataFrame(rows)


def incremental_first_hit(
    gt_df: pd.DataFrame,
    order: list[str],
    ks: list[int],
) -> pd.DataFrame:
    rows: list[dict] = []
    available = [m for m in order if f"{m}|hit@3" in gt_df.columns or any(f"{m}|hit@{k}" in gt_df.columns for k in ks)]
    for k in ks:
        for _, row in gt_df.iterrows():
            first = "never"
            for method in available:
                col = f"{method}|hit@{k}"
                if col in gt_df.columns and bool(row[col]):
                    first = method
                    break
            rows.append(
                {
                    "k": k,
                    "paper_id": row["paper_id"],
                    "figure_id": row["figure_id"],
                    "caption": row.get("caption", ""),
                    "first_hit_method": first,
                }
            )
    return pd.DataFrame(rows)


def module_miss_overlap(
    gt_df: pd.DataFrame,
    full_method: str,
    drop_methods: list[str],
    k: int,
) -> pd.DataFrame:
    full_miss = miss_set(gt_df, full_method, k)
    rows: list[dict] = []
    for drop_name in drop_methods:
        drop_miss = miss_set(gt_df, drop_name, k)
        rows.append(
            {
                "drop_method": drop_name,
                "k": k,
                "miss_jaccard_with_full": round(
                    len(full_miss & drop_miss) / max(len(full_miss | drop_miss), 1),
                    4,
                ),
                "only_full_miss": len(full_miss - drop_miss),
                "only_drop_miss": len(drop_miss - full_miss),
                "shared_miss": len(full_miss & drop_miss),
            }
        )
    return pd.DataFrame(rows)


def qwen_rescue_after_drop(
    gt_df: pd.DataFrame,
    drop_method: str,
    qwen_method: str,
    k: int,
) -> pd.DataFrame:
    drop_miss = miss_set(gt_df, drop_method, k)
    qwen_hit = hit_set(gt_df, qwen_method, k)
    rescued = drop_miss & qwen_hit
    rows = []
    for paper_id, figure_id in sorted(rescued):
        sub = gt_df[(gt_df.paper_id == paper_id) & (gt_df.figure_id == figure_id)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        rows.append(
            {
                "paper_id": paper_id,
                "figure_id": figure_id,
                "caption": r.get("caption", ""),
                "drop_method": drop_method,
                "qwen_rank": r.get(f"{qwen_method}|rank"),
                f"{drop_method}|rank": r.get(f"{drop_method}|rank"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ablation module contributions")
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
        raise SystemExit(f"Missing rankings cache: {rankings_path}")

    rankings_by_method = rankings_to_frames(records)
    ablation_methods = [
        m for m in rankings_by_method if m in INCREMENTAL_ABLATION_ORDER or m in DROP_ONE_MODULES
    ]
    if FULL_METHOD not in rankings_by_method and "LG-JSSF+ClusterAdd" in rankings_by_method:
        # treat LG-JSSF+ClusterAdd as full if FullClusterAdd absent
        pass

    # Build GT matrix only for ablation + qwen methods
    analysis_methods = list(dict.fromkeys(ablation_methods + [QWEN_METHOD, "Proposed"]))
    sub_rankings = {m: rankings_by_method[m] for m in analysis_methods if m in rankings_by_method}
    gt_df = build_gt_outcome_matrix(sub_rankings, samples, ks=RECALL_KS)

    out_root = report_dir(args.trial)
    out_root.mkdir(parents=True, exist_ok=True)

    full_ref = FULL_METHOD if FULL_METHOD in sub_rankings else "LG-JSSF+ClusterAdd"
    drop_map = {k: v for k, v in DROP_ONE_MODULES.items() if k in sub_rankings}

    drop_df = dropone_contribution(gt_df, full_ref, drop_map, RECALL_KS)
    drop_df.to_csv(out_root / "ablation_dropone_contribution.csv", index=False, encoding="utf-8-sig")
    for k in RECALL_KS:
        drop_df[drop_df["k"] == k].to_csv(
            out_root / f"ablation_dropone_contribution_k{k}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    inc_order = [m for m in INCREMENTAL_ABLATION_ORDER if m in sub_rankings]
    first_hit_df = incremental_first_hit(gt_df, inc_order, RECALL_KS)
    first_hit_df.to_csv(out_root / "ablation_incremental_first_hit.csv", index=False, encoding="utf-8-sig")

    overlap_df = module_miss_overlap(gt_df, full_ref, list(drop_map.keys()), k=5)
    overlap_df.to_csv(out_root / "ablation_miss_overlap_k5.csv", index=False, encoding="utf-8-sig")

    if QWEN_METHOD in sub_rankings and "w/o P_layout (Add)" in sub_rankings:
        qwen_df = qwen_rescue_after_drop(gt_df, "w/o P_layout (Add)", QWEN_METHOD, k=5)
        qwen_df.to_csv(out_root / "qwen_rescue_after_drop_layout_k5.csv", index=False, encoding="utf-8-sig")

    fig_dir = out_root / "figures"
    plot_dropone_contribution(drop_df, 5, fig_dir / "ablation_dropone_k5.png")
    plot_incremental_first_hit(first_hit_df, inc_order, 5, fig_dir / "ablation_incremental_k5.png")

    # Markdown summary
    lines = ["# Ablation Module Summary\n", f"Reference full method: {full_ref}\n"]
    sub5 = drop_df[drop_df["k"] == 5].sort_values("net", ascending=False)
    lines.append("## Drop-one @ K=5 (Rescue / Harm / Net)\n")
    for r in sub5.itertuples():
        lines.append(f"- **{r.module}**: rescue={r.rescue}, harm={r.harm}, net={r.net}")

    lines.append("\n## Incremental first-hit @ K=5\n")
    fh5 = first_hit_df[first_hit_df["k"] == 5]["first_hit_method"].value_counts()
    for name, cnt in fh5.items():
        lines.append(f"- {name}: {cnt}")

    (out_root / "ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Ablation analysis written to %s", out_root)


if __name__ == "__main__":
    main()
