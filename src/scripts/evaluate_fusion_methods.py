#!/usr/bin/env python3
"""Evaluate multi-ranker fusion methods on cached Stage-2 rankings."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))

from fusion_method.eval import run_fusion_eval  # noqa: E402
from fusion_method.types import DEFAULT_SOURCE_METHODS  # noqa: E402
from m3sum.config import load_config  # noqa: E402

DEFAULT_CONFIG = SRC_ROOT / "configs" / "trial_31.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs_copy" / "trial_31"
DEFAULT_RANKINGS = (
    REPO_ROOT
    / "data_analysis_scripts"
    / "bad_case_analysis"
    / "reports"
    / "trial_31"
    / "artifacts"
    / "rankings.jsonl"
)
DEFAULT_REPORT_DIR = (
    REPO_ROOT
    / "data_analysis_scripts"
    / "bad_case_analysis"
    / "reports"
    / "trial_31"
    / "fusion"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fusion methods (dual-track)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trial", default="trial_31")
    parser.add_argument("--rankings", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--pool-k", type=int, default=8)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCE_METHODS),
        help="Comma-separated source method names",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(args.config)
    config = replace(config, output_dir=args.output_dir.resolve())

    rankings_path = args.rankings or DEFAULT_RANKINGS
    report_dir = args.report_dir or (
        REPO_ROOT / "data_analysis_scripts" / "bad_case_analysis" / "reports" / args.trial / "fusion"
    )
    source_methods = [s.strip() for s in args.sources.split(",") if s.strip()]

    print(f"Config: {args.config}")
    print(f"Rankings: {rankings_path}")
    print(f"Sources: {source_methods}")
    print(f"Report dir: {report_dir}")

    per_paper, fixed_summary, pool_df = run_fusion_eval(
        config,
        rankings_path=rankings_path,
        output_dir=report_dir,
        source_methods=source_methods,
        pool_k=args.pool_k,
        rrf_k=args.rrf_k,
    )

    print("\n=== Fixed Budget Summary (mean) ===")
    print(fixed_summary.to_string(index=False))
    print("\n=== Pool Recall ===")
    print(pool_df.to_string(index=False))
    print(f"\nEvaluated {len(per_paper)} per-paper rows.")


if __name__ == "__main__":
    main()
