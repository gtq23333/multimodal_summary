#!/usr/bin/env python3
"""Evaluate Stage-3 multimodal summary generation artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from m3sum.config import load_config
from m3sum.eval.stage3_generation_eval import run_stage3_generation_eval

DEFAULT_CONFIG = SRC_ROOT / "configs" / "trial_31.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-3 多模态摘要 Likert Judge 评估")
    parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG), help="配置文件路径")
    parser.add_argument(
        "--artifact",
        action="append",
        help="仅评估指定 artifact JSON，可重复传入；默认扫描 stage3_generation/*/*.json",
    )
    parser.add_argument("--merge-results", action="store_true", default=None, help="与已有 CSV 合并")
    parser.add_argument("--no-merge-results", action="store_true", help="不合并已有 CSV")
    args = parser.parse_args()

    merge_results: bool | None = None
    if args.no_merge_results:
        merge_results = False
    elif args.merge_results:
        merge_results = True

    config = load_config(Path(args.config))
    artifact_paths = [Path(p).resolve() for p in args.artifact] if args.artifact else None
    df = run_stage3_generation_eval(
        config,
        artifact_paths=artifact_paths,
        merge_results=merge_results,
    )
    if not df.empty:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
