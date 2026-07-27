#!/usr/bin/env python3
"""Plot Stage-3 generation evaluation results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from m3sum.config import load_config
from m3sum.eval.stage3_generation_viz import load_results_and_visualize

DEFAULT_CONFIG = SRC_ROOT / "configs" / "trial_31.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-3 多模态摘要评估可视化")
    parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG), help="配置文件路径")
    parser.add_argument("--csv", default=None, help="评估结果 CSV；默认读取配置 output_dir/eval")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    csv_path = Path(args.csv).resolve() if args.csv else config.eval_dir / "stage3_generation_eval_results.csv"
    paths = load_results_and_visualize(csv_path, config.eval_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
