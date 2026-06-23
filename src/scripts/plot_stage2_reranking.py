#!/usr/bin/env python3
"""从 Stage-2 重排序评估 CSV 生成汇总图表与 HTML 报告。"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from m3sum.config import load_config
from m3sum.eval.stage2_reranking_viz import load_results_and_visualize

DEFAULT_CONFIG = SRC_ROOT / "configs" / "trial_10.yaml"


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = load_config(config_path)

    csv_path = config.eval_dir / "stage2_reranking_eval_results.csv"
    if not csv_path.is_file():
        print(f"未找到结果文件: {csv_path}")
        print("请先运行: python scripts/evaluate_stage2_reranking.py")
        sys.exit(1)

    paths = load_results_and_visualize(csv_path, config.eval_dir)
    print("可视化已生成：")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
