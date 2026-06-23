#!/usr/bin/env python3
"""Stage-2 图片重排序 baselines 与扩展指标评估。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from m3sum.config import load_config
from m3sum.eval.stage2_reranking_eval import run_stage2_reranking_eval

DEFAULT_CONFIG = SRC_ROOT / "configs" / "trial_10.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 reranking baselines 评估")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help="配置文件路径（默认 configs/trial_10.yaml）",
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="跳过 Zero-shot CLIP baseline 与 MaxSim@3（开发调试用）",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    df = run_stage2_reranking_eval(config, skip_clip=args.skip_clip)
    if not df.empty:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
