#!/usr/bin/env python3
"""在同集 trial_20 样本上对比改造前/后 LG-JSSF。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from m3sum.config import load_config
from m3sum.eval.legacy_compare_eval import run_legacy_compare_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy vs New LG-JSSF comparison")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(SRC_ROOT / "configs" / "trial_20.yaml"),
    )
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--force-legacy", action="store_true", help="强制重跑 stage2_legacy")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    run_legacy_compare_eval(
        config,
        skip_clip=args.skip_clip,
        force_legacy_rerun=args.force_legacy,
    )


if __name__ == "__main__":
    main()
