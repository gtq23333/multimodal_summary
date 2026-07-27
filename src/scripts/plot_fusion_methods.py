#!/usr/bin/env python3
"""Generate fusion method comparison charts and HTML report."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))

from fusion_method.viz import load_and_visualize  # noqa: E402

DEFAULT_REPORT_DIR = (
    REPO_ROOT
    / "data_analysis_scripts"
    / "bad_case_analysis"
    / "reports"
    / "trial_31"
    / "fusion"
)


def main() -> None:
    report_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REPORT_DIR
    print(f"Report dir: {report_dir}")
    paths = load_and_visualize(report_dir)
    print("可视化已生成：")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
