#!/usr/bin/env python3
"""Run offline evaluation — 选项见 configs/trial_10.yaml 的 eval 段。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from m3sum.config import load_config
from m3sum.eval.report import run_evaluation

DEFAULT_CONFIG = SRC_ROOT / "configs" / "trial_10.yaml"


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = load_config(config_path)

    if config.init_acceptance_csv:
        import csv

        if not config.acceptance_csv.is_file():
            config.acceptance_csv.parent.mkdir(parents=True, exist_ok=True)
            with config.acceptance_csv.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["paper_id", "reviewer", "accept", "notes"])
                for paper_id in config.sample_ids:
                    writer.writerow([paper_id, "", "", ""])

    report = run_evaluation(config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
