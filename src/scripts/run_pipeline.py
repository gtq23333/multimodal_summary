#!/usr/bin/env python3
"""Run M3-Sum pipeline — 选项见 configs/trial_10.yaml 的 run 段。"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from m3sum.config import load_config, resolve_api_credentials
from m3sum.pipeline.runner import PipelineRunner

DEFAULT_CONFIG = SRC_ROOT / "configs" / "trial_10.yaml"


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = load_config(config_path)

    if not config.dry_run:
        resolve_api_credentials(config)

    samples = config.resolved_sample_ids()
    runner = PipelineRunner(
        config,
        dry_run=config.dry_run,
        vlm_mode=config.vlm_mode,
        from_cache=config.use_cache,
    )
    runner.run(paper_ids=samples, stage=config.stage, force=config.force_rerun)
    print(f"Completed stage={config.stage} mode={config.mode} samples={len(samples)}")


if __name__ == "__main__":
    main()
