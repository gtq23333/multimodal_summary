#!/usr/bin/env python3
"""Export full Stage-2 rankings for bad-case complementarity analysis."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib import paths as _paths  # noqa: F401
from lib.io import (  # noqa: E402
    append_rankings_jsonl,
    load_existing_methods,
    load_rankings_jsonl,
    save_rankings_jsonl,
)
from lib.paths import DEFAULT_CONFIG, DEFAULT_OUTPUT_DIR, artifacts_dir, load_pipeline_config
from lib.rankings import export_all_rankings

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export full Stage-2 rankings cache")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trial", default="trial_31")
    parser.add_argument("--force", action="store_true", help="Re-export all methods")
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--no-ablation", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_pipeline_config(args.config, output_dir=args.output_dir)
    config = replace(config, mode="dry_run")

    out_path = artifacts_dir(args.trial) / "rankings.jsonl"
    existing_records = [] if args.force else load_rankings_jsonl(out_path)
    existing_methods = set() if args.force else load_existing_methods(out_path)

    new_records = export_all_rankings(
        config,
        include_ablation=not args.no_ablation,
        skip_clip=args.skip_clip,
        force=args.force,
        existing_methods=existing_methods,
    )

    if args.force:
        all_records = new_records
        save_rankings_jsonl(out_path, all_records)
    else:
        all_records = existing_records + new_records
        if new_records:
            append_rankings_jsonl(out_path, new_records)

    logger.info("Wrote %d new records; total methods=%d -> %s", len(new_records), len({r['method_name'] for r in all_records}), out_path)


if __name__ == "__main__":
    main()
