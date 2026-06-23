#!/usr/bin/env python3
"""Generate trial_10 manifest and ground truth from annotation JSON files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from m3sum.config import load_config
from m3sum.data.gt_loader import build_ground_truth_from_annotation, save_ground_truth
from m3sum.data.problem_resolver import resolve_problem_md


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC_ROOT / "configs" / "trial_10.yaml"
    config = load_config(config_path)
    repo_root = config.root
    annotation_dir = config.annotation_config.parent / "annotations"

    samples = []
    for paper_id in config.sample_ids:
        ann_path = annotation_dir / f"{paper_id}.json"
        if not ann_path.is_file():
            raise FileNotFoundError(f"Annotation not found: {ann_path}")

        problem_path = resolve_problem_md(config.problem_mds_root, paper_id)
        annotation = json.loads(ann_path.read_text(encoding="utf-8"))

        gt = build_ground_truth_from_annotation(
            paper_id, annotation, problem_path, ann_path
        )
        gt_path = config.ground_truth_dir / f"{paper_id}.json"
        save_ground_truth(gt_path, gt)

        samples.append(
            {
                "paper_id": paper_id,
                "problem_key": _problem_key(paper_id),
                "annotation_path": str(ann_path.relative_to(repo_root)),
                "problem_md_path": str(problem_path.relative_to(repo_root)),
                "ground_truth_path": str(gt_path.relative_to(repo_root)),
            }
        )

    trial_name = config.config_path.stem
    manifest = {
        "trial_name": trial_name,
        "gt_mode": config.gt_mode,
        "sample_count": len(samples),
        "samples": samples,
    }

    config.manifest.parent.mkdir(parents=True, exist_ok=True)
    config.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote manifest: {config.manifest}")
    print(f"Wrote {len(samples)} ground truth files under {config.ground_truth_dir}")


def _problem_key(paper_id: str) -> str:
    import re

    m = re.match(r"(\d{4})_G_([A-D])", paper_id)
    if not m:
        raise ValueError(paper_id)
    return f"{m.group(1)}_{m.group(2)}"


if __name__ == "__main__":
    main()
