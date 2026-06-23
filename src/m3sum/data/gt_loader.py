from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_ground_truth(gt_path: Path) -> dict[str, Any]:
    return json.loads(gt_path.read_text(encoding="utf-8"))


def build_ground_truth_from_annotation(
    paper_id: str,
    annotation: dict[str, Any],
    problem_md_path: Path,
    annotation_path: Path,
) -> dict[str, Any]:
    hashes = [ins["image_hash"] for ins in annotation.get("insertions", [])]
    return {
        "schema_version": "0.1.0",
        "paper_id": paper_id,
        "sources": {
            "annotation_path": str(annotation_path),
            "problem_md_path": str(problem_md_path),
        },
        "retrieval_gt": {
            "relevant_figure_hashes": hashes,
            "note": "暂等于 insertions，后续可扩展为 2-8 张",
        },
        "insertion_gt": {
            "selected_hashes": hashes,
            "reference_text": annotation.get("abstract", {}).get("edited_text", ""),
            "multimodal_sequence": annotation.get("multimodal_sequence", []),
        },
    }


def save_ground_truth(gt_path: Path, data: dict[str, Any]) -> None:
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
