#!/usr/bin/env python3
"""对比 export bundle 的 Top-3 与 diagnostics.jsonl 是否一致。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CASE_STUDY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CASE_STUDY_ROOT.parent / "src"
sys.path.insert(0, str(SRC_ROOT))

from m3sum.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(CASE_STUDY_ROOT / "data"))
    parser.add_argument(
        "--trial-config",
        default=str(CASE_STUDY_ROOT.parent / "src" / "configs" / "trial_20.yaml"),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config = load_config(Path(args.trial_config))
    diag_path = config.eval_dir / "stage2_reranking_diagnostics.jsonl"

    diagnostics: dict[tuple[str, str], list[str]] = {}
    for line in diag_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        top3 = rec.get("top3_predicted", [])
        if top3 and isinstance(top3[0], dict):
            top3 = [x["figure_id"] for x in top3]
        diagnostics[(rec["paper_id"], rec["method_name"])] = list(top3)

    mismatches = 0
    checked = 0
    papers_dir = data_dir / "papers"
    for bundle_path in sorted(papers_dir.glob("*.json")):
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        paper_id = bundle["paper_id"]
        for method_name, block in bundle.get("methods", {}).items():
            key = (paper_id, method_name)
            if key not in diagnostics:
                continue
            expected = diagnostics[key]
            actual = [r["figure_id"] for r in block.get("ranked_top10", [])[:3]]
            checked += 1
            if actual != expected:
                mismatches += 1
                print(f"MISMATCH {paper_id} {method_name}")
                print(f"  diagnostics: {expected}")
                print(f"  bundle:      {actual}")

    print(f"checked={checked} mismatches={mismatches}")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
