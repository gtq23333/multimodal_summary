from __future__ import annotations

import re
from pathlib import Path


def parse_problem_key(paper_id: str) -> str:
    """2016_G_A028.pdf-uuid -> 2016_A"""
    m = re.match(r"(\d{4})_G_([A-D])", paper_id)
    if not m:
        raise ValueError(f"Cannot parse problem key from paper_id: {paper_id}")
    return f"{m.group(1)}_{m.group(2)}"


def resolve_problem_md(problem_mds_root: Path, paper_id: str) -> Path:
    key = parse_problem_key(paper_id)
    year, letter = key.split("_")
    pattern = f"{year}_{letter}.*.md"
    matches = sorted(problem_mds_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No problem MD for key {key} under {problem_mds_root}"
        )
    return matches[0]


def load_problem_text(problem_mds_root: Path, paper_id: str) -> tuple[str, Path]:
    path = resolve_problem_md(problem_mds_root, paper_id)
    return path.read_text(encoding="utf-8").strip(), path
