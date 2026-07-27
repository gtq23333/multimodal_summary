from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import paths as _paths  # noqa: F401 — ensures src on sys.path

from m3sum.config import PipelineConfig


def load_diagnostics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_rankings_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_rankings_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_rankings_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_methods(path: Path) -> set[str]:
    return {r["method_name"] for r in load_rankings_jsonl(path)}


def rankings_to_frames(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """method -> paper_id -> record."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for rec in records:
        method = rec["method_name"]
        paper_id = rec["paper_id"]
        out.setdefault(method, {})[paper_id] = rec
    return out


def load_stage2_item_map(config: PipelineConfig, paper_id: str) -> dict[str, dict[str, Any]]:
    path = config.stage2_dir / f"{paper_id}.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["image_hash"]): item for item in data.get("all_scores", [])}


def load_ablation_method_names(config: PipelineConfig) -> list[str]:
    path = config.eval_dir / "stage2_ablation_results.csv"
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    return sorted(df["method_name"].unique())


def load_summary_csv(config: PipelineConfig) -> pd.DataFrame:
    path = config.eval_dir / "stage2_reranking_summary.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)
