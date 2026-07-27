from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fusion_method.types import FusionInput, SourceRanking

logger = logging.getLogger(__name__)


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


def rankings_index(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """method_name -> paper_id -> record."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for rec in records:
        method = rec["method_name"]
        paper_id = rec["paper_id"]
        out.setdefault(method, {})[paper_id] = rec
    return out


def build_fusion_input(
    paper_id: str,
    rankings_by_method: dict[str, dict[str, dict[str, Any]]],
    source_methods: list[str],
) -> FusionInput:
    sources: list[SourceRanking] = []
    for method_name in source_methods:
        rec = rankings_by_method.get(method_name, {}).get(paper_id)
        if rec is None:
            logger.warning("Missing ranking: method=%s paper_id=%s", method_name, paper_id)
            sources.append(SourceRanking(method_name=method_name, ranked_ids=[], score_by_id={}))
            continue
        sources.append(
            SourceRanking(
                method_name=method_name,
                ranked_ids=list(rec.get("ranked_ids", [])),
                score_by_id={
                    str(k): float(v) for k, v in (rec.get("score_by_id") or {}).items()
                },
            )
        )
    return FusionInput(paper_id=paper_id, sources=sources)
