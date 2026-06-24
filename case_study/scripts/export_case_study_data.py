#!/usr/bin/env python3
"""导出 Case Study UI 所需的 paper bundle 与 manifest。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

CASE_STUDY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CASE_STUDY_ROOT.parent / "src"
sys.path.insert(0, str(SRC_ROOT))

from m3sum.config import load_config  # noqa: E402
from m3sum.eval.stage2_reranking_eval import build_stage2_rankers  # noqa: E402
from m3sum.eval.stage2_reranking_viz import METHOD_ORDER  # noqa: E402
from m3sum.stage2_rerank.sample_loader import load_all_stage2_samples  # noqa: E402

logger = logging.getLogger(__name__)


def _short_paper_id(paper_id: str) -> str:
    return paper_id.split(".pdf")[0] if ".pdf" in paper_id else paper_id[:24]


def _load_case_study_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_image_path(item: dict[str, Any], hash_to_path: dict[str, str]) -> str:
    if item.get("type") != "image":
        return ""
    h = item.get("image_hash", "")
    if h in hash_to_path:
        return hash_to_path[h]
    return item.get("image_path") or item.get("img_path") or ""


def _enrich_multimodal_sequence(
    sequence: list[dict[str, Any]],
    hash_to_path: dict[str, str],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for item in sequence:
        row = dict(item)
        if row.get("type") == "image":
            row["image_path"] = _resolve_image_path(row, hash_to_path)
        enriched.append(row)
    return enriched


def _load_proposed_debug(stage2_path: Path) -> dict[str, dict[str, Any]]:
    if not stage2_path.is_file():
        return {}
    data = json.loads(stage2_path.read_text(encoding="utf-8"))
    debug_by_hash: dict[str, dict[str, Any]] = {}
    for fig in data.get("top3_figures", []):
        h = fig.get("image_hash")
        if not h:
            continue
        debug_by_hash[h] = {
            "s_direct": fig.get("s_direct"),
            "s_link": fig.get("s_link"),
            "p_layout": fig.get("p_layout"),
            "p_type": fig.get("p_type"),
            "score_base": fig.get("score_base"),
            "figure_index": fig.get("figure_index"),
            "cluster_prior": (fig.get("debug") or {}).get("cluster", {}).get(
                "cluster_prior"
            ),
        }
    return debug_by_hash


def _load_diagnostics_flags(path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    if not path.is_file():
        return {}
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        key = (rec["paper_id"], rec["method_name"])
        out[key] = rec.get("flags") or []
    return out


def _serialize_ranked(
    ranked,
    sample,
    *,
    max_rank: int,
    proposed_debug: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    hash_to_fig = {f.image_hash: f for f in sample.figures}
    gold = sample.ground_truth_ids
    rows: list[dict[str, Any]] = []
    for item in ranked[:max_rank]:
        fig = hash_to_fig.get(item.figure_id)
        caption = fig.caption if fig else ""
        image_path = fig.abs_image_path if fig else ""
        row: dict[str, Any] = {
            "rank": item.rank,
            "figure_id": item.figure_id,
            "score": round(float(item.score), 6),
            "caption": caption,
            "image_path": image_path,
            "is_gt": item.figure_id in gold,
        }
        if proposed_debug and item.figure_id in proposed_debug:
            row["debug"] = proposed_debug[item.figure_id]
        rows.append(row)
    return rows


def export_case_study_data(
    case_study_config_path: Path,
    *,
    trial_config_override: Path | None = None,
) -> Path:
    cs_cfg = _load_case_study_config(case_study_config_path)
    trial_cfg_path = trial_config_override or (
        case_study_config_path.parent / cs_cfg.get("trial_config", "../src/configs/trial_20.yaml")
    )
    trial_cfg_path = trial_cfg_path.resolve()
    pipeline_config = load_config(trial_cfg_path)

    data_dir = (case_study_config_path.parent / cs_cfg.get("data_dir", "./data")).resolve()
    papers_dir = data_dir / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    max_rank = int(cs_cfg.get("export", {}).get("max_rank", 10))
    skip_clip = bool(cs_cfg.get("export", {}).get("skip_clip", False))

    samples = load_all_stage2_samples(pipeline_config)
    if not samples:
        raise RuntimeError("无有效 Stage-2 样本，无法导出 Case Study 数据")

    rankers, _, _ = build_stage2_rankers(pipeline_config, skip_clip=skip_clip)
    methods = [m for m in METHOD_ORDER if m in pipeline_config.stage2_eval_methods]

    eval_csv = pipeline_config.eval_dir / "stage2_reranking_eval_results.csv"
    metrics_df = pd.read_csv(eval_csv) if eval_csv.is_file() else pd.DataFrame()

    diag_flags = _load_diagnostics_flags(
        pipeline_config.eval_dir / "stage2_reranking_diagnostics.jsonl"
    )

    manifest_papers: list[dict[str, Any]] = []

    for sample in samples:
        gt_path = pipeline_config.ground_truth_dir / f"{sample.paper_id}.json"
        gt = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.is_file() else {}
        insertion_gt = gt.get("insertion_gt", {})
        multimodal_sequence = insertion_gt.get("multimodal_sequence", [])

        stage1_path = pipeline_config.stage1_dir / f"{sample.paper_id}.json"
        stage1 = (
            json.loads(stage1_path.read_text(encoding="utf-8"))
            if stage1_path.is_file()
            else {}
        )

        hash_to_path = {f.image_hash: f.abs_image_path for f in sample.figures}
        enriched_sequence = _enrich_multimodal_sequence(multimodal_sequence, hash_to_path)

        proposed_debug = _load_proposed_debug(
            pipeline_config.stage2_dir / f"{sample.paper_id}.json"
        )

        method_data: dict[str, Any] = {}
        for method_name in methods:
            ranker = rankers.get(method_name)
            if ranker is None:
                continue
            ranked = ranker.rank(sample)
            debug_map = proposed_debug if method_name == "Proposed" else None
            method_data[method_name] = {
                "metrics": {},
                "flags": diag_flags.get((sample.paper_id, method_name), []),
                "ranked_top10": _serialize_ranked(
                    ranked,
                    sample,
                    max_rank=max_rank,
                    proposed_debug=debug_map,
                ),
            }
            if not metrics_df.empty:
                sub = metrics_df[
                    (metrics_df["paper_id"] == sample.paper_id)
                    & (metrics_df["method_name"] == method_name)
                ]
                if not sub.empty:
                    row = sub.iloc[0]
                    method_data[method_name]["metrics"] = {
                        k: (None if pd.isna(row[k]) else float(row[k]))
                        for k in [
                            "r_precision",
                            "ip@3",
                            "ir@3",
                            "jaccard@3",
                            "maxsim@3",
                            "map",
                            "mrr",
                        ]
                        if k in row
                    }

        bundle = {
            "paper_id": sample.paper_id,
            "short_id": _short_paper_id(sample.paper_id),
            "n_candidates": len(sample.figures),
            "n_ground_truth": len(sample.ground_truth_ids),
            "sub_queries": stage1.get("sub_queries", []),
            "ground_truth_hashes": list(sample.ground_truth_ids),
            "multimodal_sequence": enriched_sequence,
            "methods": method_data,
        }

        out_path = papers_dir / f"{sample.paper_id}.json"
        out_path.write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        proposed_metrics = method_data.get("Proposed", {}).get("metrics", {})
        manifest_papers.append(
            {
                "paper_id": sample.paper_id,
                "short_id": bundle["short_id"],
                "n_ground_truth": bundle["n_ground_truth"],
                "proposed_jaccard@3": proposed_metrics.get("jaccard@3"),
                "proposed_ip@3": proposed_metrics.get("ip@3"),
            }
        )

        logger.info("已导出: %s", out_path.name)

    manifest = {
        "trial_id": trial_cfg_path.stem,
        "export_time": datetime.now(timezone.utc).isoformat(),
        "config_path": str(trial_cfg_path),
        "methods": methods,
        "max_rank": max_rank,
        "papers": sorted(manifest_papers, key=lambda p: p["short_id"]),
    }
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("manifest: %s (%d papers)", manifest_path, len(manifest_papers))
    return data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 Case Study 数据 bundle")
    parser.add_argument(
        "--config",
        default=str(CASE_STUDY_ROOT / "config.yaml"),
        help="case_study/config.yaml 路径",
    )
    parser.add_argument(
        "--trial-config",
        default=None,
        help="覆盖 trial 配置路径（默认读 case_study config 中的 trial_config）",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    trial_override = Path(args.trial_config) if args.trial_config else None
    export_case_study_data(
        Path(args.config).resolve(),
        trial_config_override=trial_override,
    )


if __name__ == "__main__":
    main()
