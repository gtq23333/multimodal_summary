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
from m3sum.stage2_rerank.baselines.base import RankedFigure  # noqa: E402
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


def _proposed_debug_from_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "s_direct": item.get("s_direct"),
        "s_link": item.get("s_link"),
        "p_layout": item.get("p_layout"),
        "p_type": item.get("p_type"),
        "score_base": item.get("score_base"),
        "figure_index": item.get("figure_index"),
        "cluster_prior": (item.get("debug") or {}).get("cluster", {}).get(
            "cluster_prior"
        ),
    }


def _load_proposed_debug(stage2_path: Path) -> dict[str, dict[str, Any]]:
    if not stage2_path.is_file():
        return {}
    data = json.loads(stage2_path.read_text(encoding="utf-8"))
    debug_by_hash: dict[str, dict[str, Any]] = {}
    for fig in data.get("top3_figures", []):
        h = fig.get("image_hash")
        if h:
            debug_by_hash[h] = _proposed_debug_from_item(fig)
    for fig in data.get("all_scores", []):
        h = fig.get("image_hash")
        if h and h not in debug_by_hash:
            debug_by_hash[h] = _proposed_debug_from_item(fig)
    return debug_by_hash


def _load_diagnostics(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        key = (rec["paper_id"], rec["method_name"])
        out[key] = rec
    return out


def _load_proposed_ranked_from_stage2(
    stage2_path: Path,
    *,
    max_rank: int,
) -> list[RankedFigure]:
    if not stage2_path.is_file():
        return []
    data = json.loads(stage2_path.read_text(encoding="utf-8"))
    all_scores = data.get("all_scores", [])
    if not all_scores:
        top3 = data.get("top3_figures", [])
        all_scores = top3
    sorted_scores = sorted(all_scores, key=lambda x: x.get("score", 0), reverse=True)
    return [
        RankedFigure(
            figure_id=item["image_hash"],
            score=float(item.get("score", 0)),
            rank=i + 1,
            method_name="Proposed",
        )
        for i, item in enumerate(sorted_scores[:max_rank])
        if item.get("image_hash")
    ]


def _load_baseline_ranked_from_diagnostics(
    top3_ids: list[str],
) -> list[RankedFigure]:
    return [
        RankedFigure(
            figure_id=figure_id,
            score=float("nan"),
            rank=i + 1,
            method_name="",
        )
        for i, figure_id in enumerate(top3_ids[:3])
    ]


def _serialize_ranked(
    ranked: list[RankedFigure],
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
        score = item.score
        row: dict[str, Any] = {
            "rank": item.rank,
            "figure_id": item.figure_id,
            "score": None if score != score else round(float(score), 6),
            "caption": caption,
            "image_path": image_path,
            "is_gt": item.figure_id in gold,
        }
        if proposed_debug and item.figure_id in proposed_debug:
            row["debug"] = proposed_debug[item.figure_id]
        rows.append(row)
    return rows


def _metrics_from_df(
    metrics_df: pd.DataFrame,
    paper_id: str,
    method_name: str,
) -> dict[str, float | None]:
    if metrics_df.empty:
        return {}
    sub = metrics_df[
        (metrics_df["paper_id"] == paper_id)
        & (metrics_df["method_name"] == method_name)
    ]
    if sub.empty:
        return {}
    row = sub.iloc[0]
    return {
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


def _export_from_eval(
    *,
    samples,
    pipeline_config,
    methods: list[str],
    papers_dir: Path,
    max_rank: int,
    metrics_df: pd.DataFrame,
    diagnostics: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """从 eval 已有产物组装 bundle，不重新跑 ranker（无需 CLIP / 网络）。"""
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

        stage2_path = pipeline_config.stage2_dir / f"{sample.paper_id}.json"
        proposed_debug = _load_proposed_debug(stage2_path)
        proposed_ranked = _load_proposed_ranked_from_stage2(
            stage2_path,
            max_rank=max_rank,
        )

        hash_to_path = {f.image_hash: f.abs_image_path for f in sample.figures}
        enriched_sequence = _enrich_multimodal_sequence(multimodal_sequence, hash_to_path)

        method_data: dict[str, Any] = {}
        for method_name in methods:
            if method_name == "Proposed":
                ranked = proposed_ranked
                debug_map = proposed_debug
            else:
                diag = diagnostics.get((sample.paper_id, method_name), {})
                top3 = diag.get("top3_predicted", [])
                if top3 and isinstance(top3[0], dict):
                    top3 = [x["figure_id"] for x in top3]
                ranked = _load_baseline_ranked_from_diagnostics(list(top3))
                debug_map = None

            method_data[method_name] = {
                "metrics": _metrics_from_df(metrics_df, sample.paper_id, method_name),
                "flags": diagnostics.get((sample.paper_id, method_name), {}).get(
                    "flags", []
                ),
                "ranked_top10": _serialize_ranked(
                    ranked,
                    sample,
                    max_rank=max_rank,
                    proposed_debug=debug_map,
                ),
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
            "export_mode": "from_eval",
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
        logger.info("已导出(from_eval): %s", out_path.name)

    return manifest_papers


def _export_from_rerank(
    *,
    samples,
    pipeline_config,
    methods: list[str],
    papers_dir: Path,
    max_rank: int,
    skip_clip: bool,
    metrics_df: pd.DataFrame,
    diagnostics: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """重放 ranker 得到完整 Top-K（baseline 可 K>3；需本地缓存，建议 skip_clip=true）。"""
    rankers, _, _ = build_stage2_rankers(pipeline_config, skip_clip=skip_clip)
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
                logger.warning(
                    "跳过方法 %s（ranker 未构建，可能因 skip_clip=true）",
                    method_name,
                )
                continue
            ranked = ranker.rank(sample)
            debug_map = proposed_debug if method_name == "Proposed" else None
            method_data[method_name] = {
                "metrics": _metrics_from_df(metrics_df, sample.paper_id, method_name),
                "flags": diagnostics.get((sample.paper_id, method_name), {}).get(
                    "flags", []
                ),
                "ranked_top10": _serialize_ranked(
                    ranked,
                    sample,
                    max_rank=max_rank,
                    proposed_debug=debug_map,
                ),
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
            "export_mode": "rerank",
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
        logger.info("已导出(rerank): %s", out_path.name)

    return manifest_papers


def export_case_study_data(
    case_study_config_path: Path,
    *,
    trial_config_override: Path | None = None,
    mode: str | None = None,
    skip_clip: bool | None = None,
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

    export_cfg = cs_cfg.get("export", {})
    max_rank = int(export_cfg.get("max_rank", 10))
    export_mode = mode or export_cfg.get("mode", "from_eval")
    if skip_clip is None:
        skip_clip = bool(export_cfg.get("skip_clip", True))

    samples = load_all_stage2_samples(pipeline_config)
    if not samples:
        raise RuntimeError("无有效 Stage-2 样本，无法导出 Case Study 数据")

    methods = [m for m in METHOD_ORDER if m in pipeline_config.stage2_eval_methods]

    eval_csv = pipeline_config.eval_dir / "stage2_reranking_eval_results.csv"
    metrics_df = pd.read_csv(eval_csv) if eval_csv.is_file() else pd.DataFrame()
    if metrics_df.empty:
        logger.warning("未找到 eval CSV: %s", eval_csv)

    diagnostics = _load_diagnostics(
        pipeline_config.eval_dir / "stage2_reranking_diagnostics.jsonl"
    )

    if export_mode == "from_eval":
        logger.info(
            "export 模式=from_eval：读取 eval 产物，不重新跑 ranker（baseline 仅 Top-3）"
        )
        manifest_papers = _export_from_eval(
            samples=samples,
            pipeline_config=pipeline_config,
            methods=methods,
            papers_dir=papers_dir,
            max_rank=max_rank,
            metrics_df=metrics_df,
            diagnostics=diagnostics,
        )
    elif export_mode == "rerank":
        logger.info(
            "export 模式=rerank：重放 ranker 导出 Top-%d（skip_clip=%s）",
            max_rank,
            skip_clip,
        )
        manifest_papers = _export_from_rerank(
            samples=samples,
            pipeline_config=pipeline_config,
            methods=methods,
            papers_dir=papers_dir,
            max_rank=max_rank,
            skip_clip=skip_clip,
            metrics_df=metrics_df,
            diagnostics=diagnostics,
        )
    else:
        raise ValueError(f"未知 export.mode: {export_mode}，应为 from_eval 或 rerank")

    manifest = {
        "trial_id": trial_cfg_path.stem,
        "export_time": datetime.now(timezone.utc).isoformat(),
        "config_path": str(trial_cfg_path),
        "export_mode": export_mode,
        "methods": methods,
        "max_rank": max_rank,
        "baseline_rank_depth": 3 if export_mode == "from_eval" else max_rank,
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
    parser.add_argument(
        "--mode",
        choices=("from_eval", "rerank"),
        default=None,
        help="from_eval=读 eval 产物（默认）；rerank=重跑 ranker 得完整 Top-K",
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="rerank 模式下跳过 CLIP 加载（避免 HuggingFace 下载）",
    )
    parser.add_argument(
        "--with-clip",
        action="store_true",
        help="rerank 模式下强制加载 CLIP（含 Zero-shot-CLIP）",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    skip_clip: bool | None = None
    if args.skip_clip:
        skip_clip = True
    elif args.with_clip:
        skip_clip = False

    trial_override = Path(args.trial_config) if args.trial_config else None
    export_case_study_data(
        Path(args.config).resolve(),
        trial_config_override=trial_override,
        mode=args.mode,
        skip_clip=skip_clip,
    )


if __name__ == "__main__":
    main()
