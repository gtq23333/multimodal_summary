from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from abstract_eval.likert_judge import LikertJudge
from m3sum.config import PipelineConfig, resolve_api_credentials

logger = logging.getLogger(__name__)

ZH_COLUMNS = {
    "paper_id": "论文ID",
    "eval_key": "评估键",
    "experiment_id": "实验ID",
    "method_name": "候选池方法",
    "pool_size": "候选池大小",
    "strategy": "生成策略",
    "model": "生成模型",
    "judge_model": "Judge模型",
    "cr": "CR 多模态易读性",
    "icn": "ICN 信息互补性",
    "ocdu": "OCDU 整体一致性",
    "overall": "总体均分",
    "inserted_count": "插图数量",
    "candidate_count": "候选图数量",
    "selected_hit_count": "选中GT数量",
}


def discover_stage3_artifacts(config: PipelineConfig) -> list[Path]:
    root = config.stage3_generation_dir
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.glob("*/*.json")
        if path.name != "manifest.json" and path.parent.name != "cache"
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_diagnostics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _merge_rows(existing: pd.DataFrame, new: pd.DataFrame, eval_keys: set[str]) -> pd.DataFrame:
    if existing.empty:
        return new
    if "eval_key" not in existing.columns:
        existing = existing.copy()
        existing["eval_key"] = existing["experiment_id"].astype(str) + "::" + existing["paper_id"].astype(str)
    kept = existing[~existing["eval_key"].isin(eval_keys)]
    return pd.concat([kept, new], ignore_index=True)


def _merge_diag(
    existing: list[dict[str, Any]],
    new: list[dict[str, Any]],
    eval_keys: set[str],
) -> list[dict[str, Any]]:
    kept = [
        item
        for item in existing
        if f"{item.get('experiment_id')}::{item.get('paper_id')}" not in eval_keys
    ]
    return kept + new


def artifact_to_row(
    artifact: dict[str, Any],
    judge_model: str,
    judge_result: dict[str, Any],
) -> dict[str, Any]:
    pool = artifact.get("candidate_pool", {})
    candidates = pool.get("candidates", [])
    gt = set(pool.get("ground_truth_ids", []))
    selected = artifact.get("selected_image_hashes") or artifact.get("inserted_figures") or []
    retrieval_metrics = pool.get("retrieval_metrics", {})
    row: dict[str, Any] = {
        "paper_id": artifact.get("paper_id", ""),
        "eval_key": f"{artifact.get('experiment_id', '')}::{artifact.get('paper_id', '')}",
        "experiment_id": artifact.get("experiment_id", ""),
        "method_name": artifact.get("method_name", ""),
        "pool_size": int(artifact.get("pool_size", 0)),
        "strategy": artifact.get("strategy", ""),
        "model": artifact.get("model", ""),
        "judge_model": judge_model,
        "cr": judge_result["cr"],
        "icn": judge_result["icn"],
        "ocdu": judge_result["ocdu"],
        "overall": judge_result["overall"],
        "inserted_count": len(selected),
        "candidate_count": len(candidates),
        "selected_hit_count": len(set(selected) & gt),
    }
    for key, value in retrieval_metrics.items():
        row[key] = value
    return row


def run_stage3_generation_eval(
    config: PipelineConfig,
    *,
    artifact_paths: list[Path] | None = None,
    merge_results: bool | None = None,
) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = artifact_paths or discover_stage3_artifacts(config)
    if not paths:
        logger.warning("未发现 Stage3 生成产物: %s", config.stage3_generation_dir)
        return pd.DataFrame()

    eval_cfg = config.stage3_eval_config
    if merge_results is None:
        merge_results = bool(eval_cfg.get("merge_results", True))

    judge_model = config.stage3_judge_model
    judge = LikertJudge(
        model=judge_model,
        credentials=resolve_api_credentials(config),
        cache_dir=config.stage3_eval_cache_dir / "likert_judge",
        dry_run=config.dry_run,
    )

    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for path in paths:
        artifact = _load_json(path)
        logger.info("评估 Stage3: %s", artifact.get("experiment_id", path.parent.name))
        judge_result = judge.judge(artifact)
        rows.append(artifact_to_row(artifact, judge_model, judge_result))
        diagnostics.append(
            {
                "artifact_path": str(path),
                "experiment_id": artifact.get("experiment_id"),
                "paper_id": artifact.get("paper_id"),
                "method_name": artifact.get("method_name"),
                "pool_size": artifact.get("pool_size"),
                "strategy": artifact.get("strategy"),
                "model": artifact.get("model"),
                "judge": judge_result,
                "selected_image_hashes": artifact.get("selected_image_hashes")
                or artifact.get("inserted_figures")
                or [],
                "candidate_pool": artifact.get("candidate_pool", {}),
            }
        )

    df = pd.DataFrame(rows)
    config.eval_dir.mkdir(parents=True, exist_ok=True)
    en_path = config.eval_dir / "stage3_generation_eval_results.csv"
    zh_path = config.eval_dir / "stage3_generation_eval_results_zh.csv"
    diag_path = config.eval_dir / "stage3_generation_diagnostics.jsonl"

    if merge_results:
        existing_df = pd.read_csv(en_path) if en_path.is_file() else pd.DataFrame()
        eval_keys = set(df["eval_key"]) if not df.empty else set()
        df = _merge_rows(existing_df, df, eval_keys)
        diagnostics = _merge_diag(_load_diagnostics(diag_path), diagnostics, eval_keys)

    df.to_csv(en_path, index=False, encoding="utf-8-sig")
    df.rename(columns={k: v for k, v in ZH_COLUMNS.items() if k in df.columns}).to_csv(
        zh_path,
        index=False,
        encoding="utf-8-sig",
    )
    with diag_path.open("w", encoding="utf-8") as f:
        for item in diagnostics:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("Stage3 评估完成: %s", en_path)
    try:
        from m3sum.eval.stage3_generation_viz import export_stage3_generation_visuals

        paths = export_stage3_generation_visuals(df, config.eval_dir)
        logger.info("Stage3 可视化报告: %s", paths.get("html_report"))
    except Exception as exc:
        logger.warning("Stage3 可视化生成失败（不影响 CSV）: %s", exc)
    return df
