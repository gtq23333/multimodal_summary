#!/usr/bin/env python3
"""Evaluate Stage-3 artifacts with reference-based automatic metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from m3sum.config import load_config
from m3sum.eval.stage3_ref_based_eval import (
    evaluate_artifacts,
    export_ref_based_results,
    resolve_artifact_paths,
)

DEFAULT_CONFIG = SRC_ROOT / "configs" / "trial_31.yaml"


def _parse_csv(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def _parse_int_csv(raw: str | None) -> set[int] | None:
    values = _parse_csv(raw)
    return {int(v) for v in values} if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-3 ref-based 自动指标评估")
    parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG), help="配置文件路径")
    parser.add_argument("--manifest", default=None, help="Stage3 artifact manifest；默认读取 stage3_generation/manifest.json")
    parser.add_argument("--pool-sizes", default=None, help="候选池大小过滤，逗号分隔；默认读配置")
    parser.add_argument("--models", default=None, help="生成模型过滤，逗号分隔；默认读配置")
    parser.add_argument("--methods", default=None, help="方法过滤，逗号分隔；默认不过滤")
    parser.add_argument("--no-bertscore", action="store_true", help="跳过 BERTScore 计算（默认开启）")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    pool_sizes = _parse_int_csv(args.pool_sizes) or set(config.stage3_pool_sizes)
    models = _parse_csv(args.models) or set(config.stage3_generation_models)
    methods = _parse_csv(args.methods)
    manifest = Path(args.manifest).resolve() if args.manifest else None
    paths = resolve_artifact_paths(config, manifest)
    df = evaluate_artifacts(
        config,
        artifact_paths=paths,
        pool_sizes=pool_sizes,
        models=models,
        methods=methods,
        with_bertscore=not args.no_bertscore,
    )
    out = export_ref_based_results(df, config.eval_dir)
    print(f"Ref-based rows: {len(df)}")
    print(df.groupby(["method_name", "strategy"])["comprehensive_score"].mean().round(4).to_string())
    for name, path in out.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
