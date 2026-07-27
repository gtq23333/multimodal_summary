#!/usr/bin/env python3
"""Run configurable Stage-3 multimodal summary generation experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from m3sum.config import load_config
from m3sum.stage3_generation.experiment_runner import run_stage3_generation_experiments

DEFAULT_CONFIG = SRC_ROOT / "configs" / "trial_31.yaml"


def _parse_csv(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int_csv(raw: str | None) -> list[int] | None:
    values = _parse_csv(raw)
    return [int(v) for v in values] if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-3 多模态摘要生成实验")
    parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG), help="配置文件路径")
    parser.add_argument("--methods", help="仅运行指定 Stage2 方法，逗号分隔")
    parser.add_argument("--models", help="仅运行指定生成模型，逗号分隔")
    parser.add_argument("--pool-sizes", help="仅运行指定候选池大小，逗号分隔，如 3,6")
    parser.add_argument("--strategies", help="仅运行指定生成策略，逗号分隔")
    parser.add_argument("--paper-id", action="append", help="仅运行指定 paper_id，可重复传入")
    parser.add_argument("--force", action="store_true", help="覆盖已有生成结果")
    parser.add_argument("--skip-clip", action="store_true", help="跳过需要 CLIP 的候选构建")
    parser.add_argument(
        "--parallel-papers",
        type=int,
        default=None,
        help="并行处理的论文数；默认读取 stage3_generation.parallel_papers",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="不生成 Reference-Oracle 组",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    paths = run_stage3_generation_experiments(
        config,
        methods=_parse_csv(args.methods),
        models=_parse_csv(args.models),
        pool_sizes=_parse_int_csv(args.pool_sizes),
        strategies=_parse_csv(args.strategies),
        paper_ids=args.paper_id,
        force=args.force,
        skip_clip=args.skip_clip,
        include_reference=not args.no_reference,
        parallel_papers=args.parallel_papers,
    )
    print(f"Stage3 generation artifacts: {len(paths)}")
    for path in paths[:20]:
        print(path)
    if len(paths) > 20:
        print(f"... {len(paths) - 20} more")


if __name__ == "__main__":
    main()
