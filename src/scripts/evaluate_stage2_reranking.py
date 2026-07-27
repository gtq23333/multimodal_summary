#!/usr/bin/env python3
"""Stage-2 图片重排序 baselines 与扩展指标评估。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from m3sum.config import load_config
from m3sum.eval.stage2_reranking_eval import run_stage2_reranking_eval

DEFAULT_CONFIG = SRC_ROOT / "configs" / "trial_10.yaml"


def _parse_methods(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [m.strip() for m in raw.split(",") if m.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 reranking baselines 评估")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help="配置文件路径（默认 configs/trial_10.yaml）",
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="跳过 Zero-shot CLIP baseline 与 MaxSim@3（开发调试用）",
    )
    parser.add_argument(
        "--methods",
        metavar="NAME",
        help="仅评估指定方法，逗号分隔（须已在 config stage2_eval.methods 中声明）",
    )
    parser.add_argument(
        "--vl-rerank-only",
        action="store_true",
        help="仅重跑配置中的 Qwen3-VL-Rerank 系列方法，并与已有 eval 结果合并",
    )
    parser.add_argument(
        "--merge-results",
        action="store_true",
        default=None,
        help="与已有 stage2_reranking_eval_results.csv 合并（增量评估时默认开启）",
    )
    parser.add_argument(
        "--no-merge-results",
        action="store_true",
        help="不合并已有结果，CSV/diagnostics 仅保留本次运行的方法",
    )
    parser.add_argument(
        "--run-secondary",
        action="store_true",
        help="增量评估时也运行消融 / Legacy 对比 / Case Study 导出",
    )
    args = parser.parse_args()

    if args.methods and args.vl_rerank_only:
        parser.error("--methods 与 --vl-rerank-only 不能同时使用")

    merge_results: bool | None = None
    if args.no_merge_results:
        merge_results = False
    elif args.merge_results:
        merge_results = True

    config_path = Path(args.config)
    config = load_config(config_path)
    df = run_stage2_reranking_eval(
        config,
        skip_clip=args.skip_clip,
        methods=_parse_methods(args.methods),
        vl_rerank_only=args.vl_rerank_only,
        merge_results=merge_results,
        run_secondary=True if args.run_secondary else None,
    )
    if not df.empty:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
