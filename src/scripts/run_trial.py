#!/usr/bin/env python3
"""
M3-Sum 统一试跑入口 — 所有选项在 configs/trial_10.yaml 中配置。

用法:
  cd src
  python scripts/run_trial.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

DEFAULT_CONFIG = SRC_ROOT / "configs" / "trial_20.yaml"


def main() -> None:
    config_path = DEFAULT_CONFIG
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        config_path = Path(sys.argv[1])

    from m3sum.config import load_config, resolve_api_credentials

    config = load_config(config_path)
    samples = config.resolved_sample_ids()

    print("=" * 60)
    print("M3-Sum Trial Runner")
    print("=" * 60)
    print(f"配置文件 : {config_path}")
    print(f"运行模式 : {config.mode}  |  阶段: {config.stage}  |  VLM: {config.vlm_mode}")
    print(f"样本数量 : {len(samples)}")
    if config.smoke_test:
        print(">>> 冒烟模式：仅跑 1 篇样本")
    if config.dry_run:
        print(">>> dry_run：零 API 成本，验证链路")
    else:
        creds = resolve_api_credentials(config)
        print(f">>> live：{creds.base_url}")
        print(f">>> 模型 LLM={config.llm_model}  Embed={config.embed_model}  VLM={config.vlm_model}")
    print("=" * 60)

    if config.auto_prepare_manifest:
        print("\n[1/4] 准备 manifest + GT ...")
        subprocess.run(
            [sys.executable, str(SRC_ROOT / "scripts" / "prepare_trial_manifest.py"), str(config_path)],
            check=True,
        )

    print(f"\n[2/4] 运行管线 stage={config.stage} ...")
    from m3sum.pipeline.runner import PipelineRunner

    runner = PipelineRunner(
        config,
        dry_run=config.dry_run,
        vlm_mode=config.vlm_mode,
        from_cache=config.use_cache,
    )
    runner.run(paper_ids=samples, stage=config.stage, force=config.force_rerun)
    print(f"      完成 {len(samples)} 篇")

    if config.auto_sanity_check:
        stage = config.resolved_sanity_stage()
        print(f"\n[3/4] 检验点 CP{stage} ...")
        result = subprocess.run(
            [sys.executable, str(SRC_ROOT / "scripts" / "sanity_check.py"), str(config_path)],
        )
        if result.returncode != 0:
            print("      警告: 部分检验点未通过，请查看上方输出")

    if config.auto_eval:
        print("\n[4/4] 评测 + 导出报告 ...")
        from m3sum.eval.report import run_evaluation

        if config.init_acceptance_csv:
            _init_acceptance(config)

        report = run_evaluation(config)
        _print_summary(report, config)

    print("\n完成。输出目录:", config.output_dir)
    if config.export_html_report:
        print("  HTML 报告:", config.eval_dir / "report.html")
    if config.export_csv_summary:
        print("  CSV 汇总 :", config.eval_dir / "summary.csv")
    print("  JSON 报告:", config.eval_dir / "report.json")


def _init_acceptance(config) -> None:
    import csv

    if config.acceptance_csv.is_file():
        return
    config.acceptance_csv.parent.mkdir(parents=True, exist_ok=True)
    with config.acceptance_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["paper_id", "reviewer", "accept", "notes"])
        for paper_id in config.sample_ids:
            writer.writerow([paper_id, "", "", ""])
    print(f"      已创建 acceptance 模板: {config.acceptance_csv}")


def _print_summary(report: dict, config) -> None:
    ret = report.get("retrieval", {})
    ins = report.get("insertion", {})
    gen = report.get("generation", {})
    print("      --- 汇总 ---")
    print(f"      检索 Hit@1={ret.get('Hit@1')}  Hit@3={ret.get('Hit@3')}  MRR={ret.get('MRR')}")
    print(f"      插入 F1={ins.get('F1')}  ROUGE-L={gen.get('ROUGE-L')}")


if __name__ == "__main__":
    main()
