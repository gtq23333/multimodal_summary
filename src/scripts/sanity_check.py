#!/usr/bin/env python3
"""Sanity checkpoints CP0–CP8 for M3-Sum trial."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from m3sum.config import PipelineConfig, load_config
from m3sum.data.corpus_adapter import CorpusAdapter


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    details: list[str] = field(default_factory=list)


def check_cp0(config: PipelineConfig, corpus: CorpusAdapter) -> list[CheckResult]:
    results: list[CheckResult] = []
    for paper_id in config.resolved_sample_ids():
        try:
            doc = corpus.load_document(paper_id)
            ok = len(doc.blocks) >= 20 and len(doc.figures) >= 3
            results.append(
                CheckResult(
                    "CP0",
                    ok,
                    f"{paper_id[:30]}... blocks={len(doc.blocks)} figures={len(doc.figures)}",
                )
            )
        except Exception as e:
            results.append(CheckResult("CP0", False, f"{paper_id}: {e}"))
    return results


def check_cp1(config: PipelineConfig, corpus: CorpusAdapter) -> list[CheckResult]:
    results: list[CheckResult] = []
    for paper_id in config.resolved_sample_ids():
        doc = corpus.load_document(paper_id)
        ok = len(doc.problem_text) > 50 and (
            "问题" in doc.problem_text or "问题1" in doc.problem_text
        )
        results.append(
            CheckResult("CP1", ok, f"{paper_id[:30]}... problem_len={len(doc.problem_text)}")
        )
    return results


def check_cp2(config: PipelineConfig, corpus: CorpusAdapter) -> list[CheckResult]:
    results: list[CheckResult] = []
    for paper_id in config.resolved_sample_ids():
        gt_path = config.ground_truth_dir / f"{paper_id}.json"
        if not gt_path.is_file():
            results.append(CheckResult("CP2", False, f"GT missing: {paper_id}"))
            continue

        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        doc = corpus.load_document(paper_id)
        hash_to_pos = {f.image_hash: f.pos for f in doc.figures}
        gold_hashes = gt["insertion_gt"]["selected_hashes"]
        missing = [h for h in gold_hashes if hash_to_pos.get(h, -1) < 0]
        ok = len(missing) == 0
        results.append(
            CheckResult(
                "CP2",
                ok,
                f"{paper_id[:30]}... missing_pos={len(missing)}",
                details=missing,
            )
        )
    return results


def check_cp3(config: PipelineConfig) -> list[CheckResult]:
    results: list[CheckResult] = []
    for paper_id in config.resolved_sample_ids():
        path = config.stage1_dir / f"{paper_id}.json"
        if not path.is_file():
            results.append(CheckResult("CP3", False, f"Stage1 missing: {paper_id}"))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        subs = data.get("sub_queries", [])
        ok = (
            len(subs) == 3
            and all(len(s.get("query", "")) >= 10 for s in subs)
            and all(s.get("keywords") for s in subs)
        )
        results.append(CheckResult("CP3", ok, f"{paper_id[:30]}... sub_queries={len(subs)}"))
    return results


def check_cp4(config: PipelineConfig) -> list[CheckResult]:
    results: list[CheckResult] = []
    for paper_id in config.resolved_sample_ids():
        path = config.stage2_dir / f"{paper_id}.json"
        if not path.is_file():
            results.append(CheckResult("CP4", False, f"Stage2 missing: {paper_id}"))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        cf_count = data.get("recall_debug", {}).get("cf_count", 0)
        ok = cf_count >= 1
        results.append(CheckResult("CP4", ok, f"{paper_id[:30]}... cf_count={cf_count}"))
    return results


def check_cp5(config: PipelineConfig) -> list[CheckResult]:
    results: list[CheckResult] = []
    for paper_id in config.resolved_sample_ids():
        gt_path = config.ground_truth_dir / f"{paper_id}.json"
        stage2_path = config.stage2_dir / f"{paper_id}.json"
        if not stage2_path.is_file() or not gt_path.is_file():
            results.append(CheckResult("CP5", False, f"Missing files: {paper_id}"))
            continue

        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
        gold = set(gt["retrieval_gt"]["relevant_figure_hashes"])
        pool = [f["image_hash"] for f in stage2.get("all_scores", [])][: config.top_p]
        ok = any(h in gold for h in pool) if gold else True
        results.append(
            CheckResult(
                "CP5",
                ok,
                f"{paper_id[:30]}... gt_in_top_p={ok}",
            )
        )
    return results


def check_cp6(config: PipelineConfig) -> list[CheckResult]:
    results: list[CheckResult] = []
    for paper_id in config.resolved_sample_ids():
        path = config.stage2_dir / f"{paper_id}.json"
        if not path.is_file():
            results.append(CheckResult("CP6", False, f"Stage2 missing: {paper_id}"))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        top3 = data.get("top3_figures", [])
        hashes = [t["image_hash"] for t in top3]
        unique = len(hashes) == len(set(hashes))
        scores = [t["score"] for t in top3]
        monotonic = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
        ok = unique and (monotonic or len(scores) <= 1)
        results.append(
            CheckResult("CP6", ok, f"{paper_id[:30]}... unique={unique} monotonic={monotonic}")
        )
    return results


def check_cp7(config: PipelineConfig) -> list[CheckResult]:
    results: list[CheckResult] = []
    for paper_id in config.resolved_sample_ids():
        path = config.stage3_dir / f"{paper_id}.json"
        if not path.is_file():
            results.append(CheckResult("CP7", False, f"Stage3 missing: {paper_id}"))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        descs = data.get("figure_descriptions", [])
        ok = all(
            len(d.get("description", "")) >= 5
            and "无法识别" not in d.get("description", "")
            for d in descs
        ) if descs else False
        results.append(CheckResult("CP7", ok, f"{paper_id[:30]}... descs={len(descs)}"))
    return results


def check_cp8(config: PipelineConfig) -> list[CheckResult]:
    results: list[CheckResult] = []
    for paper_id in config.resolved_sample_ids():
        path = config.stage3_dir / f"{paper_id}.json"
        if not path.is_file():
            results.append(CheckResult("CP8", False, f"Stage3 missing: {paper_id}"))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        inserted = data.get("inserted_figures", [])
        ok = len(inserted) <= 2
        results.append(
            CheckResult("CP8", ok, f"{paper_id[:30]}... inserted={len(inserted)}")
        )
    return results


STAGE_CHECKS = {
    0: lambda c, corpus: check_cp0(c, corpus) + check_cp1(c, corpus) + check_cp2(c, corpus),
    1: lambda c, corpus: check_cp3(c),
    2: lambda c, corpus: check_cp4(c) + check_cp5(c) + check_cp6(c),
    3: lambda c, corpus: check_cp7(c) + check_cp8(c),
}


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC_ROOT / "configs" / "trial_10.yaml"
    config = load_config(config_path)
    corpus = CorpusAdapter(config)

    stage = config.resolved_sanity_stage()
    all_results = STAGE_CHECKS[stage](config, corpus)
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)

    print(f"\n=== Sanity Check Stage {stage} ===")
    for r in all_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name}: {r.message}")
        for d in r.details[:5]:
            print(f"       - {d}")

    print(f"\nSummary: {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
