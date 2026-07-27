from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from core.io import discover_samples, read_text, write_text
from core.types import PipelineConfig, SampleRecord, SampleStatus
from profiles.registry import get_profile


def run_pipeline(
    config: PipelineConfig,
    *,
    dry_run: bool = False,
) -> list[SampleRecord]:
    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    reports_dir = Path(config.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    profile = get_profile(config.profile)
    samples = discover_samples(input_dir)
    records: list[SampleRecord] = []

    for paper_id, full_md_path in samples:
        output_path = output_dir / f"{paper_id}.md"
        if output_path.is_file():
            records.append(
                SampleRecord(
                    paper_id=paper_id,
                    input_path=str(full_md_path),
                    output_path=str(output_path),
                    status=SampleStatus.SKIPPED_EXISTS,
                    reason="output_exists",
                )
            )
            continue

        raw = read_text(full_md_path)
        result = profile.clean(raw, paper_id=paper_id, separator=config.separator)

        if not result.success:
            records.append(
                SampleRecord(
                    paper_id=paper_id,
                    input_path=str(full_md_path),
                    output_path=str(output_path),
                    status=SampleStatus.FAILED,
                    reason=result.reason,
                    abstract_len=len(result.abstract),
                    body_len=len(result.body),
                )
            )
            continue

        if dry_run:
            records.append(
                SampleRecord(
                    paper_id=paper_id,
                    input_path=str(full_md_path),
                    output_path=str(output_path),
                    status=SampleStatus.DRY_RUN,
                    abstract_len=len(result.abstract),
                    body_len=len(result.body),
                )
            )
            continue

        write_text(output_path, result.content)
        records.append(
            SampleRecord(
                paper_id=paper_id,
                input_path=str(full_md_path),
                output_path=str(output_path),
                status=SampleStatus.PROCESSED,
                abstract_len=len(result.abstract),
                body_len=len(result.body),
            )
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = reports_dir / f"run_{stamp}.jsonl"
    with open(report_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    counts = {}
    for rec in records:
        counts[rec.status.value] = counts.get(rec.status.value, 0) + 1

    print(
        f"完成: processed={counts.get('processed', 0)} "
        f"skipped_exists={counts.get('skipped_exists', 0)} "
        f"failed={counts.get('failed', 0)} "
        f"dry_run={counts.get('dry_run', 0)} "
        f"(共扫描 {len(samples)} 个样本)"
    )
    print(f"报告: {report_path}")
    return records
