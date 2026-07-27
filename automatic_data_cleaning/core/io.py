from __future__ import annotations

from pathlib import Path

import yaml

from core.types import PipelineConfig


def load_pipeline_config(config_path: Path) -> PipelineConfig:
    base = config_path.parent.resolve()
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    input_dir = raw.get("input_dir", "")
    output_dir = raw.get("output_dir", "../usable_data/cleaned_excellent_paper_mds")
    reports_dir = raw.get("reports_dir", "./reports")

    def resolve(p: str) -> str:
        path = Path(p)
        if path.is_absolute():
            return str(path)
        return str((base / path).resolve())

    return PipelineConfig(
        input_dir=resolve(input_dir),
        output_dir=resolve(output_dir),
        profile=str(raw.get("profile", "national_competition")),
        separator=str(raw.get("separator", "##############")),
        reports_dir=resolve(reports_dir),
    )


def discover_samples(input_dir: Path) -> list[tuple[str, Path]]:
    """返回 (paper_id, full_md_path) 列表。"""
    if not input_dir.is_dir():
        return []

    samples: list[tuple[str, Path]] = []
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        full_md = child / "full.md"
        if full_md.is_file():
            samples.append((child.name, full_md))
    return samples


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
