"""Case Study UI 数据加载。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CaseStudyConfig:
    trial_config: Path
    data_dir: Path
    default_k: int = 3
    k_options: list[int] = field(default_factory=lambda: [3, 5, 10])
    default_method: str = "Proposed"


@dataclass
class Manifest:
    trial_id: str
    export_time: str
    config_path: str
    methods: list[str]
    max_rank: int
    papers: list[dict[str, Any]]


def load_case_study_config(config_path: Path) -> CaseStudyConfig:
    base = config_path.parent.resolve()
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    ui = raw.get("ui", {})
    return CaseStudyConfig(
        trial_config=(base / raw.get("trial_config", "../src/configs/trial_20.yaml")).resolve(),
        data_dir=(base / raw.get("data_dir", "./data")).resolve(),
        default_k=int(ui.get("default_k", 3)),
        k_options=[int(k) for k in ui.get("k_options", [3, 5, 10])],
        default_method=str(ui.get("default_method", "Proposed")),
    )


def load_manifest(data_dir: Path) -> Manifest:
    path = data_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"未找到 manifest: {path}，请先运行 export 脚本")
    data = json.loads(path.read_text(encoding="utf-8"))
    return Manifest(
        trial_id=data.get("trial_id", ""),
        export_time=data.get("export_time", ""),
        config_path=data.get("config_path", ""),
        methods=list(data.get("methods", [])),
        max_rank=int(data.get("max_rank", 10)),
        papers=list(data.get("papers", [])),
    )


def load_paper_bundle(data_dir: Path, paper_id: str) -> dict[str, Any]:
    path = data_dir / "papers" / f"{paper_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"未找到 paper bundle: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
