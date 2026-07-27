from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from m3sum.config import PipelineConfig, load_config  # noqa: E402


DEFAULT_CONFIG = SRC_ROOT / "configs" / "trial_31.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs_copy" / "trial_31"
DEFAULT_REPORT_DIR = ANALYSIS_ROOT / "reports" / "trial_31"

RECALL_KS = [3, 4, 5, 6, 7]

PRIMARY_METHODS = [
    "Proposed",
    "Proposed-v2",
    "Qwen3-VL-Rerank-ImgCap+Link",
    "Qwen3-VL-Rerank-ImgCap",
    "Qwen3-VL-Rerank-Img",
    "Layout-Order",
    "Caption-BM25",
    "Caption-Dense-v4",
    "Zero-shot-CLIP",
]

FOCUS_PAIRS = [
    ("Proposed", "Qwen3-VL-Rerank-ImgCap+Link"),
    ("Proposed", "Proposed-v2"),
    ("Proposed", "Layout-Order"),
    ("Proposed-v2", "Qwen3-VL-Rerank-ImgCap+Link"),
]

INCREMENTAL_ABLATION_ORDER = [
    "DirectOnly",
    "Direct+Link",
    "Direct+Link+Layout",
    "Direct+Link+Layout+Type",
    "LG-JSSF",
    "LG-JSSF+ClusterAdd",
    "FullClusterAdd",
]

DROP_ONE_MODULES = {
    "w/o S_link (Add)": "S_link",
    "w/o P_layout (Add)": "P_layout",
    "w/o P_type (Add)": "P_type",
    "w/o ClusterPrior (Add)": "ClusterPrior",
    "w/o LocalWindow (Add)": "LocalWindow",
}


def load_pipeline_config(
    config_path: Path | None = None,
    *,
    output_dir: Path | None = None,
) -> PipelineConfig:
    path = (config_path or DEFAULT_CONFIG).resolve()
    config = load_config(path)
    if output_dir is not None:
        config = replace(config, output_dir=Path(output_dir).resolve())
    return config


def report_dir(trial_name: str = "trial_31") -> Path:
    return ANALYSIS_ROOT / "reports" / trial_name


def artifacts_dir(trial_name: str = "trial_31") -> Path:
    return report_dir(trial_name) / "artifacts"
