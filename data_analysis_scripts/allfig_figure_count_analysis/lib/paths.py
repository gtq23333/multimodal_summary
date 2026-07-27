from __future__ import annotations

from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = SRC_ROOT / "configs" / "trial_31.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "trial_31"
DEFAULT_REPORT_DIR = ANALYSIS_ROOT / "reports" / "trial_31"

ALLFIG_GROUPS = {
    "end_to_end_vlm": "AllFig-E2E",
    "text_rag_then_rewrite": "AllFig-RAG",
}

PRERECALL_GROUPS = {
    "end_to_end_vlm": ["Prop-E2E-T6", "Layout-E2E-T6", "QwenVL-E2E-T6"],
    "text_rag_then_rewrite": ["Prop-RAG-T6", "Layout-RAG-T6", "QwenVL-RAG-T6"],
}

PRERECALL_LABELS = {
    "Prop-E2E-T6": "Proposed",
    "Layout-E2E-T6": "Layout",
    "QwenVL-E2E-T6": "QwenVL",
    "Prop-RAG-T6": "Proposed",
    "Layout-RAG-T6": "Layout",
    "QwenVL-RAG-T6": "QwenVL",
}

FIGURE_COUNT_BINS = [0, 10, 15, 20, 30, 100]
FIGURE_COUNT_BIN_LABELS = ["≤10", "11–15", "16–20", "21–30", "31+"]

SCORE_COLS = [
    "comprehensive_score",
    "image_f1",
    "rouge_l",
    "bertscore_f1",
]


def report_dir(trial_name: str = "trial_31") -> Path:
    return ANALYSIS_ROOT / "reports" / trial_name


def eval_csv_path(output_dir: Path) -> Path:
    return output_dir / "eval" / "stage3_ref_based_eval_results.csv"
