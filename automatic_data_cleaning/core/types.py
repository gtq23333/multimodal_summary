from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SampleStatus(str, Enum):
    PROCESSED = "processed"
    SKIPPED_EXISTS = "skipped_exists"
    FAILED = "failed"
    DRY_RUN = "dry_run"


@dataclass
class CleanResult:
    success: bool
    content: str = ""
    abstract: str = ""
    body: str = ""
    reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleRecord:
    paper_id: str
    input_path: str
    output_path: str
    status: SampleStatus
    reason: str = ""
    abstract_len: int = 0
    body_len: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "status": self.status.value,
            "reason": self.reason,
            "abstract_len": self.abstract_len,
            "body_len": self.body_len,
        }


@dataclass
class PipelineConfig:
    input_dir: str
    output_dir: str
    profile: str
    separator: str = "##############"
    reports_dir: str = "./reports"
