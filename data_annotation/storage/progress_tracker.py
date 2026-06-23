from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SkippedEntry:
    paper_id: str
    reason: str


@dataclass
class ProgressTracker:
    progress_file: Path
    md_corpus_root: str
    total: int = 0
    completed: list[str] = field(default_factory=list)
    skipped: list[SkippedEntry] = field(default_factory=list)
    current_index: int = 0

    def save(self) -> None:
        data = {
            "md_corpus_root": self.md_corpus_root,
            "total": self.total,
            "completed": self.completed,
            "skipped": [{"paper_id": s.paper_id, "reason": s.reason} for s in self.skipped],
            "current_index": self.current_index,
        }
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.progress_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_or_create(cls, progress_file: Path, md_corpus_root: str, paper_ids: list[str]) -> ProgressTracker:
        if progress_file.is_file():
            raw = json.loads(progress_file.read_text(encoding="utf-8"))
            skipped = [SkippedEntry(**s) for s in raw.get("skipped", [])]
            tracker = cls(
                progress_file=progress_file,
                md_corpus_root=md_corpus_root,
                total=len(paper_ids),
                completed=list(raw.get("completed", [])),
                skipped=skipped,
                current_index=raw.get("current_index", 0),
            )
        else:
            tracker = cls(
                progress_file=progress_file,
                md_corpus_root=md_corpus_root,
                total=len(paper_ids),
            )
            tracker.current_index = tracker.find_first_unfinished(paper_ids)
            tracker.save()
        return tracker

    def find_first_unfinished(self, paper_ids: list[str]) -> int:
        completed_set = set(self.completed)
        skipped_set = {s.paper_id for s in self.skipped}
        for i, pid in enumerate(paper_ids):
            if pid not in completed_set and pid not in skipped_set:
                return i
        return len(paper_ids)

    def mark_completed(self, paper_id: str) -> None:
        if paper_id not in self.completed:
            self.completed.append(paper_id)
        self.save()

    def mark_skipped(self, paper_id: str, reason: str) -> None:
        self.skipped = [s for s in self.skipped if s.paper_id != paper_id]
        self.skipped.append(SkippedEntry(paper_id=paper_id, reason=reason))
        self.save()

    def set_current_index(self, index: int) -> None:
        self.current_index = index
        self.save()

    def annotation_path(self, output_dir: Path, paper_id: str) -> Path:
        return output_dir / f"{paper_id}.json"

    def has_annotation(self, output_dir: Path, paper_id: str) -> bool:
        return self.annotation_path(output_dir, paper_id).is_file()
