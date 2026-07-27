from __future__ import annotations

from typing import Protocol

from core.types import CleanResult


class CleaningProfile(Protocol):
    name: str

    def clean(self, raw_md: str, *, paper_id: str, separator: str) -> CleanResult:
        """将 minerU full.md 清洗为目标 corpus 格式。"""
