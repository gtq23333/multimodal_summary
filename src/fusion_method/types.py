from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_SOURCE_METHODS = [
    "Proposed",
    "Qwen3-VL-Rerank-ImgCap+Link",
    "Layout-Order",
]

DEFAULT_SOURCE_SUFFIX = "PQL"

DEFAULT_WEIGHTS = {
    "Proposed": 0.35,
    "Qwen3-VL-Rerank-ImgCap+Link": 0.45,
    "Layout-Order": 0.20,
}


@dataclass
class SourceRanking:
    method_name: str
    ranked_ids: list[str]
    score_by_id: dict[str, float] = field(default_factory=dict)

    def rank_of(self, figure_id: str) -> int | None:
        try:
            return self.ranked_ids.index(figure_id) + 1
        except ValueError:
            return None


@dataclass
class FusionInput:
    paper_id: str
    sources: list[SourceRanking]

    @property
    def source_names(self) -> list[str]:
        return [s.method_name for s in self.sources]

    def all_figure_ids(self) -> set[str]:
        ids: set[str] = set()
        for source in self.sources:
            ids.update(source.ranked_ids)
        return ids
