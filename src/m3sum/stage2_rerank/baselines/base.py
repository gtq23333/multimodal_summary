from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from m3sum.data.schema import FigureMeta, SubQuery


@dataclass
class Stage2Sample:
    """Stage-2 单篇文档评估样本。"""

    paper_id: str
    figures: list[FigureMeta]
    sub_queries: list[SubQuery]
    ground_truth_ids: set[str]


@dataclass
class RankedFigure:
    """统一排序输出项；figure_id 对应 image_hash。"""

    figure_id: str
    score: float
    rank: int
    method_name: str


class Stage2Ranker(Protocol):
    """Stage-2 重排序 baseline 统一接口。"""

    method_name: str

    def rank(self, sample: Stage2Sample) -> list[RankedFigure]: ...


def build_ranked_list(
    scored: list[tuple[str, float]],
    method_name: str,
) -> list[RankedFigure]:
    """按分数降序构建 RankedFigure 列表。"""
    sorted_items = sorted(scored, key=lambda x: x[1], reverse=True)
    return [
        RankedFigure(
            figure_id=fig_id,
            score=score,
            rank=i + 1,
            method_name=method_name,
        )
        for i, (fig_id, score) in enumerate(sorted_items)
    ]
