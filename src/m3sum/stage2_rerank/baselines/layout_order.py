from __future__ import annotations

from m3sum.stage2_rerank.baselines.base import Stage2Sample, build_ranked_list
from m3sum.stage2_rerank.figure_number import layout_sort_key


class LayoutOrderRanker:
    """
    Layout-Order / First-Occurrence baseline。
    按 figure number 升序排序；缺失时回退 body_order。
    """

    method_name = "Layout-Order"

    def rank(self, sample: Stage2Sample) -> list:
        sorted_figs = sorted(sample.figures, key=layout_sort_key)
        scored = [(-(i + 1), fig.image_hash) for i, fig in enumerate(sorted_figs)]
        scored = [(fig_id, float(score)) for score, fig_id in scored]
        return build_ranked_list(scored, self.method_name)
