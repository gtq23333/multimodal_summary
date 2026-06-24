from __future__ import annotations

from pathlib import Path

import numpy as np

from m3sum.clients.dashscope_vl_rerank import (
    DEFAULT_INSTRUCT,
    DEFAULT_INSTRUCT_IMG_CAP,
    DEFAULT_INSTRUCT_IMG_CAP_LINK,
    DashScopeVLRerankClient,
    DocumentMode,
)
from m3sum.stage2_rerank.baselines.base import Stage2Sample, build_ranked_list
from m3sum.stage2_rerank.figure_link_context import FigureLinkContextSelector
from m3sum.stage2_rerank.vl_rerank_cache import VLRerankScoreCache


class _Qwen3VLRerankBase:
    document_mode: DocumentMode
    cache_subdir: str
    method_name: str

    def __init__(
        self,
        cache_root: Path,
        client: DashScopeVLRerankClient | None,
        dry_run: bool = False,
    ):
        self.cache = VLRerankScoreCache(
            cache_root / self.cache_subdir,
            client,
            document_mode=self.document_mode,
            dry_run=dry_run,
        )
        self._context_by_figure: dict[str, str] | None = None

    def rank(self, sample: Stage2Sample) -> list:
        if not sample.figures:
            return []

        per_query_scores = self.cache.load_or_compute(
            sample.paper_id,
            sample.sub_queries,
            sample.figures,
            context_by_figure=getattr(self, "_context_by_figure", None),
        )

        scored: list[tuple[str, float]] = []
        for fig in sample.figures:
            sims = [scores.get(fig.image_hash, 0.0) for scores in per_query_scores]
            mean_score = float(np.mean(sims)) if sims else 0.0
            scored.append((fig.image_hash, mean_score))

        return build_ranked_list(scored, self.method_name)


class Qwen3VLRerankImgRanker(_Qwen3VLRerankBase):
    """
    Qwen3-VL-Rerank 弱基线：仅传图片。
    候选集为正文带图注图片（body_with_caption 一致），不含无图注/非正文图。
    """

    method_name = "Qwen3-VL-Rerank-Img"
    document_mode = DocumentMode.IMAGE_ONLY
    cache_subdir = "img"


class Qwen3VLRerankImgCapRanker(_Qwen3VLRerankBase):
    """
    Qwen3-VL-Rerank 强基线：同时传入图片与匹配图注（text + image 双模态文档）。
    候选集同为正文带图注图片。
    """

    method_name = "Qwen3-VL-Rerank-ImgCap"
    document_mode = DocumentMode.IMAGE_CAPTION
    cache_subdir = "img_cap"


class Qwen3VLRerankImgCapLinkRanker(_Qwen3VLRerankBase):
    """
    Qwen3-VL-Rerank 更强基线：图片 + 图注 + LG-JSSF S_link 选出的最佳关联 chunk。

    关联 chunk 来自：图号显式引用、图注上下 local 窗口、caption_ref 块；
    与赛题 hybrid 召回块做加权 cosine，取全局 max 匹配块作为 Context。
    """

    method_name = "Qwen3-VL-Rerank-ImgCap+Link"
    document_mode = DocumentMode.IMAGE_CAPTION_CONTEXT
    cache_subdir = "img_cap_link"

    def __init__(
        self,
        cache_root: Path,
        client: DashScopeVLRerankClient | None,
        context_selector: FigureLinkContextSelector,
        dry_run: bool = False,
    ):
        super().__init__(cache_root, client, dry_run=dry_run)
        self.context_selector = context_selector

    def rank(self, sample: Stage2Sample) -> list:
        if not sample.figures:
            return []
        contexts, _ = self.context_selector.contexts_for_sample(sample)
        self._context_by_figure = contexts
        return super().rank(sample)


def build_vl_rerank_client(
    config,
    dry_run: bool,
    *,
    img_cap: bool = False,
    img_cap_link: bool = False,
) -> DashScopeVLRerankClient | None:
    if dry_run:
        return None
    s2ev = config.raw.get("stage2_eval", {})
    if img_cap_link:
        default_instruct = DEFAULT_INSTRUCT_IMG_CAP_LINK
        instruct_key = "vl_rerank_img_cap_link_instruct"
    elif img_cap:
        default_instruct = DEFAULT_INSTRUCT_IMG_CAP
        instruct_key = "vl_rerank_img_cap_instruct"
    else:
        default_instruct = DEFAULT_INSTRUCT
        instruct_key = "vl_rerank_instruct"
    instruct = str(s2ev.get(instruct_key, default_instruct))
    return DashScopeVLRerankClient(api_key=config.api_key, instruct=instruct)
