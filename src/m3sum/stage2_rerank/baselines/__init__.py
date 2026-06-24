from m3sum.stage2_rerank.baselines.base import RankedFigure, Stage2Ranker, Stage2Sample, build_ranked_list
from m3sum.stage2_rerank.baselines.caption_bm25 import CaptionBM25Ranker
from m3sum.stage2_rerank.baselines.caption_dense import CaptionDenseRanker
from m3sum.stage2_rerank.baselines.layout_order import LayoutOrderRanker
from m3sum.stage2_rerank.baselines.qwen3_vl_rerank import (
    Qwen3VLRerankImgCapLinkRanker,
    Qwen3VLRerankImgCapRanker,
    Qwen3VLRerankImgRanker,
)
from m3sum.stage2_rerank.baselines.zeroshot_clip import ZeroshotClipRanker

__all__ = [
    "RankedFigure",
    "Stage2Ranker",
    "Stage2Sample",
    "build_ranked_list",
    "CaptionBM25Ranker",
    "CaptionDenseRanker",
    "LayoutOrderRanker",
    "Qwen3VLRerankImgCapLinkRanker",
    "Qwen3VLRerankImgCapRanker",
    "Qwen3VLRerankImgRanker",
    "ZeroshotClipRanker",
]

def __getattr__(name: str):
    if name == "ProposedRanker":
        from m3sum.stage2_rerank.baselines.proposed import ProposedRanker
        return ProposedRanker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
