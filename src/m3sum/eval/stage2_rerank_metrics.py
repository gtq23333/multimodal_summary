from __future__ import annotations

import numpy as np

from m3sum.data.schema import FigureMeta
from m3sum.eval.retrieval_metrics import mrr
from m3sum.stage2_rerank.clip_utils import ClipImageEmbeddingCache, ChineseCLIPWrapper
from m3sum.stage2_rerank.co_occurrence import cosine_sim


def r_precision(ranked_ids: list[str], gold: set[str]) -> float:
    """
    R-Precision：取 Top-R（R=|G|）预测与 G 的交集比例。
    G 为空时调用方应 skip，此处返回 0.0。
    """
    if not gold:
        return 0.0
    r = len(gold)
    top_r = set(ranked_ids[:r])
    return len(top_r & gold) / r


def jaccard_at_k(ranked_ids: list[str], gold: set[str], k: int = 3) -> float:
    """
    Jaccard@K：Top-K 预测集合与 G 的 Jaccard 相似度。
    P∪G 为空时返回 0.0。
    """
    pred_set = set(ranked_ids[:k])
    union = pred_set | gold
    if not union:
        return 0.0
    return len(pred_set & gold) / len(union)


def image_precision_at_k(ranked_ids: list[str], gold: set[str], k: int = 3) -> float:
    """
    Image Precision (IP@K) — MSMO 多模态摘要标准图像指标 (Zhu et al., 2018)。

    IP = |rec_img ∩ ref_img| / |rec_img|

    - rec_img：系统推荐的 Top-K 图片（本任务固定为 Top-3）
    - ref_img：人工标注参考图片集合（ground truth）

    在固定 K 槽位推荐下等价于 Precision@K；分母为推荐数 K（非 |GT|）。
    K ≤ 0 或无推荐时返回 0.0。
    """
    if k <= 0:
        return 0.0
    rec = ranked_ids[:k]
    if not rec:
        return 0.0
    return len(set(rec) & gold) / k


def image_recall_at_ks(
    ranked_ids: list[str],
    gold: set[str],
    ks: list[int],
) -> dict[int, float]:
    """批量计算 IR@K，返回 {k: score}。"""
    return {k: image_recall_at_k(ranked_ids, gold, k=k) for k in ks}


def image_recall_at_k(ranked_ids: list[str], gold: set[str], k: int = 3) -> float:
    """
    Image Recall (IR@K) — MSMO / MMAE 标准图像指标，与 IP 成对使用。

    IR = |rec_img ∩ ref_img| / |ref_img|

    衡量参考 GT 图片中有多少比例被 Top-K 短名单覆盖。
    Stage-2 作为「召回+重排」时，IR@K 表示送入 Stage-3 VLM/LLM 确认前的 GT 覆盖率；
    K 固定为 3 时，|GT|>K 的论文 IR 上限为 K/|GT|。

    G 为空时返回 0.0。
    """
    if not gold:
        return 0.0
    if k <= 0 or not ranked_ids:
        return 0.0
    rec = set(ranked_ids[:k])
    return len(rec & gold) / len(gold)


def average_precision(ranked_ids: list[str], gold: set[str]) -> float:
    """
    Average Precision（单篇 AP）。
    G 为空时返回 0.0。
    """
    if not gold:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, figure_id in enumerate(ranked_ids, start=1):
        if figure_id in gold:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(gold)


def compute_mrr(ranked_ids: list[str], gold: set[str]) -> float:
    """MRR：第一个 relevant item 的 reciprocal rank；无命中返回 0.0。"""
    return mrr(ranked_ids, gold)


def maxsim_at_k(
    ranked_ids: list[str],
    gold: set[str],
    figures: list[FigureMeta],
    clip_cache: ClipImageEmbeddingCache,
    paper_id: str,
    k: int = 3,
) -> float:
    """
    MaxSim@K：Top-K 预测每张图与 G 中任意图的最大 CLIP vision cosine 相似度均值。
    P 或 G 为空时返回 0.0。
    """
    if not gold or not ranked_ids:
        return 0.0

    pred_ids = ranked_ids[:k]
    if not pred_ids:
        return 0.0

    hash_to_fig = {f.image_hash: f for f in figures}
    needed_hashes = set(pred_ids) | gold
    needed_figs = [hash_to_fig[h] for h in needed_hashes if h in hash_to_fig]
    if not needed_figs:
        return 0.0

    image_embs = clip_cache.load_or_compute(paper_id, needed_figs)

    max_sims: list[float] = []
    for pid in pred_ids:
        p_emb = image_embs.get(pid)
        if p_emb is None:
            max_sims.append(0.0)
            continue
        gold_sims: list[float] = []
        for gid in gold:
            g_emb = image_embs.get(gid)
            if g_emb is None:
                continue
            gold_sims.append(cosine_sim(p_emb, g_emb))
        max_sims.append(max(gold_sims) if gold_sims else 0.0)

    return float(np.mean(max_sims)) if max_sims else 0.0
