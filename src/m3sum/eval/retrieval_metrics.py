from __future__ import annotations


def hit_at_k(predicted: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    top_k = predicted[:k]
    return 1.0 if any(h in gold for h in top_k) else 0.0


def mrr(predicted: list[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    for i, h in enumerate(predicted):
        if h in gold:
            return 1.0 / (i + 1)
    return 0.0


def aggregate_retrieval(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"Hit@1": 0.0, "Hit@3": 0.0, "MRR": 0.0, "n": 0}

    h1 = sum(r["hit@1"] for r in results) / n
    h3 = sum(r["hit@3"] for r in results) / n
    mrr_val = sum(r["mrr"] for r in results) / n
    recall_top_p = sum(r.get("gt_in_pool", 0) for r in results) / n

    return {
        "Hit@1": round(h1, 4),
        "Hit@3": round(h3, 4),
        "MRR": round(mrr_val, 4),
        "gt_in_top_p_recall": round(recall_top_p, 4),
        "n": n,
    }
