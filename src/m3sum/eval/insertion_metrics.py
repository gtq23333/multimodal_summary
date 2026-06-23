from __future__ import annotations


def precision_recall_f1(predicted: set[str], gold: set[str]) -> dict[str, float]:
    if not predicted and not gold:
        return {"Precision": 1.0, "Recall": 1.0, "F1": 1.0}
    if not predicted:
        return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0}
    if not gold:
        return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0}

    tp = len(predicted & gold)
    precision = tp / len(predicted)
    recall = tp / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
    }


def aggregate_insertion(results: list[dict]) -> dict:
    if not results:
        return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0, "n": 0}

    p = sum(r["Precision"] for r in results) / len(results)
    r = sum(r["Recall"] for r in results) / len(results)
    f = sum(r["F1"] for r in results) / len(results)
    return {
        "Precision": round(p, 4),
        "Recall": round(r, 4),
        "F1": round(f, 4),
        "n": len(results),
    }
