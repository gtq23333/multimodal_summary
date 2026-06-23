from __future__ import annotations

from rouge_score import rouge_scorer


def rouge_l(prediction: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(reference, prediction)
    return scores["rougeL"].fmeasure


def aggregate_rouge(scores: list[float]) -> dict:
    if not scores:
        return {"ROUGE-L": 0.0, "n": 0}
    return {"ROUGE-L": round(sum(scores) / len(scores), 4), "n": len(scores)}
