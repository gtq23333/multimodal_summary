from __future__ import annotations

from rouge_score import rouge_scorer

ROUGE_KEYS = ("rouge1", "rouge2", "rougeL")


def compute_rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(list(ROUGE_KEYS), use_stemmer=False)
    scores = scorer.score(reference, prediction)
    return {
        "rouge_1": scores["rouge1"].fmeasure,
        "rouge_2": scores["rouge2"].fmeasure,
        "rouge_l": scores["rougeL"].fmeasure,
    }


def rouge_l(prediction: str, reference: str) -> float:
    return compute_rouge_scores(prediction, reference)["rouge_l"]


def aggregate_rouge(scores: list[float]) -> dict:
    if not scores:
        return {"ROUGE-L": 0.0, "n": 0}
    return {"ROUGE-L": round(sum(scores) / len(scores), 4), "n": len(scores)}
