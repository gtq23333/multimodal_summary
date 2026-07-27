from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from m3sum.config import PipelineConfig
from m3sum.data.corpus_adapter import CorpusAdapter
from m3sum.data.schema import DocumentBundle, FigureMeta
from m3sum.eval.stage2_rerank_metrics import (
    image_precision_at_k,
    image_recall_at_k,
    jaccard_at_k,
    maxsim_at_k,
)
from m3sum.eval.stage2_reranking_eval import build_stage2_rankers
from m3sum.stage2_rerank.baselines.base import RankedFigure, Stage2Sample
from m3sum.stage2_rerank.sample_loader import load_all_stage2_samples


ALL_FIGURES_METHOD = "All-Figures"
DYNAMIC_UNION_METHOD = "Dynamic-Union-PQL"
SPECIAL_POOL_METHODS = frozenset({ALL_FIGURES_METHOD, DYNAMIC_UNION_METHOD})
DEFAULT_DYNAMIC_UNION_SOURCES = [
    "Proposed",
    "Qwen3-VL-Rerank-ImgCap+Link",
    "Layout-Order",
]


@dataclass
class CandidateFigure:
    candidate_id: str
    image_hash: str
    rank: int
    score: float | None
    caption: str
    source_type: str
    image_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "image_hash": self.image_hash,
            "rank": self.rank,
            "score": self.score,
            "caption": self.caption,
            "source_type": self.source_type,
            "image_path": self.image_path,
        }


@dataclass
class CandidatePool:
    paper_id: str
    method_name: str
    pool_size: int
    candidates: list[CandidateFigure]
    ground_truth_ids: list[str]
    retrieval_metrics: dict[str, float]
    reference_summary: str | None = None
    reference_sequence: list[dict[str, Any]] | None = None
    pool_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def experiment_key(self) -> str:
        return safe_experiment_id(self.method_name, f"top{self.pool_size}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "method_name": self.method_name,
            "pool_size": self.pool_size,
            "candidates": [c.to_dict() for c in self.candidates],
            "ground_truth_ids": self.ground_truth_ids,
            "retrieval_metrics": self.retrieval_metrics,
            "reference_summary": self.reference_summary,
            "reference_sequence": self.reference_sequence,
            "pool_metadata": self.pool_metadata,
        }


def safe_experiment_id(*parts: object) -> str:
    raw = "__".join(str(p) for p in parts if p is not None and str(p))
    safe = []
    for ch in raw:
        safe.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(safe).strip("_")


def load_ground_truth(config: PipelineConfig, paper_id: str) -> dict[str, Any]:
    path = config.ground_truth_dir / f"{paper_id}.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _figure_lookup(figures: list[FigureMeta]) -> dict[str, FigureMeta]:
    return {fig.image_hash: fig for fig in figures}


def _candidate_from_figure(
    fig: FigureMeta,
    *,
    candidate_id: str,
    rank: int,
    score: float | None,
) -> CandidateFigure:
    return CandidateFigure(
        candidate_id=candidate_id,
        image_hash=fig.image_hash,
        rank=rank,
        score=score,
        caption=fig.caption or "",
        source_type=fig.source_type,
        image_path=fig.abs_image_path,
    )


def _candidates_from_ranked(
    ranked: list[RankedFigure],
    figures: list[FigureMeta],
    pool_size: int,
) -> list[CandidateFigure]:
    by_hash = _figure_lookup(figures)
    candidates: list[CandidateFigure] = []
    for idx, item in enumerate(ranked[:pool_size], start=1):
        fig = by_hash.get(item.figure_id)
        if fig is None:
            continue
        candidates.append(
            _candidate_from_figure(
                fig,
                candidate_id=f"C{idx}",
                rank=idx,
                score=float(item.score),
            )
        )
    return candidates


def _retrieval_metrics(
    ranked_ids: list[str],
    ground_truth_ids: set[str],
    pool_size: int,
    *,
    sample: Stage2Sample | None = None,
    clip_cache: Any | None = None,
) -> dict[str, float]:
    metrics = {
        f"ip@{pool_size}": round(image_precision_at_k(ranked_ids, ground_truth_ids, pool_size), 6),
        f"ir@{pool_size}": round(image_recall_at_k(ranked_ids, ground_truth_ids, pool_size), 6),
        f"jaccard@{pool_size}": round(jaccard_at_k(ranked_ids, ground_truth_ids, pool_size), 6),
    }
    if sample is not None and clip_cache is not None:
        metrics[f"maxsim@{pool_size}"] = round(
            maxsim_at_k(
                ranked_ids,
                ground_truth_ids,
                sample.figures,
                clip_cache,
                sample.paper_id,
                k=pool_size,
            ),
            6,
        )
    return metrics


def build_ranker_candidate_pools(
    config: PipelineConfig,
    *,
    methods: list[str] | None = None,
    pool_sizes: list[int] | None = None,
    skip_clip: bool = False,
) -> list[CandidatePool]:
    active_methods = methods if methods is not None else config.stage3_rerank_methods
    active_methods = [m for m in active_methods if m not in SPECIAL_POOL_METHODS]
    if not active_methods:
        return []
    active_pool_sizes = pool_sizes or config.stage3_pool_sizes
    samples = load_all_stage2_samples(config)
    if not samples:
        return []

    rankers, clip_cache, _ = build_stage2_rankers(
        config,
        skip_clip=skip_clip,
        active_methods=active_methods,
    )

    pools: list[CandidatePool] = []
    for method_name in active_methods:
        ranker = rankers.get(method_name)
        if ranker is None:
            continue
        for sample in samples:
            ranked = ranker.rank(sample)
            ranked_ids = [r.figure_id for r in ranked]
            for pool_size in active_pool_sizes:
                pools.append(
                    CandidatePool(
                        paper_id=sample.paper_id,
                        method_name=method_name,
                        pool_size=pool_size,
                        candidates=_candidates_from_ranked(
                            ranked,
                            sample.figures,
                            pool_size,
                        ),
                        ground_truth_ids=sorted(sample.ground_truth_ids),
                        retrieval_metrics=_retrieval_metrics(
                            ranked_ids,
                            sample.ground_truth_ids,
                            pool_size,
                            sample=sample,
                            clip_cache=clip_cache,
                        ),
                    )
                )
    return pools


def _rank_dynamic_union(
    ranked_by_method: dict[str, list[RankedFigure]],
    *,
    source_top_k: int,
    rrf_k: int,
) -> list[RankedFigure]:
    """Build an untruncated union; RRF only determines prompt order."""
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranked in ranked_by_method.values():
        for rank, item in enumerate(ranked[:source_top_k], start=1):
            scores[item.figure_id] = scores.get(item.figure_id, 0.0) + 1.0 / (rrf_k + rank)
            best_rank[item.figure_id] = min(best_rank.get(item.figure_id, rank), rank)

    ordered = sorted(
        scores,
        key=lambda figure_id: (-scores[figure_id], best_rank[figure_id], figure_id),
    )
    return [
        RankedFigure(
            figure_id=figure_id,
            score=scores[figure_id],
            rank=rank,
            method_name=DYNAMIC_UNION_METHOD,
        )
        for rank, figure_id in enumerate(ordered, start=1)
    ]


def build_special_candidate_pools(
    config: PipelineConfig,
    *,
    methods: list[str] | None = None,
    skip_clip: bool = False,
) -> list[CandidatePool]:
    """Build full-document and untruncated dynamic-union Stage-3 baselines."""
    active_methods = methods if methods is not None else config.stage3_rerank_methods
    requested = set(active_methods) & SPECIAL_POOL_METHODS
    if not requested:
        return []

    samples = load_all_stage2_samples(config)
    pools: list[CandidatePool] = []

    if ALL_FIGURES_METHOD in requested:
        for sample in samples:
            candidates = [
                _candidate_from_figure(
                    figure,
                    candidate_id=f"C{idx}",
                    rank=idx,
                    score=None,
                )
                for idx, figure in enumerate(sample.figures, start=1)
            ]
            ranked_ids = [candidate.image_hash for candidate in candidates]
            actual_size = len(candidates)
            pools.append(
                CandidatePool(
                    paper_id=sample.paper_id,
                    method_name=ALL_FIGURES_METHOD,
                    pool_size=actual_size,
                    candidates=candidates,
                    ground_truth_ids=sorted(sample.ground_truth_ids),
                    retrieval_metrics=_retrieval_metrics(
                        ranked_ids,
                        sample.ground_truth_ids,
                        actual_size,
                    ),
                    pool_metadata={
                        "pool_type": "all_figures",
                        "actual_candidate_count": actual_size,
                    },
                )
            )

    if DYNAMIC_UNION_METHOD in requested:
        dynamic_cfg = config.stage3_generation_config.get("dynamic_union", {})
        source_methods = list(
            dynamic_cfg.get("source_methods", DEFAULT_DYNAMIC_UNION_SOURCES)
        )
        source_top_k = int(dynamic_cfg.get("source_top_k", 6))
        rrf_k = int(dynamic_cfg.get("rrf_k", 60))
        rankers, _, _ = build_stage2_rankers(
            config,
            skip_clip=skip_clip,
            active_methods=source_methods,
        )
        missing = [method for method in source_methods if method not in rankers]
        if missing:
            raise ValueError(f"Dynamic Union 缺少 Stage-2 ranker: {missing}")

        for sample in samples:
            ranked_by_method = {
                method: rankers[method].rank(sample) for method in source_methods
            }
            union_ranked = _rank_dynamic_union(
                ranked_by_method,
                source_top_k=source_top_k,
                rrf_k=rrf_k,
            )
            actual_size = len(union_ranked)
            ranked_ids = [item.figure_id for item in union_ranked]
            pools.append(
                CandidatePool(
                    paper_id=sample.paper_id,
                    method_name=DYNAMIC_UNION_METHOD,
                    pool_size=actual_size,
                    candidates=_candidates_from_ranked(
                        union_ranked,
                        sample.figures,
                        actual_size,
                    ),
                    ground_truth_ids=sorted(sample.ground_truth_ids),
                    retrieval_metrics=_retrieval_metrics(
                        ranked_ids,
                        sample.ground_truth_ids,
                        actual_size,
                    ),
                    pool_metadata={
                        "pool_type": "dynamic_union",
                        "source_methods": source_methods,
                        "source_top_k": source_top_k,
                        "rrf_k": rrf_k,
                        "nominal_budget": len(source_methods) * source_top_k,
                        "actual_candidate_count": actual_size,
                    },
                )
            )

    return pools


def build_reference_candidate_pool(
    config: PipelineConfig,
    sample: Stage2Sample,
    doc: DocumentBundle,
    pool_size: int,
) -> CandidatePool:
    gt = load_ground_truth(config, sample.paper_id)
    insertion_gt = gt.get("insertion_gt", {})
    selected_hashes = list(
        insertion_gt.get("selected_hashes")
        or gt.get("retrieval_gt", {}).get("relevant_figure_hashes", [])
    )
    selected_hashes = selected_hashes[:pool_size]
    by_hash = _figure_lookup(doc.figures)
    candidates: list[CandidateFigure] = []
    for idx, image_hash in enumerate(selected_hashes, start=1):
        fig = by_hash.get(image_hash)
        if fig is None:
            continue
        candidates.append(
            _candidate_from_figure(
                fig,
                candidate_id=f"C{idx}",
                rank=idx,
                score=1.0,
            )
        )
    ranked_ids = [c.image_hash for c in candidates]
    return CandidatePool(
        paper_id=sample.paper_id,
        method_name=config.stage3_reference_method,
        pool_size=pool_size,
        candidates=candidates,
        ground_truth_ids=sorted(sample.ground_truth_ids),
        retrieval_metrics=_retrieval_metrics(ranked_ids, sample.ground_truth_ids, pool_size),
        reference_summary=insertion_gt.get("reference_text") or doc.abstract_text,
        reference_sequence=insertion_gt.get("multimodal_sequence"),
    )


def build_reference_candidate_pools(
    config: PipelineConfig,
    *,
    pool_sizes: list[int] | None = None,
) -> list[CandidatePool]:
    active_pool_sizes = pool_sizes or config.stage3_pool_sizes
    corpus = CorpusAdapter(config)
    pools: list[CandidatePool] = []
    for sample in load_all_stage2_samples(config):
        doc = corpus.load_document(sample.paper_id)
        for pool_size in active_pool_sizes:
            pools.append(build_reference_candidate_pool(config, sample, doc, pool_size))
    return pools


def build_all_candidate_pools(
    config: PipelineConfig,
    *,
    methods: list[str] | None = None,
    pool_sizes: list[int] | None = None,
    include_reference: bool = True,
    skip_clip: bool = False,
) -> list[CandidatePool]:
    pools = build_ranker_candidate_pools(
        config,
        methods=methods,
        pool_sizes=pool_sizes,
        skip_clip=skip_clip,
    )
    pools.extend(
        build_special_candidate_pools(
            config,
            methods=methods,
            skip_clip=skip_clip,
        )
    )
    if include_reference:
        pools.extend(build_reference_candidate_pools(config, pool_sizes=pool_sizes))
    return pools


def save_candidate_pool_manifest(path: Path, pools: list[CandidatePool]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [pool.to_dict() for pool in pools]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
