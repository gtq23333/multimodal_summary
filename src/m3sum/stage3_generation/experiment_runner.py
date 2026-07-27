from __future__ import annotations

import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from m3sum.config import PipelineConfig, resolve_api_credentials
from m3sum.data.corpus_adapter import CorpusAdapter
from m3sum.data.schema import QueryBundle, SubQuery
from m3sum.stage3_generation.candidate_pool import (
    CandidatePool,
    build_all_candidate_pools,
    safe_experiment_id,
    save_candidate_pool_manifest,
)
from m3sum.stage3_generation.generators import generate_for_pool
from m3sum.stage3_generation.text_context_builder import build_generation_context

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def query_bundle_from_stage1(config: PipelineConfig, paper_id: str) -> QueryBundle:
    stage1_path = config.stage1_dir / f"{paper_id}.json"
    if not stage1_path.is_file():
        raise FileNotFoundError(f"缺少 Stage-1 输出: {stage1_path}")
    data = _load_json(stage1_path)
    return QueryBundle(
        paper_id=data["paper_id"],
        problem_text=data.get("problem_text", ""),
        sub_queries=[
            SubQuery(
                dimension=q["dimension"],
                query=q["query"],
                keywords=q.get("keywords", []),
            )
            for q in data.get("sub_queries", [])
        ],
    )


def _selected_pools(
    pools: list[CandidatePool],
    *,
    paper_ids: set[str] | None,
) -> list[CandidatePool]:
    if not paper_ids:
        return pools
    return [pool for pool in pools if pool.paper_id in paper_ids]


def _artifact_path(
    config: PipelineConfig,
    *,
    method_name: str,
    pool_size: int,
    strategy: str,
    model: str,
    paper_id: str,
) -> Path:
    experiment_id = safe_experiment_id(method_name, f"top{pool_size}", strategy, model)
    return config.stage3_generation_dir / experiment_id / f"{paper_id}.json"


def _dry_artifact(
    config: PipelineConfig,
    *,
    model: str,
    strategy: str,
    pool: CandidatePool,
    generation_context: dict[str, Any],
) -> dict[str, Any]:
    summary = pool.reference_summary or "DRY-RUN Stage3 多模态摘要。"
    inserted = [c.image_hash for c in pool.candidates[:2]]
    return {
        "schema_version": "0.1.0",
        "experiment_id": safe_experiment_id(pool.method_name, f"top{pool.pool_size}", strategy, model),
        "paper_id": pool.paper_id,
        "method_name": pool.method_name,
        "pool_size": pool.pool_size,
        "strategy": strategy,
        "model": model,
        "candidate_pool": pool.to_dict(),
        "generation_context": {
            "retrieved_evidence": generation_context.get("retrieved_evidence", []),
            "body_char_count": len(generation_context.get("body_text", "")),
        },
        "generated_summary": summary,
        "inserted_figures": inserted,
        "selected_image_hashes": inserted,
        "placeholders": [f"[Insert Figure C{i + 1}]" for i in range(len(inserted))],
        "rationale": "dry_run placeholder",
    }


def _combos_for_pool(
    config: PipelineConfig,
    pool: CandidatePool,
    *,
    active_models: list[str],
    active_strategies: list[str],
) -> list[tuple[str, str]]:
    if pool.method_name == config.stage3_reference_method:
        return [("reference_oracle", "reference")]
    return [(strategy, model) for strategy in active_strategies for model in active_models]


def _run_paper_pools(
    config: PipelineConfig,
    *,
    creds,
    pools: list[CandidatePool],
    active_models: list[str],
    active_strategies: list[str],
    force: bool,
) -> list[Path]:
    if not pools:
        return []

    paper_id = pools[0].paper_id
    corpus = CorpusAdapter(config)
    doc = corpus.load_document(paper_id)
    query_bundle = query_bundle_from_stage1(config, paper_id)
    generation_context = build_generation_context(config, doc, query_bundle.sub_queries)
    written: list[Path] = []

    for pool in pools:
        for strategy, model in _combos_for_pool(
            config,
            pool,
            active_models=active_models,
            active_strategies=active_strategies,
        ):
            out_path = _artifact_path(
                config,
                method_name=pool.method_name,
                pool_size=pool.pool_size,
                strategy=strategy,
                model=model,
                paper_id=pool.paper_id,
            )
            if out_path.is_file() and not force:
                written.append(out_path)
                continue

            logger.info(
                "生成 Stage3: paper=%s | method=%s | top=%d | strategy=%s | model=%s",
                pool.paper_id,
                pool.method_name,
                pool.pool_size,
                strategy,
                model,
            )
            if config.dry_run:
                artifact = _dry_artifact(
                    config,
                    model=model,
                    strategy=strategy,
                    pool=pool,
                    generation_context=generation_context,
                )
            else:
                artifact = generate_for_pool(
                    config,
                    creds,
                    model=model,
                    strategy=strategy,
                    doc=doc,
                    query_bundle=query_bundle,
                    pool=pool,
                    generation_context=generation_context,
                )
            _save_json(out_path, artifact)
            written.append(out_path)
    return written


def run_stage3_generation_experiments(
    config: PipelineConfig,
    *,
    methods: list[str] | None = None,
    models: list[str] | None = None,
    pool_sizes: list[int] | None = None,
    strategies: list[str] | None = None,
    paper_ids: list[str] | None = None,
    force: bool = False,
    skip_clip: bool = False,
    include_reference: bool = True,
    parallel_papers: int | None = None,
) -> list[Path]:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    active_models = models or config.stage3_generation_models
    active_strategies = strategies or config.stage3_strategies
    active_pool_sizes = pool_sizes or config.stage3_pool_sizes
    gen_cfg = config.stage3_generation_config
    max_workers = max(1, int(parallel_papers or gen_cfg.get("parallel_papers", 1)))

    pools = build_all_candidate_pools(
        config,
        methods=methods,
        pool_sizes=active_pool_sizes,
        include_reference=include_reference,
        skip_clip=skip_clip,
    )
    pools = _selected_pools(pools, paper_ids=set(paper_ids or []))
    save_candidate_pool_manifest(config.stage3_generation_dir / "candidate_pools.json", pools)

    if not pools:
        logger.warning("没有可运行的 Stage3 候选池")
        return []

    creds = resolve_api_credentials(config)
    written: list[Path] = []
    pools_by_paper: dict[str, list[CandidatePool]] = defaultdict(list)
    for pool in pools:
        pools_by_paper[pool.paper_id].append(pool)

    logger.info("Stage3 生成并发论文数: %d", max_workers)
    if max_workers == 1:
        for paper_pools in pools_by_paper.values():
            written.extend(
                _run_paper_pools(
                    config,
                    creds=creds,
                    pools=paper_pools,
                    active_models=active_models,
                    active_strategies=active_strategies,
                    force=force,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _run_paper_pools,
                    config,
                    creds=creds,
                    pools=paper_pools,
                    active_models=active_models,
                    active_strategies=active_strategies,
                    force=force,
                )
                for paper_pools in pools_by_paper.values()
            ]
            for future in as_completed(futures):
                written.extend(future.result())

    manifest = {
        "schema_version": "0.1.0",
        "artifact_count": len(written),
        "artifacts": [str(path) for path in written],
    }
    _save_json(config.stage3_generation_dir / "manifest.json", manifest)
    return written
