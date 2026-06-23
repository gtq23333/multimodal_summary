from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from m3sum.clients.openai_embedder import OpenAIEmbedder
from m3sum.clients.openai_llm import OpenAILLMClient
from m3sum.clients.openai_vlm import OpenAIVLMClient
from m3sum.config import PipelineConfig, resolve_api_credentials
from m3sum.data.corpus_adapter import CorpusAdapter
from m3sum.data.schema import QueryBundle, SubQuery
from m3sum.stage1_query.query_builder import build_queries
from m3sum.stage2_rerank.hybrid_retriever import EmbeddingCache, HybridRetriever
from m3sum.stage2_rerank.reranker import rerank_figures
from m3sum.stage3_generation.adaptive_generator import generate_summary
from m3sum.stage3_generation.vlm_describer import describe_figures


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _query_bundle_from_dict(data: dict[str, Any]) -> QueryBundle:
    return QueryBundle(
        paper_id=data["paper_id"],
        problem_text=data["problem_text"],
        sub_queries=[
            SubQuery(
                dimension=q["dimension"],
                query=q["query"],
                keywords=q["keywords"],
            )
            for q in data["sub_queries"]
        ],
    )


class PipelineRunner:
    def __init__(
        self,
        config: PipelineConfig,
        dry_run: bool = False,
        vlm_mode: str = "caption",
        from_cache: bool = True,
    ):
        self.config = config
        self.dry_run = dry_run
        self.vlm_mode = vlm_mode
        self.from_cache = from_cache
        self.corpus = CorpusAdapter(config)

        if not dry_run:
            creds = resolve_api_credentials(config)
            self.llm = OpenAILLMClient(config.llm_model, creds)
            self.embedder = OpenAIEmbedder(config.embed_model, creds)
            self.vlm = OpenAIVLMClient(config.vlm_model, creds)
        else:
            self.llm = None
            self.embedder = None
            self.vlm = None

        self.embed_cache = EmbeddingCache(
            config.embed_cache_dir,
            self.embedder,
        )
        self.hybrid = HybridRetriever(
            bm25_weight=config.bm25_weight,
            vector_weight=config.vector_weight,
            top_p=config.top_p,
        )

    def run_stage1(self, paper_id: str, force: bool = False) -> dict[str, Any]:
        out_path = self.config.stage1_dir / f"{paper_id}.json"
        if self.from_cache and not force and out_path.is_file():
            return _load_json(out_path)

        doc = self.corpus.load_document(paper_id)
        bundle = build_queries(
            paper_id,
            doc.problem_text,
            self.llm,
            dry_run=self.dry_run,
        )
        result = bundle.to_dict()
        _save_json(out_path, result)
        return result

    def run_stage2(self, paper_id: str, force: bool = False) -> dict[str, Any]:
        out_path = self.config.stage2_dir / f"{paper_id}.json"
        if self.from_cache and not force and out_path.is_file():
            return _load_json(out_path)

        stage1_path = self.config.stage1_dir / f"{paper_id}.json"
        if not stage1_path.is_file():
            self.run_stage1(paper_id)

        stage1 = _load_json(stage1_path)
        query_bundle = _query_bundle_from_dict(stage1)
        doc = self.corpus.load_document(paper_id)

        block_embs, fig_embs = self.embed_cache.load_or_compute(
            paper_id,
            doc.blocks,
            doc.figures,
            dry_run=self.dry_run,
        )

        query_embeddings: list[np.ndarray] = []
        for q in query_bundle.sub_queries:
            if self.dry_run:
                dim = next(iter(block_embs.values())).shape[0] if block_embs else 64
                query_embeddings.append(np.random.randn(dim).astype(np.float32))
            else:
                vec = self.embedder.embed_one(q.query + " " + " ".join(q.keywords))
                query_embeddings.append(np.array(vec, dtype=np.float32))

        result = rerank_figures(
            paper_id=paper_id,
            sub_queries=query_bundle.sub_queries,
            blocks=doc.blocks,
            figures=doc.figures,
            block_embeddings=block_embs,
            figure_embeddings=fig_embs,
            query_embeddings=query_embeddings,
            caption_patterns=self.config.caption_patterns,
            alpha=self.config.alpha,
            distance_tiers=self.config.distance_tiers,
            hybrid=self.hybrid,
        )
        _save_json(out_path, result)
        return result

    def run_stage3(self, paper_id: str, force: bool = False) -> dict[str, Any]:
        out_path = self.config.stage3_dir / f"{paper_id}.json"
        if self.from_cache and not force and out_path.is_file():
            return _load_json(out_path)

        stage1_path = self.config.stage1_dir / f"{paper_id}.json"
        stage2_path = self.config.stage2_dir / f"{paper_id}.json"
        if not stage1_path.is_file():
            self.run_stage1(paper_id)
        if not stage2_path.is_file():
            self.run_stage2(paper_id)

        stage1 = _load_json(stage1_path)
        stage2 = _load_json(stage2_path)
        query_bundle = _query_bundle_from_dict(stage1)
        doc = self.corpus.load_document(paper_id)

        hash_to_fig = {f.image_hash: f for f in doc.figures}
        top3_hashes = [item["image_hash"] for item in stage2.get("top3_figures", [])]
        top3_figs = [hash_to_fig[h] for h in top3_hashes if h in hash_to_fig]

        descriptions = describe_figures(
            top3_figs,
            vlm=self.vlm,
            mode=self.vlm_mode if not self.dry_run else "caption",
        )

        gen = generate_summary(
            doc.abstract_text,
            query_bundle,
            descriptions,
            self.llm,
            dry_run=self.dry_run,
        )

        result = {
            "paper_id": paper_id,
            "figure_descriptions": descriptions,
            **gen,
        }
        _save_json(out_path, result)
        return result

    def run(
        self,
        paper_ids: list[str] | None = None,
        stage: str = "all",
        force: bool = False,
    ) -> None:
        ids = paper_ids or self.config.resolved_sample_ids()
        for paper_id in ids:
            if stage in ("1", "all"):
                self.run_stage1(paper_id, force=force)
            if stage in ("2", "all"):
                self.run_stage2(paper_id, force=force)
            if stage in ("3", "all"):
                self.run_stage3(paper_id, force=force)
