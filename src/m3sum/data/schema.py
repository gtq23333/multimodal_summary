from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from m3sum.stage2_rerank.query_text import sub_query_search_text


@dataclass
class Block:
    block_id: str
    block_idx: int
    block_type: Literal["text", "figure"]
    text: str
    char_start: int
    char_end: int
    image_hash: str | None = None
    caption_refs: list[tuple[int, ...]] = field(default_factory=list)
    has_caption_ref: bool = False


@dataclass
class FigureMeta:
    image_hash: str
    caption: str
    source_type: str
    pos: int
    page_idx: int | None
    body_order: int
    abs_image_path: str
    img_path: str = ""


@dataclass
class SubQuery:
    dimension: str
    query: str
    keywords: list[str]
    embedding: list[float] | None = None

    def search_text(self, *, use_keywords: bool = True) -> str:
        """Stage-2 检索 query 文本。"""
        return sub_query_search_text(self, use_keywords=use_keywords)


@dataclass
class QueryBundle:
    paper_id: str
    problem_text: str
    sub_queries: list[SubQuery]

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "problem_text": self.problem_text,
            "sub_queries": [
                {
                    "dimension": q.dimension,
                    "query": q.query,
                    "keywords": q.keywords,
                }
                for q in self.sub_queries
            ],
        }


@dataclass
class DocumentBundle:
    paper_id: str
    abstract_text: str
    body_text: str
    blocks: list[Block]
    figures: list[FigureMeta]
    problem_text: str
