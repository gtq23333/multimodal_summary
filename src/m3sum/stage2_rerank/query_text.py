from __future__ import annotations

from typing import Protocol


class _QueryLike(Protocol):
    query: str
    keywords: list[str]


def sub_query_search_text(q: _QueryLike, *, use_keywords: bool) -> str:
    """Stage-2 检索 query 文本；use_keywords=False 时仅用 query 字段。"""
    if use_keywords and q.keywords:
        return q.query + " " + " ".join(q.keywords)
    return q.query


def sub_query_search_texts(
    sub_queries: list[_QueryLike],
    *,
    use_keywords: bool,
) -> list[str]:
    return [sub_query_search_text(q, use_keywords=use_keywords) for q in sub_queries]
