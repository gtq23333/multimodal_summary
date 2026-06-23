from __future__ import annotations


def tokenize(text: str) -> list[str]:
    """确定性字符级分词，与 HybridRetriever BM25 保持一致。"""
    return list(text)
