from __future__ import annotations

from pathlib import Path

from m3sum.clients.openai_llm import OpenAILLMClient
from m3sum.data.schema import QueryBundle, SubQuery

PROMPT_PATH = Path(__file__).parent / "prompts" / "query_construction.txt"


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_queries(
    paper_id: str,
    problem_text: str,
    llm: OpenAILLMClient,
    dry_run: bool = False,
) -> QueryBundle:
    if dry_run:
        return QueryBundle(
            paper_id=paper_id,
            problem_text=problem_text,
            sub_queries=[
                SubQuery("分析", "分析问题背景与关键约束条件", ["背景", "约束"]),
                SubQuery("建模", "建立数学模型与核心假设", ["模型", "假设"]),
                SubQuery("求解", "求解算法与结果验证方法", ["算法", "结果"]),
            ],
        )

    system = _load_prompt()
    user = f"赛题全文：\n\n{problem_text}"
    data = llm.chat_json(system, user)

    sub_queries: list[SubQuery] = []
    for item in data.get("sub_queries", []):
        sub_queries.append(
            SubQuery(
                dimension=item.get("dimension", ""),
                query=item.get("query", ""),
                keywords=item.get("keywords", []),
            )
        )

    if len(sub_queries) != 3:
        raise ValueError(f"Expected 3 sub_queries, got {len(sub_queries)} for {paper_id}")

    return QueryBundle(
        paper_id=paper_id,
        problem_text=problem_text,
        sub_queries=sub_queries,
    )
