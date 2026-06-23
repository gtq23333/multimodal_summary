from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from m3sum.clients.openai_llm import OpenAILLMClient
from m3sum.data.schema import QueryBundle

PROMPT_PATH = Path(__file__).parent / "prompts" / "generation.txt"


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def generate_summary(
    abstract_text: str,
    query_bundle: QueryBundle,
    figure_descriptions: list[dict[str, str]],
    llm: OpenAILLMClient,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dry_run:
        inserted = [figure_descriptions[0]["image_hash"]] if figure_descriptions else []
        prefix = inserted[0][:8] if inserted else ""
        return {
            "generated_summary": abstract_text,
            "inserted_figures": inserted,
            "placeholders": [f"[Insert Figure {prefix}]"] if inserted else [],
        }

    system = _load_prompt()
    user_payload = {
        "original_abstract": abstract_text,
        "sub_queries": query_bundle.to_dict()["sub_queries"],
        "candidate_figures": figure_descriptions,
    }
    user = json.dumps(user_payload, ensure_ascii=False, indent=2)
    data = llm.chat_json(system, user)

    inserted = data.get("inserted_figures", [])
    if len(inserted) > 2:
        inserted = inserted[:2]

    summary = data.get("generated_summary", abstract_text)
    placeholders = data.get("placeholders", [])
    if not placeholders and inserted:
        placeholders = [f"[Insert Figure {h[:8]}]" for h in inserted]

    return {
        "generated_summary": summary,
        "inserted_figures": inserted,
        "placeholders": placeholders,
    }
