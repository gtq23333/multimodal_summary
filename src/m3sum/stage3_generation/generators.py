from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from m3sum.clients.api_client import build_openai_client
from m3sum.clients.openai_llm import OpenAILLMClient
from m3sum.config import ApiCredentials, PipelineConfig
from m3sum.data.schema import DocumentBundle, QueryBundle
from m3sum.stage3_generation.candidate_pool import CandidatePool, safe_experiment_id

TEXT_RAG_SYSTEM_PROMPT = """你是数学建模论文摘要写作专家。
请基于论文全文与检索增强证据，生成一段中文纯文本摘要。摘要应覆盖问题背景、建模思路、主要方法、关键结果与结论，避免编造未在材料中出现的信息。"""

REWRITE_SYSTEM_PROMPT = """你是数学建模论文多模态摘要修订专家。
你将收到一段纯文本摘要以及若干张候选图片。请重写为图文融合摘要，并在最能降低理解成本的位置插入候选图。

规则：
- 优先选择方法图、流程图、架构图、关键结果图；
- 避免只因浅层语义相关而插入装饰性图片；
- 若候选图没有明显帮助，可以少插或不插；
- 占位符格式必须为 [Insert Figure C1]，其中 C1 是候选编号；
- 仅输出 JSON。"""

END_TO_END_SYSTEM_PROMPT = """你是数学建模论文多模态摘要生成专家。
你将收到论文正文、检索增强证据以及候选图片。请端到端生成一份图文融合摘要，并选择有认知增益的图片插入到合适位置。

规则：
- 摘要应覆盖问题背景、模型方法、关键结果和结论；
- 图片应服务于理解建模方法、算法流程、系统结构或核心结果；
- 避免插入装饰性或重复文本信息的图片；
- 占位符格式必须为 [Insert Figure C1]，其中 C1 是候选编号；
- 仅输出 JSON。"""


def resolve_multimodal_model(config: PipelineConfig, model: str) -> tuple[str, bool]:
    """Return the model used for image-capable calls and whether fallback was applied."""
    capable = set(config.stage3_multimodal_models)
    if model in capable:
        return model, False
    return config.stage3_multimodal_fallback, True


def encode_image_part(image_path: str) -> dict[str, Any]:
    path = Path(image_path)
    if not path.is_file():
        return {"type": "text", "text": f"[图片文件不存在: {path.name}]"}
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix or "png"
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/{mime};base64,{data}"},
    }


def build_candidate_list(pool: CandidatePool) -> tuple[str, dict[str, str]]:
    lines: list[str] = []
    cid_to_hash: dict[str, str] = {}
    for candidate in pool.candidates:
        cid_to_hash[candidate.candidate_id] = candidate.image_hash
        score = "" if candidate.score is None else f", rerank_score={candidate.score:.6f}"
        lines.append(
            f"- {candidate.candidate_id}: image_hash={candidate.image_hash}{score}, "
            f"caption={candidate.caption or '(无图注)'}"
        )
    return "\n".join(lines), cid_to_hash


def build_multimodal_messages(
    system_prompt: str,
    user_text: str,
    pool: CandidatePool,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for candidate in pool.candidates:
        content.append({"type": "text", "text": f"\n--- 候选图 {candidate.candidate_id} ---"})
        if candidate.caption:
            content.append({"type": "text", "text": f"图注：{candidate.caption}"})
        content.append(encode_image_part(candidate.image_path))
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def call_multimodal_json(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)


def generate_text_rag_summary(
    llm: OpenAILLMClient,
    doc: DocumentBundle,
    query_bundle: QueryBundle,
    generation_context: dict[str, Any],
) -> str:
    user_payload = {
        "problem_text": query_bundle.problem_text,
        "sub_queries": query_bundle.to_dict()["sub_queries"],
        "original_abstract": doc.abstract_text,
        "retrieved_evidence": generation_context["retrieved_evidence_text"],
        "body_text": generation_context["body_text"],
        "output_requirements": "输出一段中文纯文本摘要，不要包含图片占位符。",
    }
    user = json.dumps(user_payload, ensure_ascii=False, indent=2)
    return llm.chat_text(TEXT_RAG_SYSTEM_PROMPT, user).strip()


def generate_text_rag_then_rewrite(
    config: PipelineConfig,
    creds: ApiCredentials,
    *,
    model: str,
    doc: DocumentBundle,
    query_bundle: QueryBundle,
    pool: CandidatePool,
    generation_context: dict[str, Any],
) -> dict[str, Any]:
    gen_cfg = config.stage3_generation_config
    llm = OpenAILLMClient(model, creds)
    text_summary = generate_text_rag_summary(llm, doc, query_bundle, generation_context)
    multimodal_model, used_fallback = resolve_multimodal_model(config, model)
    candidate_list, _ = build_candidate_list(pool)
    user_payload = {
        "text_rag_summary": text_summary,
        "candidate_figures": candidate_list,
        "output_json_schema": {
            "generated_summary": "含 [Insert Figure Cn] 占位符的摘要全文",
            "inserted_figures": ["C1"],
            "selected_image_hashes": ["完整 image_hash"],
            "placeholders": ["[Insert Figure C1]"],
            "rationale": "简要说明选图和插入理由",
        },
    }
    messages = build_multimodal_messages(
        REWRITE_SYSTEM_PROMPT,
        json.dumps(user_payload, ensure_ascii=False, indent=2),
        pool,
    )
    client = build_openai_client(creds)
    data = call_multimodal_json(
        client,
        multimodal_model,
        messages,
        max_tokens=int(gen_cfg.get("max_output_tokens", 4096)),
    )
    data["text_rag_summary"] = text_summary
    payload = _normalize_generation_payload(data, pool, fallback_summary=text_summary)
    payload["multimodal_model"] = multimodal_model
    payload["multimodal_model_fallback"] = used_fallback
    return payload


def generate_end_to_end_vlm(
    config: PipelineConfig,
    creds: ApiCredentials,
    *,
    model: str,
    doc: DocumentBundle,
    query_bundle: QueryBundle,
    pool: CandidatePool,
    generation_context: dict[str, Any],
) -> dict[str, Any]:
    gen_cfg = config.stage3_generation_config
    multimodal_model, used_fallback = resolve_multimodal_model(config, model)
    candidate_list, _ = build_candidate_list(pool)
    user_payload = {
        "problem_text": query_bundle.problem_text,
        "sub_queries": query_bundle.to_dict()["sub_queries"],
        "original_abstract": doc.abstract_text,
        "retrieved_evidence": generation_context["retrieved_evidence_text"],
        "body_text": generation_context["body_text"],
        "candidate_figures": candidate_list,
        "output_json_schema": {
            "generated_summary": "含 [Insert Figure Cn] 占位符的摘要全文",
            "inserted_figures": ["C1"],
            "selected_image_hashes": ["完整 image_hash"],
            "placeholders": ["[Insert Figure C1]"],
            "rationale": "简要说明选图和插入理由",
        },
    }
    messages = build_multimodal_messages(
        END_TO_END_SYSTEM_PROMPT,
        json.dumps(user_payload, ensure_ascii=False, indent=2),
        pool,
    )
    client = build_openai_client(creds)
    data = call_multimodal_json(
        client,
        multimodal_model,
        messages,
        max_tokens=int(gen_cfg.get("max_output_tokens", 4096)),
    )
    payload = _normalize_generation_payload(data, pool, fallback_summary=doc.abstract_text)
    payload["multimodal_model"] = multimodal_model
    payload["multimodal_model_fallback"] = used_fallback
    return payload


def build_reference_artifact(
    *,
    config: PipelineConfig,
    doc: DocumentBundle,
    pool: CandidatePool,
) -> dict[str, Any]:
    summary = pool.reference_summary or doc.abstract_text
    inserted = [c.image_hash for c in pool.candidates]
    placeholders = [f"[Insert Figure {c.candidate_id}]" for c in pool.candidates]
    return {
        "generated_summary": summary,
        "inserted_figures": inserted,
        "selected_image_hashes": inserted,
        "placeholders": placeholders,
        "rationale": "Reference-Oracle 使用人工标注摘要与图片序列。",
        "reference_sequence": pool.reference_sequence,
    }


def generate_for_pool(
    config: PipelineConfig,
    creds: ApiCredentials,
    *,
    model: str,
    strategy: str,
    doc: DocumentBundle,
    query_bundle: QueryBundle,
    pool: CandidatePool,
    generation_context: dict[str, Any],
) -> dict[str, Any]:
    if pool.method_name == config.stage3_reference_method:
        payload = build_reference_artifact(config=config, doc=doc, pool=pool)
    elif strategy == "text_rag_then_rewrite":
        payload = generate_text_rag_then_rewrite(
            config,
            creds,
            model=model,
            doc=doc,
            query_bundle=query_bundle,
            pool=pool,
            generation_context=generation_context,
        )
    elif strategy == "end_to_end_vlm":
        payload = generate_end_to_end_vlm(
            config,
            creds,
            model=model,
            doc=doc,
            query_bundle=query_bundle,
            pool=pool,
            generation_context=generation_context,
        )
    else:
        raise ValueError(f"未知 Stage3 生成策略: {strategy}")

    experiment_id = safe_experiment_id(pool.method_name, f"top{pool.pool_size}", strategy, model)
    return {
        "schema_version": "0.1.0",
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
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
        **payload,
    }


def _normalize_generation_payload(
    data: dict[str, Any],
    pool: CandidatePool,
    *,
    fallback_summary: str,
) -> dict[str, Any]:
    cid_to_hash = {c.candidate_id: c.image_hash for c in pool.candidates}
    selected_hashes = _resolve_selected_hashes(data, cid_to_hash)
    placeholders = data.get("placeholders") or [
        f"[Insert Figure {cid}]"
        for cid, image_hash in cid_to_hash.items()
        if image_hash in selected_hashes
    ]
    return {
        "generated_summary": data.get("generated_summary") or fallback_summary,
        "inserted_figures": selected_hashes,
        "selected_image_hashes": selected_hashes,
        "placeholders": placeholders,
        "rationale": data.get("rationale", ""),
        **({"text_rag_summary": data["text_rag_summary"]} if "text_rag_summary" in data else {}),
    }


def _resolve_selected_hashes(data: dict[str, Any], cid_to_hash: dict[str, str]) -> list[str]:
    selected: list[str] = []
    raw_items = data.get("selected_image_hashes") or data.get("inserted_figures") or []
    for item in raw_items:
        value = str(item).strip()
        image_hash = cid_to_hash.get(value, value)
        if image_hash in cid_to_hash.values() and image_hash not in selected:
            selected.append(image_hash)
    return selected
