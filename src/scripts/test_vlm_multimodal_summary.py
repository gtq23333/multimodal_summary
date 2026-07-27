#!/usr/bin/env python3
"""
VLM 多模态摘要 A/B 测试脚本（独立运行，直接耦合项目数据层）。

从一篇样本的主方法 Stage-2 重排结果取 Top-6 候选图，分别用两种方式调用 VLM：
  A) 完整正文（无摘要）+ 6 张候选图 → 同时生成摘要并选图
  B) 原摘要 + 6 张候选图 → 向摘要中插入图片

API / 路径等配置写死读 configs/trial_31.yaml。

用法:
  cd src
  python scripts/test_vlm_multimodal_summary.py
  python scripts/test_vlm_multimodal_summary.py --paper-id 2018_G_A466.pdf-f42c5f5a-ee2d-4f66-84b0-d3e32153a4e5
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))

from m3sum.clients.api_client import build_openai_client
from m3sum.config import ApiCredentials, PipelineConfig, load_config, resolve_api_credentials
from m3sum.data.corpus_adapter import CorpusAdapter
from m3sum.data.schema import FigureMeta

CONFIG_PATH = SRC_ROOT / "configs" / "trial_31.yaml"
BODY_IMG_RE = re.compile(r"!\[\]\(images/([a-f0-9]+)\.(?:jpg|png|jpeg)\)", re.I)
DEFAULT_MODEL = "qwen3.6-27b"
DEFAULT_TOP_K = 6
DEFAULT_PAPER_ID = "2018_G_A466.pdf-f42c5f5a-ee2d-4f66-84b0-d3e32153a4e5"

SYSTEM_PROMPT_A = """你是数学建模论文多模态摘要生成专家。
你将收到一篇论文的完整正文（不含摘要）以及若干张候选图片。
请阅读正文与图片，完成两项任务：
1. 撰写一段适合作为论文摘要的多模态摘要文本；
2. 从候选图中选出 2-6 张对理解建模方法/框架最有认知增益的图，并在摘要相应位置插入占位符。

规则：
- 优先选择方法图、流程图、架构图；避免仅重复数值趋势的折线图；
- 若候选图皆冗余，inserted_figures 为空；
- 占位符格式：[Insert Figure {candidate_id}]，candidate_id 使用输入中的候选编号（如 C1）；
- 摘要应覆盖问题背景、建模思路、主要方法、关键结论。

仅输出 JSON。"""

SYSTEM_PROMPT_B = """你是数学建模论文多模态摘要修订专家。
你将收到论文原摘要以及若干张候选图片。
请在不大幅改变原摘要结构与措辞的前提下，向摘要中插入 2-6 张最有认知增益的候选图。

规则：
- 只做必要的插入与少量衔接句调整，不要完全重写摘要；
- 优先方法/流程/架构类图，避免冗余折线图；
- 若无需插图，inserted_figures 为空；
- 占位符格式：[Insert Figure {candidate_id}]，candidate_id 使用输入中的候选编号（如 C1）。

仅输出 JSON。"""

USER_TEMPLATE_A = """## 任务
请基于下方论文正文与候选图片，同时生成多模态摘要并选择要插入的图片。

## 论文正文（无摘要；正文中其它图片引用已移除，仅通过下方候选图提供）
{body_text}

## 候选图片（共 {n} 张，按相关性排序）
{candidate_list}

## 输出 JSON 格式
{{
  "generated_summary": "含占位符的摘要全文",
  "inserted_figures": ["C1", "C3"],
  "selected_image_hashes": ["完整image_hash", ...],
  "placeholders": ["[Insert Figure C1]", ...],
  "rationale": "简要说明选图理由"
}}"""

USER_TEMPLATE_B = """## 任务
请基于下方原摘要与候选图片，向摘要中插入合适的图片。

## 原摘要
{abstract_text}

## 候选图片（共 {n} 张，按相关性排序）
{candidate_list}

## 输出 JSON 格式
{{
  "generated_summary": "修订后的摘要全文（含占位符）",
  "inserted_figures": ["C1", "C3"],
  "selected_image_hashes": ["完整image_hash", ...],
  "placeholders": ["[Insert Figure C1]", ...],
  "rationale": "简要说明选图理由"
}}"""


def resolve_stage2_dir(config: PipelineConfig) -> Path:
    primary = config.stage2_dir
    if primary.is_dir() and any(primary.glob("*.json")):
        return primary
    trial_name = config.output_dir.name
    fallback = REPO_ROOT / "outputs_copy" / trial_name / "stage2"
    if fallback.is_dir() and any(fallback.glob("*.json")):
        return fallback
    return primary


def load_top_figures(stage2_path: Path, top_k: int) -> list[dict[str, Any]]:
    data = json.loads(stage2_path.read_text(encoding="utf-8"))
    items = sorted(data.get("all_scores", []), key=lambda x: x.get("score", 0), reverse=True)
    if not items:
        raise RuntimeError(f"Stage-2 结果为空: {stage2_path}")
    return items[:top_k]


def figure_by_hash(figures: list[FigureMeta]) -> dict[str, FigureMeta]:
    return {fig.image_hash: fig for fig in figures}


def strip_body_images(body_text: str) -> str:
    """移除正文中全部 markdown 图片引用；候选图仅通过 VLM 附件提供。"""
    cleaned = BODY_IMG_RE.sub("", body_text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


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


def build_candidate_list(top_items: list[dict[str, Any]]) -> tuple[str, dict[str, str]]:
    """返回候选说明文本与 C编号 -> image_hash 映射。"""
    lines: list[str] = []
    cid_to_hash: dict[str, str] = {}
    for idx, item in enumerate(top_items, start=1):
        cid = f"C{idx}"
        image_hash = item["image_hash"]
        cid_to_hash[cid] = image_hash
        caption = (item.get("caption") or "").strip()
        score = item.get("score", 0)
        lines.append(
            f"- {cid}: image_hash={image_hash}, rerank_score={score}, caption={caption or '(无图注)'}"
        )
    return "\n".join(lines), cid_to_hash


def build_multimodal_messages(
    system_prompt: str,
    user_text: str,
    figures: list[FigureMeta],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for idx, fig in enumerate(figures, start=1):
        content.append({"type": "text", "text": f"\n--- 候选图 C{idx} ---"})
        if fig.caption:
            content.append({"type": "text", "text": f"图注：{fig.caption}"})
        content.append(encode_image_part(fig.abs_image_path))
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def call_vlm_json(
    creds: ApiCredentials,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    client = build_openai_client(creds)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)


def run_method_a(
    creds: ApiCredentials,
    model: str,
    body_text: str,
    top_items: list[dict[str, Any]],
    figures: list[FigureMeta],
) -> dict[str, Any]:
    candidate_list, _ = build_candidate_list(top_items)
    user_text = USER_TEMPLATE_A.format(
        body_text=body_text,
        n=len(figures),
        candidate_list=candidate_list,
    )
    messages = build_multimodal_messages(SYSTEM_PROMPT_A, user_text, figures)
    return call_vlm_json(creds, model, messages)


def run_method_b(
    creds: ApiCredentials,
    model: str,
    abstract_text: str,
    top_items: list[dict[str, Any]],
    figures: list[FigureMeta],
) -> dict[str, Any]:
    candidate_list, _ = build_candidate_list(top_items)
    user_text = USER_TEMPLATE_B.format(
        abstract_text=abstract_text,
        n=len(figures),
        candidate_list=candidate_list,
    )
    messages = build_multimodal_messages(SYSTEM_PROMPT_B, user_text, figures)
    return call_vlm_json(creds, model, messages)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM 多模态摘要两种方式对比测试")
    parser.add_argument(
        "--paper-id",
        default=DEFAULT_PAPER_ID,
        help=f"样本 paper_id（默认 {DEFAULT_PAPER_ID}）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"取主方法重排 Top-K 候选图（默认 {DEFAULT_TOP_K}）",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"VLM 模型名（默认 {DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="结果输出目录；默认 outputs_copy/trial_31/vlm_summary_test",
    )
    parser.add_argument(
        "--method",
        choices=["both", "a", "b"],
        default="both",
        help="运行哪种方式：both / a(正文+选图) / b(摘要+插圖)",
    )
    parser.add_argument(
        "--max-body-chars",
        type=int,
        default=120000,
        help="正文最大字符数，超出则截断并标注",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(CONFIG_PATH)
    creds = resolve_api_credentials(config)
    paper_id = args.paper_id

    stage2_dir = resolve_stage2_dir(config)
    stage2_path = stage2_dir / f"{paper_id}.json"
    if not stage2_path.is_file():
        raise FileNotFoundError(f"找不到 Stage-2 结果: {stage2_path}")

    top_items = load_top_figures(stage2_path, args.top_k)
    corpus = CorpusAdapter(config)
    doc = corpus.load_document(paper_id)
    fig_map = figure_by_hash(doc.figures)

    figures: list[FigureMeta] = []
    missing: list[str] = []
    for item in top_items:
        fig = fig_map.get(item["image_hash"])
        if fig is None:
            missing.append(item["image_hash"])
            continue
        figures.append(fig)
    if missing:
        print(f"警告：{len(missing)} 张 Top-K 图在 corpus 中未找到，已跳过。")
    if not figures:
        raise RuntimeError("没有可用的候选图片。")

    body_text = strip_body_images(doc.body_text)
    if len(body_text) > args.max_body_chars:
        body_text = (
            body_text[: args.max_body_chars]
            + f"\n\n[正文已截断，原长度 {len(doc.body_text)} 字符]"
        )

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else REPO_ROOT / "outputs_copy" / "trial_31" / "vlm_summary_test"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VLM Multimodal Summary Test")
    print("=" * 60)
    print(f"config     : {CONFIG_PATH}")
    print(f"paper_id   : {paper_id}")
    print(f"model      : {args.model}")
    print(f"api        : {creds.base_url}")
    print(f"stage2     : {stage2_path}")
    print(f"top-{args.top_k}  : {[f['image_hash'][:8] for f in top_items]}")
    print(f"out_dir    : {out_dir}")
    print("=" * 60)

    result: dict[str, Any] = {
        "paper_id": paper_id,
        "model": args.model,
        "top_k": args.top_k,
        "stage2_path": str(stage2_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidates": [
            {
                "rank": i + 1,
                "candidate_id": f"C{i + 1}",
                "image_hash": item["image_hash"],
                "caption": item.get("caption", ""),
                "score": item.get("score"),
            }
            for i, item in enumerate(top_items)
        ],
        "methods": {},
    }

    if args.method in ("both", "a"):
        print("\n[Method A] 完整正文 + 候选图 → 生成摘要并选图 ...")
        method_a = run_method_a(creds, args.model, body_text, top_items, figures)
        result["methods"]["full_body_generate_and_select"] = method_a
        print("Method A 完成。")
        print(json.dumps(method_a, ensure_ascii=False, indent=2)[:2000])

    if args.method in ("both", "b"):
        print("\n[Method B] 原摘要 + 候选图 → 插入图片 ...")
        method_b = run_method_b(creds, args.model, doc.abstract_text, top_items, figures)
        result["methods"]["abstract_insert_figures"] = method_b
        print("Method B 完成。")
        print(json.dumps(method_b, ensure_ascii=False, indent=2)[:2000])

    out_path = out_dir / f"{paper_id}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
