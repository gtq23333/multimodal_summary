#!/usr/bin/env python3
"""
将 vlm_summary_test 的 JSON 结果渲染为可浏览 HTML，占位符处 inline 插入图片。

用法:
  cd src
  python scripts/view_vlm_summary.py
  python scripts/view_vlm_summary.py --json ../outputs_copy/trial_31/vlm_summary_test/xxx.json
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import webbrowser
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parent
sys_path_inserted = False


def _ensure_import_path() -> None:
    global sys_path_inserted
    if not sys_path_inserted:
        import sys

        sys.path.insert(0, str(SRC_ROOT))
        sys_path_inserted = True


DEFAULT_JSON = (
    REPO_ROOT
    / "outputs_copy"
    / "trial_31"
    / "vlm_summary_test"
    / "2018_G_A466.pdf-f42c5f5a-ee2d-4f66-84b0-d3e32153a4e5.json"
)
CONFIG_PATH = SRC_ROOT / "configs" / "trial_31.yaml"

PLACEHOLDER_RE = re.compile(r"\[Insert Figure (C\d+)\]", re.I)

METHOD_TITLES = {
    "full_body_generate_and_select": "方式 A：完整正文 → 生成摘要并选图",
    "abstract_insert_figures": "方式 B：原摘要 → 插入图片",
}


def rel_href(from_dir: Path, target: Path) -> str:
    try:
        return Path(os.path.relpath(target.resolve(), from_dir.resolve())).as_posix()
    except ValueError:
        return target.resolve().as_uri()


def load_image_map(paper_id: str, candidates: list[dict]) -> dict[str, dict[str, str]]:
    _ensure_import_path()
    from m3sum.config import load_config
    from m3sum.data.corpus_adapter import CorpusAdapter

    config = load_config(CONFIG_PATH)
    doc = CorpusAdapter(config).load_document(paper_id)
    hash_to_fig = {fig.image_hash: fig for fig in doc.figures}

    mapping: dict[str, dict[str, str]] = {}
    for item in candidates:
        cid = item["candidate_id"]
        image_hash = item["image_hash"]
        fig = hash_to_fig.get(image_hash)
        mapping[cid.upper()] = {
            "image_hash": image_hash,
            "caption": item.get("caption") or (fig.caption if fig else ""),
            "abs_path": fig.abs_image_path if fig else "",
        }
    return mapping


def render_summary_html(summary: str, image_map: dict[str, dict[str, str]], html_dir: Path) -> str:
    parts: list[str] = []
    last = 0
    for match in PLACEHOLDER_RE.finditer(summary):
        text_chunk = summary[last : match.start()]
        if text_chunk:
            parts.append(format_text_block(text_chunk))

        cid = match.group(1).upper()
        info = image_map.get(cid)
        if info and info.get("abs_path") and Path(info["abs_path"]).is_file():
            src = rel_href(html_dir, Path(info["abs_path"]))
            caption = html.escape(info.get("caption") or cid)
            parts.append(
                f'<figure class="inserted-figure">'
                f'<div class="fig-label">{html.escape(cid)}</div>'
                f'<img src="{html.escape(src)}" alt="{caption}" loading="lazy">'
                f"<figcaption>{caption}</figcaption>"
                f"</figure>"
            )
        else:
            parts.append(
                f'<div class="missing-figure">[图片缺失: {html.escape(cid)}]</div>'
            )
        last = match.end()

    tail = summary[last:]
    if tail:
        parts.append(format_text_block(tail))
    return "\n".join(parts)


def format_text_block(text: str) -> str:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return ""
    return "\n".join(f"<p>{html.escape(p)}</p>" for p in paragraphs)


def build_html(payload: dict, html_dir: Path) -> str:
    paper_id = payload["paper_id"]
    image_map = load_image_map(paper_id, payload.get("candidates", []))
    methods = payload.get("methods", {})

    sections: list[str] = []
    for key, title in METHOD_TITLES.items():
        method = methods.get(key)
        if not method:
            continue
        summary_html = render_summary_html(
            method.get("generated_summary", ""),
            image_map,
            html_dir,
        )
        inserted = ", ".join(method.get("inserted_figures") or [])
        rationale = html.escape(method.get("rationale") or "")
        sections.append(
            f"""
<section class="method-block">
  <h2>{html.escape(title)}</h2>
  <div class="meta">插入候选：{html.escape(inserted or "无")}</div>
  <article class="summary">{summary_html}</article>
  <details class="rationale">
    <summary>选图理由</summary>
    <p>{rationale}</p>
  </details>
</section>
"""
        )

    candidate_cards = []
    for item in payload.get("candidates", []):
        cid = item["candidate_id"]
        info = image_map.get(cid.upper(), {})
        abs_path = info.get("abs_path", "")
        img_tag = ""
        if abs_path and Path(abs_path).is_file():
            src = rel_href(html_dir, Path(abs_path))
            img_tag = f'<img src="{html.escape(src)}" alt="{html.escape(cid)}">'
        candidate_cards.append(
            f"""
<div class="candidate-card">
  <div class="candidate-head">{html.escape(cid)} · score={item.get("score", "")}</div>
  {img_tag}
  <div class="candidate-caption">{html.escape(item.get("caption") or "")}</div>
</div>
"""
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VLM 多模态摘要预览 · {html.escape(paper_id[:20])}…</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --border: #e5e7eb;
      --accent: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.7;
    }}
    header, main {{ max-width: 920px; margin: 0 auto; padding: 24px 20px; }}
    header {{
      background: linear-gradient(135deg, #1d4ed8, #2563eb);
      color: #fff;
      max-width: none;
    }}
    header h1 {{ margin: 0 0 8px; font-size: 1.5rem; }}
    header .sub {{ opacity: 0.9; font-size: 0.95rem; }}
    .method-block {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px 22px;
      margin-bottom: 24px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    .method-block h2 {{ margin: 0 0 8px; font-size: 1.2rem; color: var(--accent); }}
    .meta {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 16px; }}
    .summary p {{ margin: 0 0 12px; text-align: justify; }}
    .inserted-figure {{
      margin: 18px 0;
      padding: 12px;
      border: 1px dashed #cbd5e1;
      border-radius: 10px;
      background: #fafafa;
      text-align: center;
    }}
    .fig-label {{
      display: inline-block;
      margin-bottom: 8px;
      padding: 2px 10px;
      border-radius: 999px;
      background: #dbeafe;
      color: #1d4ed8;
      font-size: 0.85rem;
      font-weight: 600;
    }}
    .inserted-figure img {{
      max-width: 100%;
      height: auto;
      border-radius: 6px;
      border: 1px solid var(--border);
    }}
    figcaption {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .missing-figure {{
      color: #b91c1c;
      background: #fef2f2;
      border: 1px solid #fecaca;
      border-radius: 8px;
      padding: 10px;
      margin: 12px 0;
    }}
    .rationale {{
      margin-top: 16px;
      padding-top: 12px;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .candidates {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px 22px;
      margin-bottom: 24px;
    }}
    .candidates h2 {{ margin-top: 0; }}
    .candidate-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 14px;
    }}
    .candidate-card {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      background: #fcfcfd;
    }}
    .candidate-head {{ font-weight: 600; font-size: 0.85rem; margin-bottom: 8px; }}
    .candidate-card img {{
      width: 100%;
      height: auto;
      border-radius: 4px;
      border: 1px solid var(--border);
    }}
    .candidate-caption {{
      margin-top: 6px;
      font-size: 0.82rem;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <header>
    <h1>VLM 多模态摘要预览</h1>
    <div class="sub">paper_id: {html.escape(paper_id)} · model: {html.escape(payload.get("model", ""))}</div>
  </header>
  <main>
    <section class="candidates">
      <h2>Top-{payload.get("top_k", "?")} 候选图</h2>
      <div class="candidate-grid">
        {"".join(candidate_cards)}
      </div>
    </section>
    {"".join(sections)}
  </main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM 多模态摘要 HTML 预览")
    parser.add_argument(
        "--json",
        default=str(DEFAULT_JSON),
        help="vlm_summary_test 结果 JSON 路径",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="输出 HTML 路径；默认与 JSON 同目录同名 .html",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="生成后用浏览器打开",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path = Path(args.json).resolve()
    if not json_path.is_file():
        raise FileNotFoundError(f"找不到结果 JSON: {json_path}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    out_path = Path(args.out).resolve() if args.out else json_path.with_suffix(".html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html_text = build_html(payload, out_path.parent)
    out_path.write_text(html_text, encoding="utf-8")

    print(f"JSON : {json_path}")
    print(f"HTML : {out_path}")
    print("用浏览器打开 HTML 文件即可查看插入效果。")

    if args.open:
        webbrowser.open(out_path.as_uri())


if __name__ == "__main__":
    main()
