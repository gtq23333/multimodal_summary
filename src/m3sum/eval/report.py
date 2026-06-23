from __future__ import annotations

import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from m3sum.config import PipelineConfig
from m3sum.eval.insertion_metrics import aggregate_insertion, precision_recall_f1
from m3sum.eval.retrieval_metrics import aggregate_retrieval, hit_at_k, mrr
from m3sum.eval.rouge_eval import aggregate_rouge, rouge_l


def run_evaluation(config: PipelineConfig) -> dict[str, Any]:
    retrieval_results: list[dict] = []
    insertion_results: list[dict] = []
    rouge_scores: list[float] = []
    per_paper_detail: list[dict] = []

    for paper_id in config.resolved_sample_ids():
        gt_path = config.ground_truth_dir / f"{paper_id}.json"
        if not gt_path.is_file():
            continue

        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        gold_retrieval = set(gt["retrieval_gt"]["relevant_figure_hashes"])
        gold_insertion = set(gt["insertion_gt"]["selected_hashes"])

        stage2_path = config.stage2_dir / f"{paper_id}.json"
        stage3_path = config.stage3_dir / f"{paper_id}.json"

        paper_detail: dict[str, Any] = {
            "paper_id": paper_id,
            "paper_short": paper_id.split(".pdf")[0],
            "gold_figures": list(gold_retrieval),
            "pred_top3": [],
            "inserted": [],
            "gold_inserted": list(gold_insertion),
        }

        if stage2_path.is_file():
            stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
            pred_ranked = [f["image_hash"] for f in stage2.get("top3_figures", [])]
            all_ranked = [f["image_hash"] for f in stage2.get("all_scores", [])]
            top_p = config.top_p

            paper_detail["pred_top3"] = [
                {
                    "rank": f.get("rank"),
                    "image_hash": f["image_hash"][:12] + "...",
                    "caption": f.get("caption", "")[:60],
                    "score": f.get("score"),
                }
                for f in stage2.get("top3_figures", [])
            ]

            retrieval_results.append(
                {
                    "paper_id": paper_id,
                    "hit@1": hit_at_k(pred_ranked, gold_retrieval, 1),
                    "hit@3": hit_at_k(pred_ranked, gold_retrieval, 3),
                    "mrr": mrr(pred_ranked, gold_retrieval),
                    "gt_in_pool": 1.0 if any(h in gold_retrieval for h in all_ranked[:top_p]) else 0.0,
                }
            )

        if stage3_path.is_file():
            stage3 = json.loads(stage3_path.read_text(encoding="utf-8"))
            pred_ins = set(stage3.get("inserted_figures", []))
            paper_detail["inserted"] = list(pred_ins)
            ins_metrics = precision_recall_f1(pred_ins, gold_insertion)
            insertion_results.append({"paper_id": paper_id, **ins_metrics})

            ref_text = gt["insertion_gt"].get("reference_text", "")
            gen_text = stage3.get("generated_summary", "")
            if ref_text and gen_text:
                rouge_scores.append(rouge_l(gen_text, ref_text))
                paper_detail["rouge_l"] = round(rouge_l(gen_text, ref_text), 4)

        per_paper_detail.append(paper_detail)

    acceptance_rate = None
    pending = config.resolved_sample_ids()
    if config.acceptance_csv.is_file():
        rows = list(csv.DictReader(config.acceptance_csv.open(encoding="utf-8")))
        reviewed = [r for r in rows if r.get("accept", "").strip() in ("0", "1")]
        if reviewed:
            acceptance_rate = sum(int(r["accept"]) for r in reviewed) / len(reviewed)
            pending = [
                pid for pid in config.resolved_sample_ids()
                if pid not in {r["paper_id"] for r in reviewed}
            ]

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_mode": config.mode,
        "run_stage": config.stage,
        "vlm_mode": config.vlm_mode,
        "gt_mode": config.gt_mode,
        "sample_count": len(config.resolved_sample_ids()),
        "retrieval": aggregate_retrieval(retrieval_results),
        "insertion": aggregate_insertion(insertion_results),
        "generation": aggregate_rouge(rouge_scores),
        "acceptance": {
            "rate": round(acceptance_rate, 4) if acceptance_rate is not None else None,
            "pending_reviews": len(pending),
        },
        "per_paper": {
            "retrieval": retrieval_results,
            "insertion": insertion_results,
            "detail": per_paper_detail,
        },
    }

    config.eval_dir.mkdir(parents=True, exist_ok=True)
    (config.eval_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if config.export_csv_summary:
        _export_csv(config, report)
    if config.export_markdown_summary:
        _export_markdown(config, report)
    if config.export_html_report:
        _export_html(config, report)

    return report


def _export_csv(config: PipelineConfig, report: dict[str, Any]) -> None:
    path = config.eval_dir / "summary.csv"
    rows = []
    ret_map = {r["paper_id"]: r for r in report["per_paper"]["retrieval"]}
    ins_map = {r["paper_id"]: r for r in report["per_paper"]["insertion"]}
    for detail in report["per_paper"]["detail"]:
        pid = detail["paper_id"]
        r = ret_map.get(pid, {})
        i = ins_map.get(pid, {})
        rows.append(
            {
                "paper_id": detail["paper_short"],
                "hit@1": r.get("hit@1", ""),
                "hit@3": r.get("hit@3", ""),
                "mrr": r.get("mrr", ""),
                "gt_in_pool": r.get("gt_in_pool", ""),
                "insertion_p": i.get("Precision", ""),
                "insertion_r": i.get("Recall", ""),
                "insertion_f1": i.get("F1", ""),
                "rouge_l": detail.get("rouge_l", ""),
            }
        )

    if not rows:
        return

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _export_markdown(config: PipelineConfig, report: dict[str, Any]) -> None:
    path = config.eval_dir / "report.md"
    ret = report["retrieval"]
    ins = report["insertion"]
    gen = report["generation"]
    lines = [
        "# M3-Sum 试跑报告",
        "",
        f"- 生成时间: {report['generated_at']}",
        f"- 运行模式: `{report['run_mode']}` | 阶段: `{report['run_stage']}` | VLM: `{report['vlm_mode']}`",
        f"- 样本数: {report['sample_count']} | GT模式: `{report['gt_mode']}`",
        "",
        "## 汇总指标",
        "",
        "| 层级 | 指标 | 值 |",
        "|------|------|-----|",
        f"| 阶段二 图表检索 | Hit@1 | {ret.get('Hit@1', 'N/A')} |",
        f"| 阶段二 图表检索 | Hit@3 | {ret.get('Hit@3', 'N/A')} |",
        f"| 阶段二 图表检索 | MRR | {ret.get('MRR', 'N/A')} |",
        f"| 阶段二 图表检索 | GT在Top-P召回率 | {ret.get('gt_in_top_p_recall', 'N/A')} |",
        f"| 阶段三 插入决策 | Precision | {ins.get('Precision', 'N/A')} |",
        f"| 阶段三 插入决策 | Recall | {ins.get('Recall', 'N/A')} |",
        f"| 阶段三 插入决策 | F1 | {ins.get('F1', 'N/A')} |",
        f"| 阶段三 文本生成 | ROUGE-L | {gen.get('ROUGE-L', 'N/A')} |",
        "",
        "## 流程说明",
        "",
        "- **stage=2**：仅图片选择验证（检索+重排），不生成摘要",
        "- **stage=3**：仅摘要生成+插入决策（依赖 stage2 输出）",
        "- **stage=all**：完整三阶段多模态摘要管线",
        "",
        "详细 HTML 报告见 `report.html`，逐样本 CSV 见 `summary.csv`。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _export_html(config: PipelineConfig, report: dict[str, Any]) -> None:
    path = config.eval_dir / "report.html"
    ret = report["retrieval"]
    ins = report["insertion"]
    gen = report["generation"]

    detail_rows = ""
    for d in report["per_paper"]["detail"]:
        pid = html.escape(d["paper_short"])
        top3 = html.escape(", ".join(t["image_hash"] for t in d.get("pred_top3", [])) or "-")
        detail_rows += f"<tr><td>{pid}</td><td>{top3}</td><td>{html.escape(str(d.get('rouge_l', '-')))}</td></tr>\n"

    per_ret = ""
    for r in report["per_paper"]["retrieval"]:
        per_ret += (
            f"<tr><td>{html.escape(r['paper_id'][:35])}...</td>"
            f"<td>{r['hit@1']}</td><td>{r['hit@3']}</td>"
            f"<td>{r['mrr']:.4f}</td><td>{r['gt_in_pool']}</td></tr>\n"
        )

    content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>M3-Sum 试跑报告</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #f8f9fa; }}
.card {{ background: #fff; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
h1 {{ color: #1a1a2e; }}
h2 {{ color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: .3rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #dee2e6; padding: .5rem .75rem; text-align: left; }}
th {{ background: #0f3460; color: #fff; }}
.metric {{ display: inline-block; background: #e8f4fd; padding: .75rem 1.25rem; margin: .25rem; border-radius: 6px; }}
.metric b {{ font-size: 1.4rem; color: #0f3460; }}
.tag {{ display: inline-block; padding: .2rem .6rem; border-radius: 4px; font-size: .85rem; }}
.tag-dry {{ background: #fff3cd; }}
.tag-live {{ background: #d1e7dd; }}
</style>
</head>
<body>
<h1>M3-Sum 试跑报告</h1>
<p>
  <span class="tag {'tag-dry' if report['run_mode']=='dry_run' else 'tag-live'}">{report['run_mode']}</span>
  阶段: <b>{report['run_stage']}</b> |
  VLM: <b>{report['vlm_mode']}</b> |
  样本: <b>{report['sample_count']}</b> |
  生成: {html.escape(report['generated_at'][:19])}
</p>

<div class="card">
<h2>汇总指标</h2>
<div>
  <div class="metric">Hit@1<br><b>{ret.get('Hit@1', '—')}</b></div>
  <div class="metric">Hit@3<br><b>{ret.get('Hit@3', '—')}</b></div>
  <div class="metric">MRR<br><b>{ret.get('MRR', '—')}</b></div>
  <div class="metric">GT Top-P召回<br><b>{ret.get('gt_in_top_p_recall', '—')}</b></div>
  <div class="metric">Insertion F1<br><b>{ins.get('F1', '—')}</b></div>
  <div class="metric">ROUGE-L<br><b>{gen.get('ROUGE-L', '—')}</b></div>
</div>
</div>

<div class="card">
<h2>当前运行范围说明</h2>
<ul>
  <li><b>stage=1</b>：赛题子查询构建</li>
  <li><b>stage=2</b>：图表检索 + 布局重排（<b>图片选择验证</b>，不含摘要生成）</li>
  <li><b>stage=3</b>：VLM描述 + 摘要改写 + 插入决策</li>
  <li><b>stage=all</b>：上述三阶段完整多模态摘要管线</li>
  <li><b>dry_run</b>：零 API 链路冒烟；<b>live</b>：真实 OpenAI 调用</li>
</ul>
<p>你当前的 dry-run + stage=all 跑的是<b>完整三阶段流程</b>，但 API 用 mock 数据，指标仅供链路验证。</p>
</div>

<div class="card">
<h2>逐样本检索指标</h2>
<table>
<tr><th>paper_id</th><th>Hit@1</th><th>Hit@3</th><th>MRR</th><th>GT∈Top-P</th></tr>
{per_ret}
</table>
</div>

<div class="card">
<h2>逐样本 Top-3 预测</h2>
<table>
<tr><th>paper_id</th><th>Top-3 图表 hash</th><th>ROUGE-L</th></tr>
{detail_rows}
</table>
</div>
</body>
</html>"""
    path.write_text(content, encoding="utf-8")
