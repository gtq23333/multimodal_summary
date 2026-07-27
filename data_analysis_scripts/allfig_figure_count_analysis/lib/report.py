from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .visualize import encode_image


def _interpretation_block(
    paired_e2e: pd.DataFrame,
    paired_rag: pd.DataFrame,
    corr_e2e: pd.DataFrame,
    corr_rag: pd.DataFrame,
    thresh_e2e: pd.DataFrame,
    thresh_rag: pd.DataFrame,
) -> str:
    lines = ["<h2>判读摘要</h2>", "<ul>"]

    for label, paired, corr, thresh in [
        ("E2E", paired_e2e, corr_e2e, thresh_e2e),
        ("RAG", paired_rag, corr_rag, thresh_rag),
    ]:
        if paired.empty:
            continue
        win_rate = paired["allfig_wins"].mean()
        delta_mean = paired["delta_allfig_minus_best"].mean()
        lines.append(
            f"<li><strong>{label}</strong>：AllFig 整体胜率 {win_rate:.1%}，"
            f"平均 Δ(AllFig−BestPreRecall)={delta_mean:+.3f}。"
        )
        if not corr.empty:
            row = corr[corr["column"] == "delta_allfig_minus_best"]
            if not row.empty:
                rho = row.iloc[0]["spearman_rho"]
                pval = row.iloc[0]["p_value"]
                direction = "图片越多优势越小" if rho < 0 else "图片越多优势越大"
                sig = "（显著）" if pval < 0.05 else "（不显著）"
                lines.append(
                    f" Δ 与图片数的 Spearman ρ={rho:.3f}, p={pval:.4f}{sig}，"
                    f"趋势上{direction}。"
                )
        if len(thresh) == 2:
            low, high = thresh.iloc[0], thresh.iloc[1]
            lines.append(
                f" ≤30 张：Δ 均值 {low['delta_mean']:+.3f}，胜率 {low['allfig_win_rate']:.1%}；"
                f"&gt;30 张：Δ 均值 {high['delta_mean']:+.3f}，胜率 {high['allfig_win_rate']:.1%}。"
            )
        lines.append("</li>")

    lines.append("</ul>")
    lines.append(
        "<p class='note'>说明：PreRecall 基线为同策略下 Proposed / Layout / QwenVL 的 Top-6 三组中"
        "逐论文 comprehensive 最高者；图片总数取 All-Figures 候选池大小（=正文全部候选图）。"
        "若 ρ&lt;0 且高图数段胜率下降，则支持「图多 AllFig 相对更弱」假设；否则为全线优势或混杂。</p>"
    )
    return "\n".join(lines)


def build_html_report(
    out_path: Path,
    *,
    paired_e2e: pd.DataFrame,
    paired_rag: pd.DataFrame,
    bin_e2e: pd.DataFrame,
    bin_rag: pd.DataFrame,
    corr_e2e: pd.DataFrame,
    corr_rag: pd.DataFrame,
    thresh_e2e: pd.DataFrame,
    thresh_rag: pd.DataFrame,
    figures: dict[str, Path],
) -> None:
    imgs = []
    titles = {
        "e2e_score_vs_figures": "E2E：Comprehensive vs 图片数",
        "e2e_delta_scatter": "E2E：AllFig 优势 vs 图片数",
        "e2e_delta_bars": "E2E：逐论文 Δ",
        "e2e_binned_means": "E2E：分段均值",
        "e2e_win_rate": "E2E：分段胜率",
        "e2e_prerecall_breakdown": "E2E：分段 vs 各 PreRecall",
        "e2e_image_f1_bins": "E2E：Image F1 分段",
        "e2e_rouge_l_bins": "E2E：ROUGE-L 分段",
        "rag_score_vs_figures": "RAG：Comprehensive vs 图片数",
        "rag_delta_scatter": "RAG：AllFig 优势 vs 图片数",
        "rag_delta_bars": "RAG：逐论文 Δ",
        "rag_binned_means": "RAG：分段均值",
        "rag_win_rate": "RAG：分段胜率",
        "rag_prerecall_breakdown": "RAG：分段 vs 各 PreRecall",
        "rag_image_f1_bins": "RAG：Image F1 分段",
        "rag_rouge_l_bins": "RAG：ROUGE-L 分段",
    }
    for key, path in figures.items():
        if not path.is_file():
            continue
        title = titles.get(key, key)
        b64 = encode_image(path)
        imgs.append(f"<h2>{title}</h2><img src='data:image/png;base64,{b64}' alt='{title}' />")

    interp = _interpretation_block(
        paired_e2e, paired_rag, corr_e2e, corr_rag, thresh_e2e, thresh_rag
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>AllFig vs PreRecall × 图片数 — trial_31</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 24px; color: #222; max-width: 1200px; }}
    h1 {{ border-bottom: 2px solid #457b9d; padding-bottom: 8px; }}
    img {{ max-width: 100%; border: 1px solid #ddd; margin: 12px 0 28px; }}
    table {{ border-collapse: collapse; font-size: 13px; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
    th {{ background: #f6f6f6; text-align: center; }}
    td:first-child, th:first-child {{ text-align: left; }}
    .note {{ color: #555; line-height: 1.65; }}
    section {{ margin-bottom: 32px; }}
  </style>
</head>
<body>
  <h1>AllFig vs PreRecall 专题：正文图片数量的调节效应</h1>
  <p class="note">trial_31 · 31 篇 · 指标来源 stage3_ref_based_eval · 对比 AllFig-E2E/RAG 与同策略 Top-6 PreRecall（Proposed / Layout / QwenVL）。</p>

  <section>
    <h2>分析设计</h2>
    <ul class="note">
      <li><strong>自变量</strong>：正文图片总数（All-Figures 候选池大小，5–44）。</li>
      <li><strong>因变量</strong>：ref-based comprehensive（图像+文本指标均值）；并分解 image_f1、rouge_l。</li>
      <li><strong>配对基线</strong>：同策略、同论文下 PreRecall 三组 comprehensive 的逐论文最大值（Best PreRecall）。</li>
      <li><strong>分段</strong>：≤10 / 11–15 / 16–20 / 21–30 / 31+ 张。</li>
      <li><strong>假设检验</strong>：Spearman(图片数, AllFig−BestPreRecall)；以及 ≤30 vs &gt;30 分段胜率/均值对比。</li>
    </ul>
  </section>

  {interp}

  <section>
    <h2>E2E 逐论文配对表（节选 Top/Bottom Δ）</h2>
    {_table_top_bottom(paired_e2e)}
  </section>

  <section>
    <h2>RAG 逐论文配对表（节选 Top/Bottom Δ）</h2>
    {_table_top_bottom(paired_rag)}
  </section>

  <section>
    <h2>E2E 分段汇总</h2>
    {bin_e2e.to_html(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x)}
  </section>

  <section>
    <h2>RAG 分段汇总</h2>
    {bin_rag.to_html(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x)}
  </section>

  <section>
    <h2>Spearman 相关（E2E）</h2>
    {corr_e2e.to_html(index=False) if not corr_e2e.empty else "<p>样本不足</p>"}
    <h2>Spearman 相关（RAG）</h2>
    {corr_rag.to_html(index=False) if not corr_rag.empty else "<p>样本不足</p>"}
  </section>

  <section>
    <h2>30 张阈值对比（E2E）</h2>
    {thresh_e2e.to_html(index=False) if not thresh_e2e.empty else "<p>—</p>"}
    <h2>30 张阈值对比（RAG）</h2>
    {thresh_rag.to_html(index=False) if not thresh_rag.empty else "<p>—</p>"}
  </section>

  <section>
    <h2>可视化</h2>
    {''.join(imgs)}
  </section>
</body>
</html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def _table_top_bottom(paired: pd.DataFrame, n: int = 8) -> str:
    if paired.empty:
        return "<p>无数据</p>"
    cols = [
        "paper_id",
        "total_figure_count",
        "allfig_score",
        "best_prerecall_method",
        "best_prerecall_score",
        "delta_allfig_minus_best",
        "allfig_wins",
    ]
    top = paired.nlargest(n, "delta_allfig_minus_best")[cols]
    bottom = paired.nsmallest(n, "delta_allfig_minus_best")[cols]
    return (
        "<h3>AllFig 优势最大</h3>"
        + top.to_html(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x)
        + "<h3>AllFig 劣势最大</h3>"
        + bottom.to_html(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    )
