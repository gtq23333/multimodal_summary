# AllFig vs PreRecall × 正文图片数

trial_31 专题：检验 All-Figures 相对 PreRecall（Proposed / Layout / QwenVL Top-6）的优势是否随正文图片总数变化。

## 运行

```bash
cd data_analysis_scripts/allfig_figure_count_analysis
python run_analysis.py
```

可选：`--output-dir` 指向含 `eval/stage3_ref_based_eval_results.csv` 的 trial 目录（默认 `outputs/trial_31`）。

## 输出

`reports/trial_31/report.html` — 主报告（分段表、相关、阈值对比、图表）

CSV：`paired_e2e.csv`、`paired_rag.csv`、`bin_summary_*.csv`、`correlation_*.csv`

## 指标口径

- **图片总数**：All-Figures 候选池 `pool_size`（正文全部候选图）
- **PreRecall 基线**：同策略下 Proposed/Layout/QwenVL Top-6 的逐论文 comprehensive 最高值
- **Δ**：AllFig comprehensive − Best PreRecall comprehensive
