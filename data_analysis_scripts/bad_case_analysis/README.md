# Stage-2 Bad Case 互补性分析

离线分析 `trial_31` Stage-2 各方法的**漏召回分布是否一致**、是否存在**互补救援**，并对消融模块做 GT 图粒度的贡献量化。

## 快速运行

```bash
cd data_analysis_scripts/bad_case_analysis
python run_all.py --output-dir ../../outputs_copy/trial_31
```

分步执行：

```bash
python export_rankings.py --output-dir ../../outputs_copy/trial_31
python analyze_complementarity.py --output-dir ../../outputs_copy/trial_31
python analyze_ablation_modules.py --output-dir ../../outputs_copy/trial_31
python analyze_union_pool.py --output-dir ../../outputs_copy/trial_31
```

一键含融合与 Union 池分析：

```bash
python run_all.py --output-dir ../../outputs_copy/trial_31 --skip-export --with-union-pool --with-fusion
```

## 输入

- `outputs_copy/trial_31/stage2/*.json` — Proposed 特征与全排序
- `outputs_copy/trial_31/cache/stage2_eval/vl_rerank/` — Qwen 分数缓存
- `outputs_copy/trial_31/eval/stage2_ablation_results.csv` — 消融方法列表
- `data/trial_31/ground_truth/` — GT 图

## 输出 (`reports/trial_31/`)

| 文件 | 含义 |
|---|---|
| `artifacts/rankings.jsonl` | 各方法完整 ranked_ids |
| `gt_outcome_matrix.csv` | GT 图 × 方法 hit/rank @ K=3..7 |
| `method_pair_overlap_k*.csv` | 方法对 miss Jaccard、rescue、Cohen κ |
| `union_oracle_ir.csv` | 单方法与 Union 的 IR@K |
| `shared_hard_cases_k5.csv` | 多方法共同漏召回难例 |
| `ablation_dropone_contribution_k*.csv` | 模块 Rescue/Harm/Net |
| `union_pool/` | Union 候选池动态预算、重叠度、四象限分析 |
| `report.md` / `report.html` | 汇总报告 |

## Union 池动态预算（`union_pool/`）

| 文件 | 含义 |
|---|---|
| `union_pool_per_paper.csv` | 每篇 × 方法组 × K：actual_budget、compression、pool_gt_recall、union_gain |
| `union_pool_distribution.csv` | 组级分布汇总（median/mean/p25/p75） |
| `union_pool_pairwise_k6.csv` | 36 对 PRIMARY_METHODS 在 K=6 的候选池 Jaccard 与动态预算 |
| `union_pool_quadrant_k6.csv` | PQL / 9-way 四象限标签（冗余 / 共识 / 互补 / 分散失败） |
| `union_pool_summary.md` | 判读摘要 |

**口径**：`actual_budget = |∪ top-K|`；`nominal_budget = n_methods × K`；`union_gain = pool_gt_recall - best_single_gt_recall`。

## 判读规则

- **miss Jaccard 高**（如 >0.6）且 union IR ≈ 单方法最优 → 共同难例主导
- **miss Jaccard 低** 且双向 unique_rescue > 0、union IR 明显提升 → 方法互补，可做 candidate pool union
- 消融 **Rescue >> Harm** 的模块为有效模块；**Net ≈ 0** 为边际弱模块
