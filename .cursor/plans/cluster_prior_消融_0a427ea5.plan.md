---
name: Cluster Prior 消融
overview: 为 LG-JSSF 新增基于 Chinese-CLIP 图像 embedding 与 `cluster_prior.json` 质心相似度的可配置聚类先验模块，同时实现 additive/multiplicative 两种融合策略、递增式与 drop-one 两套消融，并接入统一评估与可视化。
todos:
  - id: cluster-prior-loader
    content: 实现 ClusterPriorScorer，加载 cluster_prior.json 质心和权重并计算门控先验
    status: completed
  - id: fusion-ablation-modules
    content: 抽离融合逻辑与 ablation 配置，支持 additive/multiplicative 和模块开关
    status: completed
  - id: reranker-integration
    content: 将 cluster prior 接入 reranker，并输出完整 cluster debug 字段
    status: completed
  - id: eval-ablation-grid
    content: 扩展评估循环，生成 cluster grid search、递增式消融和 drop-one 消融结果
    status: completed
  - id: viz-ablation
    content: 扩展 HTML/PNG 可视化，加入 cluster 融合对比、消融图和 grid search 图表
    status: completed
  - id: trial10-verify
    content: 运行 trial_10，比较 LG-JSSF 与 ClusterPrior 变体的 R-Precision、MAP、MRR
    status: completed
isProject: false
---

# Cluster Prior 增量计划

## 目标

在现有 LG-JSSF Stage-2 重排基础上，引入 [`src/m3sum/cluster_prior.json`](src/m3sum/cluster_prior.json) 中的领域聚类质心与类型指导权重。该模块必须与启发式重排核心解耦，支持消融实验和融合策略对比。

## 已确认决策

- 聚类先验使用与质心一致的模型：`OFA-Sys/chinese-clip-vit-base-patch16`，复用现有 [`clip_utils.py`](src/m3sum/stage2_rerank/clip_utils.py) 的图像 embedding 与缓存。
- 融合策略同时实现两类：
  - additive：`score = lg_jssf_score + beta * cluster_prior`
  - multiplicative：`score = lg_jssf_score * (1 + beta * cluster_prior)`
- 阈值和稀释逻辑全部配置化，默认采用 Top-1 阈值 + Top1/Top2 margin 稀释。
- 默认超参不拍脑袋固定，先在 `trial_10` 上做小网格搜索：`tau ∈ {0.72, 0.75, 0.78}`，`beta ∈ {0.15, 0.25, 0.35}`。
- 消融实验实现两套：核心递增式 + drop-one。

## 聚类先验打分设计

新增 `ClusterPriorScorer`：

```python
cluster_prior_raw = sum(w_k * cosine(image_emb, centroid_k))
top1_sim = max_k cosine(image_emb, centroid_k)
margin = top1_sim - top2_sim
```

默认门控逻辑：

```python
if top1_sim < tau:
    cluster_prior = 0.0
else:
    dilution = min(1.0, max(0.0, margin / margin_tau))
    cluster_prior = cluster_prior_raw * dilution
```

设计意图：

- `tau` 防止离群图被错误赋权。
- `margin` 防止多个质心相似度接近时强行归类。
- `cluster_prior_raw` 保留人工类型权重的相对偏好。

## 模块拆分

计划把 [`reranker.py`](src/m3sum/stage2_rerank/reranker.py) 中的特征计算拆成更清晰的模块，便于消融：

- `S_direct`：query-caption 语义相似度
- `S_link`：query block 与 evidence block 最大桥接相似度
- `P_layout`：全局布局衰减先验
- `P_type`：caption keyword 启发式类型先验
- `P_cluster` / `cluster_prior`：CLIP 质心指导先验

建议新增：

- [`src/m3sum/stage2_rerank/cluster_prior.py`](src/m3sum/stage2_rerank/cluster_prior.py)
- [`src/m3sum/stage2_rerank/fusion.py`](src/m3sum/stage2_rerank/fusion.py)
- [`src/m3sum/stage2_rerank/ablation.py`](src/m3sum/stage2_rerank/ablation.py)

## 融合与消融方法名

统一评估中新增这些 Proposed 变体：

递增式：

- `DirectOnly`
- `Direct+Link`
- `Direct+Link+Layout`
- `Direct+Link+Layout+Type`
- `LG-JSSF`
- `LG-JSSF+ClusterAdd`
- `LG-JSSF+ClusterMul`

Drop-one：

- `FullClusterAdd`
- `FullClusterMul`
- `w/o S_link`
- `w/o P_layout`
- `w/o P_type`
- `w/o ClusterPrior`
- `w/o LocalWindow`

现有 baselines 保留：

- `Layout-Order`
- `Caption-BM25`
- `Caption-Dense-v4`
- `Zero-shot-CLIP`

## 配置扩展

在 [`src/configs/trial_10.yaml`](src/configs/trial_10.yaml) 增加：

```yaml
cluster_prior:
  enabled: true
  path: "../m3sum/cluster_prior.json"
  clip_model: "OFA-Sys/chinese-clip-vit-base-patch16"
  threshold_mode: "top1_margin"
  tau_grid: [0.72, 0.75, 0.78]
  beta_grid: [0.15, 0.25, 0.35]
  margin_tau: 0.03
  fusion_modes: ["additive", "multiplicative"]
```

也会在 [`config.py`](src/m3sum/config.py) 中加入对应字段与默认值。

## 评估与可视化

扩展现有 [`stage2_reranking_eval.py`](src/m3sum/eval/stage2_reranking_eval.py) 和 [`stage2_reranking_viz.py`](src/m3sum/eval/stage2_reranking_viz.py)：

- `stage2_reranking_eval_results.csv`：继续保留所有方法逐样本结果。
- 新增 `stage2_ablation_results.csv`：只保存 Proposed 变体和消融组。
- 新增 `stage2_cluster_grid_search.csv`：保存每组 `tau/beta/fusion_mode` 的平均 R-Precision、MAP、MRR。
- HTML 中新增：
  - Cluster prior 融合对比图
  - 递增式消融图
  - drop-one 消融图
  - grid search 热力图或表格

## 实现顺序

1. 读取并验证 `cluster_prior.json`：模型名、质心维度、归一化状态、权重字段。
2. 实现 `ClusterPriorScorer`：加载质心、计算 top1/top2、threshold、margin dilution、debug 输出。
3. 抽离融合逻辑到 `fusion.py`：支持 additive/multiplicative 与模块开关。
4. 改造 `reranker.py`：保留默认 LG-JSSF 行为，同时允许传入 ablation/fusion 配置。
5. 实现 ablation method 生成器：统一生成各 Proposed 变体排序结果。
6. 扩展 evaluation loop：把 cluster/add/mul、递增式、drop-one 方法纳入同一 DataFrame。
7. 扩展可视化：新增单独消融图和 grid-search 结果图。
8. 运行 `trial_10` Stage-2、评估、可视化，报告最佳融合策略和是否超过当前 LG-JSSF。

## 验收标准

- 所有聚类先验计算都使用 `cluster_prior.json` 指定的同一 Chinese-CLIP 图像 embedding。
- 每张图的 debug 输出包含：`cluster_top1_label`, `cluster_top1_sim`, `cluster_margin`, `cluster_prior_raw`, `cluster_prior`, `cluster_gate_passed`, `cluster_fusion_mode`。
- additive 与 multiplicative 两种融合策略都能跑通。
- 递增式消融和 drop-one 消融都出现在 CSV 与 HTML 可视化中。
- `stage2_cluster_grid_search.csv` 能显示不同 `tau/beta/fusion_mode` 的平均指标。
- 最终报告说明：最佳 ClusterPrior 方案相对当前 LG-JSSF 在 R-Precision、MAP、MRR 上提升、持平或下降。

## 风险

- `cluster_prior.json` 的质心来自外部路径，但 JSON 内已有 `centroid_vector`，实现会优先使用内嵌向量，避免依赖绝对路径。
- 聚类先验可能偏向视觉类型而非任务语义，因此默认作为弱先验，强度由 `beta` 和 grid search 控制。
- `trial_10` 样本量小，grid search 仅作为开发集选择依据，不能视为泛化结论。