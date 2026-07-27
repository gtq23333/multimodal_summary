# 多路融合方法判读摘要

- Fusion-RRF 相对 Proposed 的 IR@5 提升约 +0.038。
- Fusion-RRF 相对 Proposed 的 IR@6 提升约 +0.043。
- Fusion-RRF 相对 Proposed 的 IR@7 提升约 +0.051。
- Fusion-RRF IR@5 (0.636) 仍低于 Union Oracle (0.789)，差距约 0.154。
- Fusion-RRF IR@6 (0.696) 仍低于 Union Oracle (0.821)，差距约 0.126。
- Fusion-RRF IR@7 (0.725) 仍低于 Union Oracle (0.853)，差距约 0.127。
- Fusion-RRF 的 IP@3 (0.419) vs Qwen (0.441)：融合可能以精度换召回。
- 口径说明：Track A 为固定输出预算（与单方法公平对比）；Track B 的 Union Oracle 为不等预算上限，适合作为 proposal 阶段召回潜力证据。

## 固定预算

| method_name                 |   r_precision |   ip@3 |   jaccard@3 |    map |    mrr |   ir@3 |   ir@4 |   ir@5 |   ir@6 |   ir@7 |
|:----------------------------|--------------:|-------:|------------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| Fusion-Borda-PQL            |        0.4531 | 0.3871 |      0.2586 | 0.5789 | 0.7943 | 0.4463 | 0.5681 | 0.6294 | 0.681  | 0.7281 |
| Fusion-Cascade-PQL          |        0.4827 | 0.4624 |      0.3223 | 0.5715 | 0.7581 | 0.5146 | 0.5778 | 0.6152 | 0.7297 | 0.7682 |
| Fusion-RRF-PQL              |        0.4703 | 0.4194 |      0.2745 | 0.5889 | 0.8247 | 0.4675 | 0.5799 | 0.6358 | 0.6955 | 0.7254 |
| Fusion-UnionRRF-PQL         |        0.4977 | 0.4409 |      0.3024 | 0.5958 | 0.8326 | 0.5012 | 0.6205 | 0.6791 | 0.7047 | 0.7703 |
| Fusion-Weighted-PQL         |        0.4477 | 0.4409 |      0.2987 | 0.5727 | 0.7799 | 0.4917 | 0.5791 | 0.6507 | 0.6934 | 0.7536 |
| Layout-Order                |        0.4654 | 0.3978 |      0.2721 | 0.5823 | 0.8794 | 0.432  | 0.5027 | 0.5737 | 0.6382 | 0.6654 |
| Proposed                    |        0.4004 | 0.4086 |      0.2862 | 0.5426 | 0.7128 | 0.4635 | 0.5458 | 0.598  | 0.6528 | 0.6743 |
| Proposed-v2                 |        0.4149 | 0.4301 |      0.2967 | 0.5647 | 0.7678 | 0.4619 | 0.5557 | 0.6219 | 0.681  | 0.709  |
| Qwen3-VL-Rerank-ImgCap+Link |        0.5072 | 0.4409 |      0.3096 | 0.5859 | 0.7145 | 0.4942 | 0.6179 | 0.6598 | 0.7039 | 0.7472 |

## 候选池召回

| method_name         | metric_type     |   k |     ir@k |   gt_hits |   gt_total |   pool_k |   pool_gt_coverage |
|:--------------------|:----------------|----:|---------:|----------:|-----------:|---------:|-------------------:|
| Union-Oracle-PQL    | union_oracle_ir |   3 |   0.6316 |        60 |         95 |      nan |           nan      |
| Union-Oracle-PQL    | union_oracle_ir |   4 |   0.7263 |        69 |         95 |      nan |           nan      |
| Union-Oracle-PQL    | union_oracle_ir |   5 |   0.7895 |        75 |         95 |      nan |           nan      |
| Union-Oracle-PQL    | union_oracle_ir |   6 |   0.8211 |        78 |         95 |      nan |           nan      |
| Union-Oracle-PQL    | union_oracle_ir |   7 |   0.8526 |        81 |         95 |      nan |           nan      |
| Pool-Union-PQL      | pool_coverage   | nan | nan      |        86 |         95 |        8 |             0.9053 |
| Fusion-UnionRRF-PQL | fusion_pool_ir  |   3 |   0.5012 |       nan |        nan |        8 |           nan      |
| Fusion-UnionRRF-PQL | fusion_pool_ir  |   4 |   0.6205 |       nan |        nan |        8 |           nan      |
| Fusion-UnionRRF-PQL | fusion_pool_ir  |   5 |   0.6791 |       nan |        nan |        8 |           nan      |
| Fusion-UnionRRF-PQL | fusion_pool_ir  |   6 |   0.7047 |       nan |        nan |        8 |           nan      |
| Fusion-UnionRRF-PQL | fusion_pool_ir  |   7 |   0.7703 |       nan |        nan |        8 |           nan      |