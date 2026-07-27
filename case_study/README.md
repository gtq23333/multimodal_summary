# Stage-2 Case Study UI

PyQt6 桌面工具：左侧展示 GT 多模态标注序列，右侧 Tab 切换各方法 Top-K 选图结果，顶部展示逐方法指标。

## 为什么需要 export？

Stage-2 **eval 已经落盘**了这些内容：

| 已有产物 | 路径 | 内容 |
|---------|------|------|
| 逐方法指标 | `outputs/trial_20/eval/stage2_reranking_eval_results.csv` | IP@3、Jaccard@3 等 |
| Top-3 预测 | `outputs/trial_20/eval/stage2_reranking_diagnostics.jsonl` | 每方法仅 3 个 figure_id |
| Proposed 完整排序 | `outputs/trial_20/stage2/{paper_id}.json` | `all_scores` 含全部候选分数 |
| GT 序列 | `data/trial_20/ground_truth/{paper_id}.json` | 左栏 multimodal_sequence |

**没有落盘的是**：各 baseline 的 Top-4~10 排序（eval 内存里算过，但只写了 top3）。Case Study UI 需要把上述分散文件**聚合成一个 bundle**（`case_study/data/papers/*.json`），并解析图片绝对路径，供 PyQt6 直接读取。

因此 export 不等于「重跑 eval」，默认模式 **`from_eval` 只读已有文件，秒级完成，不下载 CLIP**。

## 依赖

```bash
cd case_study
pip install -r requirements.txt
```

项目根目录的 `src/` 与主 pipeline 依赖（pandas 等）需已安装。UI 需 PyQt6。

## 1. 导出数据（推荐默认）

```bash
cd case_study
python scripts/export_case_study_data.py --config config.yaml
```

默认 `export.mode: from_eval`：

- **Proposed**：从 `stage2/{paper_id}.json` 的 `all_scores` 读取，支持 K=3/5/10
- **各 baseline**：从 `diagnostics.jsonl` 读取 **Top-3**（K>3 时 baseline Tab 最多 3 张）
- **指标**：来自 eval CSV
- **无需** HuggingFace / CLIP / 重调 API

若需要 baseline 的 Top-5/10，使用 rerank 模式（需本地 VL/embedding 缓存）：

```bash
python scripts/export_case_study_data.py --mode rerank --skip-clip
```

`--skip-clip` 跳过 Chinese-CLIP 下载，可导出除 `Zero-shot-CLIP` 外的 7 个方法；要包含 CLIP baseline 需 `--with-clip` 且模型已缓存。

### 可选：eval 后自动 export

在 `src/configs/trial_20.yaml` 中设置：

```yaml
case_study_export: true
```

## 2. 启动 UI

```bash
cd case_study
python -m app.main
```

## 3. 验证 Top-3 与 diagnostics 一致

```bash
python scripts/verify_export.py
```

应输出 `checked=160 mismatches=0`（20 篇 × 8 方法）。

## 配置说明（`config.yaml`）

| 键 | 说明 |
|---|---|
| `export.mode` | `from_eval`（默认）或 `rerank` |
| `export.max_rank` | bundle 中每方法最多条目数 |
| `export.skip_clip` | `rerank` 模式下是否跳过 CLIP（默认 `true`） |
| `ui.default_k` | UI 默认 Top-K |

## 目录结构

```
case_study/
  config.yaml
  scripts/
    export_case_study_data.py
    verify_export.py
  data/                 # export 产物（已 gitignore）
  app/
    main.py
    main_window.py
    data_loader.py
    widgets/
```
