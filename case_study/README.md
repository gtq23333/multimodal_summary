# Stage-2 Case Study UI

PyQt6 桌面工具：左侧展示 GT 多模态标注序列，右侧 Tab 切换各方法 Top-K 选图结果，顶部展示逐方法指标。

## 依赖

```bash
cd case_study
pip install -r requirements.txt
```

项目根目录的 `src/` 与主 pipeline 依赖（pandas、torch 等）需已安装。

## 1. 导出数据

在 Stage-2 eval 完成后执行（会重放各 ranker，VL/CLIP 缓存命中时较快）：

```bash
cd case_study
python scripts/export_case_study_data.py --config config.yaml
```

产物：

- `data/manifest.json` — 论文列表与方法名
- `data/papers/{paper_id}.json` — 每篇 bundle（GT 序列、sub_queries、各方法 metrics + ranked_top10）

### 可选：eval 后自动导出

在 `src/configs/trial_20.yaml`（或 local 覆盖）中设置：

```yaml
case_study_export: true
```

eval 结束时会调用同一 export 脚本。

## 2. 启动 UI

```bash
cd case_study
python -m app.main
```

可选参数：

```bash
python -m app.main --config config.yaml
```

## 3. 验证 export 与 diagnostics 一致

```bash
cd case_study
python scripts/verify_export.py
```

应输出 `checked=160 mismatches=0`（20 篇 × 8 方法，以实际 manifest 为准）。

## 配置说明（`config.yaml`）

| 键 | 说明 |
|---|---|
| `trial_config` | 指向 `src/configs/trial_20.yaml` |
| `data_dir` | export 产物目录 |
| `export.max_rank` | 每方法最多导出排名数（默认 10） |
| `export.skip_clip` | 与 eval 一致时设为 `false` |
| `ui.default_k` / `ui.k_options` | UI Top-K 默认值与可选范围 |

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
      metrics_bar.py
      gt_sequence_panel.py
      method_rank_panel.py
```
