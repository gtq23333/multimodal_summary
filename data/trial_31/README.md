# trial_31 数据批次

- **样本量**: 31 篇（`trial_20` 的 20 篇 + 11 篇新增标注）
- **GT 来源**: `data_annotation/annotations/*.json` → `ground_truth/`
- **对应配置**: `src/configs/trial_31.yaml`
- **输出目录**: `outputs/trial_31/`（与 `trial_10`、`trial_20` 隔离，互不覆盖）

## 新增样本（相对 trial_20）

- `2018_G_A440.pdf-065775f5-ff13-48f5-b12d-623da439971f`
- `2018_G_B334.pdf-9618d410-024a-4368-8c34-abb687df4146`
- `2023_G_A092.pdf-5db87c85-f001-44bb-9cd1-933c32b9efdf`
- `2023_G_A127.pdf-c7af97bb-63f5-4e7a-ae28-7f7b3d4f016c`
- `2023_G_A175.pdf-8d4a4c05-3206-497f-bcd2-34719146185f`
- `2023_G_B226.pdf-138a457c-a3d1-4dec-a2b6-3b5dc1265cc7`
- `2023_G_B311.pdf-4c40818a-19f0-4e8f-bea2-d2c28c9a7a93`
- `2023_G_C050.pdf-2049a35a-b3ce-49bb-a0db-827272b74086`
- `2023_G_C228.pdf-b2387ac6-a2c5-4f0e-a228-e82e75c63418`
- `2023_G_E032.pdf-a2ac9d24-a978-4750-bdfc-d48988ea3d38`
- `2023_G_E176.pdf-6a7252f2-fcac-48bd-9de1-2adf091599c5`

## 重新生成 manifest / GT

```bash
cd src
python scripts/prepare_trial_manifest.py configs/trial_31.yaml
```
