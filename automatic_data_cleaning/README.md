# automatic_data_cleaning

国赛 / 研赛 / 其它比赛 minerU `full.md` 启发式预清洗。

## 用法

```bash
cd automatic_data_cleaning
python main.py
python main.py --dry-run
python main.py --profile national_competition
python main.py --config config_graduate.yaml --dry-run
python main.py --config config_other.yaml
```

## 配置（`config*.yaml`）

| 键 | 说明 |
|---|---|
| `input_dir` | 语料目录，每个子文件夹含 `full.md` |
| `output_dir` | 输出 MD 目录（**不覆盖**已有文件） |
| `profile` | 清洗规则集 |
| `separator` | 摘要/正文分隔符，默认 `##############` |

| Profile | 输入目录 | 输出目录 |
|---|---|---|
| `national_competition` | `暂存` | `cleaned_excellent_paper_mds` |
| `graduate_competition` | `研赛` | `cleaned_graduate_paper_mds` |
| `other_competition` | `其它比赛` | `cleaned_other_paper_mds` |

## 规则概要

**研赛 (`graduate_competition`)**
- 摘要：`摘 要` / `摘要` 至 `关键词` / `关键字` / `[关键词]`
- 正文：`问题重述` / `问题背景` 等锚点至 `参考文献`
- 去除目录、承诺书、队号/队员信息、水印

**其它比赛 (`other_competition`)**
- 跳过承诺书、编号专用页等前置内容
- 四位数字开头文件名启用数学中国杯等格式增强
- 去除英文摘要、页眉页脚、队员签名等噪音

**质量校验（共用）**
- 摘要过短/过长、正文过短
- 正文残留目录（多点引导行比例过高）

## 扩展新赛事/格式

1. 在 `rules/<profile_name>/` 添加步骤模块
2. 在 `profiles/<profile_name>.py` 编排步骤
3. 在 `profiles/registry.py` 注册

## 验证

```bash
python -m pytest tests/ -q
python scripts/eval_against_gold.py
```