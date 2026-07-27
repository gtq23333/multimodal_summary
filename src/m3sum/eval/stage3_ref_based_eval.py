from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m3sum.config import PipelineConfig
from m3sum.eval.rouge_eval import compute_rouge_scores

PLACEHOLDER_RE = re.compile(r"\[Insert Figure\s+([^\]]+)\]", re.I)
METRIC_COLS = [
    "image_precision",
    "image_recall",
    "image_f1",
    "image_ordering_score",
    "image_position_score",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "bertscore_f1",
    "comprehensive_score",
]

METHOD_ABBR = {
    "Proposed": "Prop",
    "Qwen3-VL-Rerank-ImgCap": "QwenVL",
    "Layout-Order": "Layout",
    "Reference-Oracle": "RefGT",
    "All-Figures": "AllFig",
    "Dynamic-Union-PQL": "DynUnion",
}

VARIABLE_POOL_METHODS = {"All-Figures", "Dynamic-Union-PQL"}

STRATEGY_ABBR = {
    "end_to_end_vlm": "E2E",
    "text_rag_then_rewrite": "RAG",
    "reference_oracle": "Oracle",
}

METRIC_ZH = {
    "image_precision": "Image Precision",
    "image_recall": "Image Recall",
    "image_f1": "Image F1",
    "image_ordering_score": "Image Ordering",
    "image_position_score": "Image Position",
    "pred_image_count": "Selected Images",
    "rouge_1": "ROUGE-1",
    "rouge_2": "ROUGE-2",
    "rouge_l": "ROUGE-L",
    "bertscore_f1": "BERTScore F1",
    "comprehensive_score": "Comprehensive",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_artifact_paths(config: PipelineConfig, manifest_path: Path | None = None) -> list[Path]:
    """Collect all per-paper Stage3 artifacts on disk (manifest may be partial after incremental runs)."""
    skip_names = {"manifest.json", "candidate_pools.json"}
    seen: set[Path] = set()
    paths: list[Path] = []
    for path in sorted(config.stage3_generation_dir.glob("*/*.json")):
        if path.name in skip_names:
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(path)

    if manifest_path and manifest_path.is_file():
        for raw in load_json(manifest_path).get("artifacts", []):
            candidate = _resolve_manifest_path(config, str(raw))
            if candidate.is_file():
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    paths.append(candidate)
    return paths


def _resolve_manifest_path(config: PipelineConfig, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute() and path.is_file():
        return path
    candidates = [
        Path.cwd() / path,
        config.root / path,
        config.stage3_generation_dir / path,
    ]
    raw_norm = raw.replace("\\", "/")
    marker = "outputs/trial_31/stage3_generation/"
    if marker in raw_norm:
        rel = raw_norm.split(marker, 1)[1]
        candidates.append(config.stage3_generation_dir / rel)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return path


def filter_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    pool_sizes: set[int] | None = None,
    models: set[str] | None = None,
    methods: set[str] | None = None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for artifact in artifacts:
        method = str(artifact.get("method_name", ""))
        model = str(artifact.get("model", ""))
        pool_size = int(artifact.get("pool_size", 0))
        if pool_sizes and method not in VARIABLE_POOL_METHODS and pool_size not in pool_sizes:
            continue
        if models and model not in models and method != "Reference-Oracle":
            continue
        if methods and method not in methods:
            continue
        filtered.append(artifact)
    return filtered


def load_ground_truth(config: PipelineConfig, paper_id: str) -> dict[str, Any]:
    return load_json(config.ground_truth_dir / f"{paper_id}.json")


def reference_text(gt: dict[str, Any]) -> str:
    return str(gt.get("insertion_gt", {}).get("reference_text", ""))


def reference_image_sequence(gt: dict[str, Any]) -> list[str]:
    seq = [
        str(item.get("image_hash", ""))
        for item in gt.get("insertion_gt", {}).get("multimodal_sequence", [])
        if item.get("type") == "image" and item.get("image_hash")
    ]
    if seq:
        return seq
    return [str(x) for x in gt.get("insertion_gt", {}).get("selected_hashes", [])]


def reference_positions(gt: dict[str, Any]) -> dict[str, float]:
    sequence = gt.get("insertion_gt", {}).get("multimodal_sequence", [])
    text_total = sum(len(str(item.get("content", ""))) for item in sequence if item.get("type") == "text")
    if text_total <= 0:
        refs = reference_image_sequence(gt)
        return _even_positions(refs)

    seen_text = 0
    positions: dict[str, float] = {}
    for item in sequence:
        if item.get("type") == "text":
            seen_text += len(str(item.get("content", "")))
        elif item.get("type") == "image" and item.get("image_hash"):
            positions[str(item["image_hash"])] = seen_text / text_total
    return positions


def predicted_image_sequence(artifact: dict[str, Any]) -> list[str]:
    if artifact.get("method_name") == "Reference-Oracle":
        ref_seq = artifact.get("reference_sequence") or artifact.get("candidate_pool", {}).get("reference_sequence")
        if ref_seq:
            return [
                str(item.get("image_hash", ""))
                for item in ref_seq
                if item.get("type") == "image" and item.get("image_hash")
            ]

    candidates = artifact.get("candidate_pool", {}).get("candidates", [])
    cid_to_hash = {str(c.get("candidate_id")): str(c.get("image_hash")) for c in candidates}
    summary = str(artifact.get("generated_summary", ""))
    placeholder_ids = [m.group(1).strip() for m in PLACEHOLDER_RE.finditer(summary)]
    seq = [cid_to_hash[cid] for cid in placeholder_ids if cid in cid_to_hash]
    if seq:
        return _dedupe(seq)
    return _dedupe([str(x) for x in artifact.get("selected_image_hashes") or artifact.get("inserted_figures") or []])


def predicted_positions(artifact: dict[str, Any], pred_seq: list[str]) -> dict[str, float]:
    if artifact.get("method_name") == "Reference-Oracle":
        ref_seq = artifact.get("reference_sequence") or artifact.get("candidate_pool", {}).get("reference_sequence")
        if ref_seq:
            fake_gt = {"insertion_gt": {"multimodal_sequence": ref_seq}}
            return reference_positions(fake_gt)

    candidates = artifact.get("candidate_pool", {}).get("candidates", [])
    cid_to_hash = {str(c.get("candidate_id")): str(c.get("image_hash")) for c in candidates}
    summary = str(artifact.get("generated_summary", ""))
    if not summary:
        return _even_positions(pred_seq)
    positions: dict[str, float] = {}
    denom = max(1, len(summary))
    for match in PLACEHOLDER_RE.finditer(summary):
        cid = match.group(1).strip()
        image_hash = cid_to_hash.get(cid)
        if image_hash and image_hash not in positions:
            positions[image_hash] = match.start() / denom
    if positions:
        return positions
    return _even_positions(pred_seq)


def strip_placeholders(text: str) -> str:
    return re.sub(r"\s+", " ", PLACEHOLDER_RE.sub("", text)).strip()


def image_prf(pred_seq: list[str], ref_seq: list[str]) -> tuple[float, float, float]:
    pred = set(pred_seq)
    ref = set(ref_seq)
    if not pred:
        precision = 1.0 if not ref else 0.0
    else:
        precision = len(pred & ref) / len(pred)
    recall = len(pred & ref) / len(ref) if ref else (1.0 if not pred else 0.0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def image_ordering_score(pred_seq: list[str], ref_seq: list[str]) -> float:
    if not pred_seq and not ref_seq:
        return 1.0
    if not pred_seq or not ref_seq:
        return 0.0
    dist = _levenshtein(pred_seq, ref_seq)
    return max(0.0, 1.0 - dist / max(len(pred_seq), len(ref_seq)))


def image_position_score(
    pred_seq: list[str],
    ref_seq: list[str],
    pred_pos: dict[str, float],
    ref_pos: dict[str, float],
) -> float:
    if not ref_seq:
        return 1.0 if not pred_seq else 0.0
    scores: list[float] = []
    for image_hash in ref_seq:
        if image_hash not in pred_seq:
            scores.append(0.0)
            continue
        scores.append(max(0.0, 1.0 - abs(pred_pos.get(image_hash, 0.5) - ref_pos.get(image_hash, 0.5))))
    return float(np.mean(scores)) if scores else 0.0


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    *,
    model_type: str = "bert-base-chinese",
) -> list[float | None]:
    try:
        from bert_score import score as bert_score
    except Exception:
        return [None for _ in predictions]
    _, _, f1 = bert_score(
        predictions,
        references,
        lang="zh",
        model_type=model_type,
        verbose=False,
        rescale_with_baseline=False,
    )
    return [float(x) for x in f1.tolist()]


def evaluate_artifacts(
    config: PipelineConfig,
    *,
    artifact_paths: list[Path] | None = None,
    pool_sizes: set[int] | None = None,
    models: set[str] | None = None,
    methods: set[str] | None = None,
    with_bertscore: bool = False,
) -> pd.DataFrame:
    paths = artifact_paths or resolve_artifact_paths(config)
    artifacts = [load_json(path) | {"_artifact_path": str(path)} for path in paths]
    artifacts = filter_artifacts(artifacts, pool_sizes=pool_sizes, models=models, methods=methods)

    rows: list[dict[str, Any]] = []
    pred_texts: list[str] = []
    ref_texts: list[str] = []
    for artifact in artifacts:
        gt = load_ground_truth(config, str(artifact["paper_id"]))
        ref_seq = reference_image_sequence(gt)
        pred_seq = predicted_image_sequence(artifact)
        pred_pos = predicted_positions(artifact, pred_seq)
        ref_pos = reference_positions(gt)
        precision, recall, f1 = image_prf(pred_seq, ref_seq)
        pred_text = strip_placeholders(str(artifact.get("generated_summary", "")))
        ref_text = reference_text(gt)
        pred_texts.append(pred_text)
        ref_texts.append(ref_text)
        rouge_scores = compute_rouge_scores(pred_text, ref_text) if ref_text else {
            "rouge_1": None,
            "rouge_2": None,
            "rouge_l": None,
        }
        rows.append(
            {
                "paper_id": artifact.get("paper_id", ""),
                "experiment_id": artifact.get("experiment_id", ""),
                "method_name": artifact.get("method_name", ""),
                "pool_size": artifact.get("pool_size", ""),
                "strategy": artifact.get("strategy", ""),
                "model": artifact.get("model", ""),
                "artifact_path": artifact.get("_artifact_path", ""),
                "pred_image_count": len(pred_seq),
                "ref_image_count": len(ref_seq),
                "image_precision": precision,
                "image_recall": recall,
                "image_f1": f1,
                "image_ordering_score": image_ordering_score(pred_seq, ref_seq),
                "image_position_score": image_position_score(pred_seq, ref_seq, pred_pos, ref_pos),
                "rouge_1": rouge_scores["rouge_1"],
                "rouge_2": rouge_scores["rouge_2"],
                "rouge_l": rouge_scores["rouge_l"],
                "bertscore_f1": None,
                "pred_image_sequence": "|".join(pred_seq),
                "ref_image_sequence": "|".join(ref_seq),
            }
        )

    if rows:
        bert_scores = compute_bertscore(pred_texts, ref_texts) if with_bertscore else [None] * len(rows)
        for row, score in zip(rows, bert_scores):
            row["bertscore_f1"] = score

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    score_cols = [
        "image_f1",
        "image_ordering_score",
        "image_position_score",
        "rouge_1",
        "rouge_2",
        "rouge_l",
    ]
    if "bertscore_f1" in df.columns and df["bertscore_f1"].notna().any():
        score_cols.append("bertscore_f1")
    df["comprehensive_score"] = df[score_cols].mean(axis=1, skipna=True)
    return df


def export_ref_based_results(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "stage3_ref_based_eval_results.csv"
    summary_path = out_dir / "stage3_ref_based_summary.csv"
    image_count_summary_path = out_dir / "stage3_ref_based_image_count_summary.csv"
    html_path = out_dir / "stage3_ref_based_report.html"
    df = attach_plot_labels(df)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary = build_summary(df)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    image_count_summary = build_image_count_summary(df)
    image_count_summary.to_csv(image_count_summary_path, index=False, encoding="utf-8-sig")
    label_map = build_label_mapping(df)
    fig_paths = export_figures(df, out_dir)
    html_path.write_text(
        build_html(df, summary, fig_paths, label_map, image_count_summary),
        encoding="utf-8",
    )
    return {
        "csv": csv_path,
        "summary_csv": summary_path,
        "image_count_summary_csv": image_count_summary_path,
        "html_report": html_path,
        **fig_paths,
    }


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in METRIC_COLS if c in df.columns]
    return (
        df.groupby(["method_name", "strategy", "model", "pool_size"])[metric_cols]
        .agg(["mean", "std", "count"])
        .round(4)
        .reset_index()
    )


def export_figures(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    _setup_matplotlib_zh()
    paths: dict[str, Path] = {}
    chart_specs = [
        ("image_metrics", ["image_precision", "image_recall", "image_f1"], "图像检索/插入指标"),
        ("layout_metrics", ["image_ordering_score", "image_position_score"], "图像顺序与位置指标"),
        ("text_metrics", ["rouge_1", "rouge_2", "rouge_l", "bertscore_f1"], "文本相似度指标"),
    ]
    for key, metrics, title in chart_specs:
        metrics = [m for m in metrics if m in df.columns]
        if not metrics:
            continue
        path = out_dir / f"stage3_ref_based_{key}.png"
        plot_grouped_metric_bars(df, metrics, title).savefig(
            path, dpi=150, bbox_inches="tight", facecolor="white"
        )
        plt.close()
        paths[key] = path

    path = out_dir / "stage3_ref_based_comprehensive_horizontal.png"
    plot_comprehensive_horizontal(df).savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    paths["comprehensive_horizontal"] = path

    path = out_dir / "stage3_ref_based_winner_heatmap.png"
    plot_winner_heatmap(df).savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    paths["heatmap"] = path

    path = out_dir / "stage3_ref_based_image_count_distribution.png"
    plot_image_count_distribution(df).savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    paths["image_count_distribution"] = path

    path = out_dir / "stage3_ref_based_image_count_mean_median.png"
    plot_image_count_mean_median(df).savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    paths["image_count_mean_median"] = path
    return paths


def attach_plot_labels(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["group_full"] = work.apply(_full_group_label, axis=1)
    work["group_code"] = work.apply(_short_group_code, axis=1)
    return work


def build_label_mapping(df: pd.DataFrame) -> pd.DataFrame:
    mapping = (
        df[["group_code", "group_full", "method_name", "strategy", "model", "pool_size"]]
        .drop_duplicates()
        .sort_values(["method_name", "strategy", "model"])
        .reset_index(drop=True)
    )
    return mapping


def _full_group_label(row: pd.Series) -> str:
    method = str(row.get("method_name", ""))
    strategy = str(row.get("strategy", ""))
    model = str(row.get("model", ""))
    pool = row.get("pool_size", "")
    return f"{method} | {strategy} | model={model} | top={pool}"


def _short_group_code(row: pd.Series) -> str:
    method_name = str(row.get("method_name", ""))
    method = METHOD_ABBR.get(method_name, method_name[:6])
    strategy = STRATEGY_ABBR.get(str(row.get("strategy", "")), str(row.get("strategy", ""))[:4])
    if method_name in VARIABLE_POOL_METHODS:
        return f"{method}-{strategy}"
    pool = row.get("pool_size", "")
    return f"{method}-{strategy}-T{pool}"


def _setup_matplotlib_zh() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _group_means(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby("group_code", sort=False)[metric_cols]
        .mean()
        .reindex(_ordered_group_codes(df))
    )


def _ordered_group_codes(df: pd.DataFrame) -> list[str]:
    order = (
        df.groupby("group_code")["comprehensive_score"].mean().sort_values(ascending=False).index.tolist()
        if "comprehensive_score" in df.columns
        else df["group_code"].drop_duplicates().tolist()
    )
    return order


def plot_grouped_metric_bars(df: pd.DataFrame, metric_cols: list[str], title: str) -> plt.Figure:
    means = _group_means(df, metric_cols)
    n_groups = len(means)
    n_metrics = len(metric_cols)
    fig, ax = plt.subplots(figsize=(max(8, n_groups * 0.55), 5.2))
    x = np.arange(n_groups)
    width = 0.8 / max(n_metrics, 1)
    offsets = [(i - (n_metrics - 1) / 2) * width for i in range(n_metrics)]
    colors = ["#457b9d", "#2a9d8f", "#e76f51", "#e9c46a", "#6a4c93", "#1d3557", "#f4a261"]
    for idx, metric in enumerate(metric_cols):
        ax.bar(
            x + offsets[idx],
            means[metric].values,
            width=width * 0.92,
            label=METRIC_ZH.get(metric, metric),
            color=colors[idx % len(colors)],
        )
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(means.index.tolist(), rotation=45, ha="right", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.subplots_adjust(bottom=0.28)
    return fig


def plot_comprehensive_horizontal(df: pd.DataFrame) -> plt.Figure:
    means = (
        df.groupby("group_code")["comprehensive_score"]
        .mean()
        .reindex(_ordered_group_codes(df))
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, len(means) * 0.45)))
    ax.barh(range(len(means)), means.values, color="#457b9d")
    ax.set_yticks(range(len(means)))
    ax.set_yticklabels(means.index.tolist(), fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Comprehensive Score")
    ax.set_title("综合分对比（缩写标签）", fontsize=13)
    ax.grid(axis="x", alpha=0.25)
    for idx, val in enumerate(means.values):
        ax.text(val + 0.01, idx, f"{val:.3f}", va="center", fontsize=9)
    return fig


def plot_winner_heatmap(df: pd.DataFrame) -> plt.Figure:
    pivot = df.pivot_table(
        index="paper_id",
        columns="group_code",
        values="comprehensive_score",
        aggfunc="max",
    )
    col_order = _ordered_group_codes(df)
    pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])
    winners = pivot.eq(pivot.max(axis=1), axis=0).astype(float)
    fig, ax = plt.subplots(figsize=(max(9, winners.shape[1] * 0.9), max(6, winners.shape[0] * 0.28)))
    im = ax.imshow(winners.values, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(winners.shape[1]))
    ax.set_xticklabels(winners.columns.tolist(), rotation=45, ha="right", fontsize=10)
    ax.set_yticks(np.arange(winners.shape[0]))
    ax.set_yticklabels([_short_paper_id(p) for p in winners.index], fontsize=7)
    ax.set_title("逐论文 Comprehensive Winner（缩写列名）", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.subplots_adjust(bottom=0.22)
    return fig


def build_image_count_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_code, sub in df.groupby("group_code", sort=False):
        counts = sub["pred_image_count"].astype(int)
        dist = counts.value_counts().sort_index()
        row: dict[str, Any] = {
            "group_code": group_code,
            "sample_count": int(len(sub)),
            "mean": round(float(counts.mean()), 3),
            "median": float(counts.median()),
            "min": int(counts.min()),
            "max": int(counts.max()),
            "std": round(float(counts.std(ddof=0)), 3),
            "ref_image_count_mean": round(float(sub["ref_image_count"].mean()), 3),
        }
        for image_count, freq in dist.items():
            row[f"count_{int(image_count)}"] = int(freq)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_image_count_distribution(df: pd.DataFrame) -> plt.Figure:
    groups = _ordered_group_codes(df)
    n_groups = len(groups)
    n_cols = 3
    n_rows = int(np.ceil(n_groups / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.2, n_rows * 3.4))
    axes_list = np.array(axes).reshape(-1)
    max_count = int(df["pred_image_count"].max())
    x_bins = list(range(0, max_count + 1))

    for idx, group_code in enumerate(groups):
        ax = axes_list[idx]
        sub = df[df["group_code"] == group_code]
        dist = sub["pred_image_count"].astype(int).value_counts().reindex(x_bins, fill_value=0)
        ax.bar(dist.index, dist.values, color="#457b9d", edgecolor="white", linewidth=0.6)
        mean_val = sub["pred_image_count"].mean()
        median_val = sub["pred_image_count"].median()
        ax.axvline(mean_val, color="#e76f51", linestyle="--", linewidth=1.2, label=f"均值={mean_val:.2f}")
        ax.axvline(median_val, color="#2a9d8f", linestyle="-.", linewidth=1.2, label=f"中位数={median_val:.1f}")
        ax.set_title(group_code, fontsize=10)
        ax.set_xlabel("选取图片数")
        ax.set_ylabel("论文数")
        ax.set_xticks(x_bins)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.25)

    for ax in axes_list[n_groups:]:
        ax.axis("off")

    fig.suptitle("各组选取图片数量分布（条形图 + 均值/中位数）", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def plot_image_count_mean_median(df: pd.DataFrame) -> plt.Figure:
    stats = (
        df.groupby("group_code", sort=False)["pred_image_count"]
        .agg(["mean", "median"])
        .reindex(_ordered_group_codes(df))
    )
    fig, ax = plt.subplots(figsize=(max(8, len(stats) * 0.65), 5.2))
    x = np.arange(len(stats))
    width = 0.36
    ax.bar(x - width / 2, stats["mean"], width=width, label="均值", color="#457b9d")
    ax.bar(x + width / 2, stats["median"], width=width, label="中位数", color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels(stats.index.tolist(), rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("选取图片数")
    ax.set_title("各组选取图片数量：均值 vs 中位数", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.subplots_adjust(bottom=0.28)
    return fig


def build_html(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    fig_paths: dict[str, Path],
    label_map: pd.DataFrame,
    image_count_summary: pd.DataFrame | None = None,
) -> str:
    top = (
        df.groupby(["group_code", "method_name", "strategy", "model", "pool_size"])[
            [
                "image_f1",
                "image_ordering_score",
                "image_position_score",
                "rouge_1",
                "rouge_2",
                "rouge_l",
                "bertscore_f1",
                "comprehensive_score",
            ]
        ]
        .mean()
        .sort_values("comprehensive_score", ascending=False)
        .round(4)
        .reset_index()
    )
    images = []
    chart_titles = {
        "image_metrics": "图像检索/插入指标（分组柱状图）",
        "layout_metrics": "图像顺序与位置指标（分组柱状图）",
        "text_metrics": "文本相似度指标（ROUGE-1/2/L + BERTScore）",
        "comprehensive_horizontal": "综合分横向对比（缩写标签）",
        "heatmap": "逐论文 Comprehensive Winner 热力图",
        "image_count_distribution": "各组选取图片数量分布（逐组条形图）",
        "image_count_mean_median": "各组选取图片数量均值/中位数对比",
    }
    for name, path in fig_paths.items():
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        title = chart_titles.get(name, name)
        images.append(f"<h2>{title}</h2><img src='data:image/png;base64,{data}' />")
    abbr_method = ", ".join(f"{k}={v}" for k, v in METHOD_ABBR.items())
    abbr_strategy = ", ".join(f"{k}={v}" for k, v in STRATEGY_ABBR.items())
    image_count_html = ""
    if image_count_summary is not None and not image_count_summary.empty:
        display_cols = [
            c
            for c in [
                "group_code",
                "sample_count",
                "mean",
                "median",
                "min",
                "max",
                "std",
                "ref_image_count_mean",
            ]
            if c in image_count_summary.columns
        ]
        count_cols = sorted(
            [c for c in image_count_summary.columns if c.startswith("count_")],
            key=lambda c: int(c.split("_", 1)[1]),
        )
        image_count_html = f"""
  <h2>选取图片数量统计</h2>
  <p class="note">All-Figures / Dynamic-Union-PQL 等方法在生成阶段未硬编码选图上限；候选池大小不等于最终插入数。Stage3 主生成提示词见 <code>src/m3sum/stage3_generation/generators.py</code>（E2E / RAG 重写）；旧版 <code>prompts/generation.txt</code> 含“最多 0-2 张”规则，但当前批量实验未使用该文件。</p>
  <h3>汇总（均值 / 中位数 / 范围）</h3>
  {image_count_summary[display_cols].round(3).to_html(index=False)}
  <h3>逐组频次分布（count_k = 选取 k 张图的论文数）</h3>
  {image_count_summary[display_cols[:1] + count_cols].fillna(0).astype({c: int for c in count_cols}).to_html(index=False)}
"""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Stage3 Ref-based Evaluation</title>
  <style>
    body {{ font-family: Arial, 'Microsoft YaHei', sans-serif; margin: 24px; }}
    img {{ max-width: 100%; border: 1px solid #ddd; margin-bottom: 24px; }}
    table {{ border-collapse: collapse; font-size: 13px; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; }}
    th {{ background: #f6f6f6; }}
    .note {{ color: #555; line-height: 1.6; }}
  </style>
</head>
<body>
  <h1>Stage3 Ref-based Evaluation</h1>
  <p class="note">Rows: {len(df)}; Papers: {df['paper_id'].nunique()}; Groups: {df['group_code'].nunique()}.</p>
  <p class="note">Implemented metrics: Image Precision / Recall / F1, Image Ordering, Image Position, ROUGE-1 / ROUGE-2 / ROUGE-L, BERTScore F1, Comprehensive（图像+文本指标均值）。</p>
  <p class="note">缩写规则：方法 {abbr_method}；策略 {abbr_strategy}；固定池方法格式为 <code>方法-策略-T池大小</code>（如 <code>Prop-RAG-T6</code>）；All-Figures / Dynamic-Union-PQL 为动态池，格式为 <code>方法-策略</code>（如 <code>AllFig-E2E</code>）。</p>
  <h2>标签缩写映射表</h2>
  {label_map.to_html(index=False)}
  <h2>Top Groups</h2>
  {top.to_html(index=False)}
  {image_count_html}
  {''.join(images)}
</body>
</html>"""


def _levenshtein(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        curr = [i]
        for j, y in enumerate(b, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (0 if x == y else 1)))
        prev = curr
    return prev[-1]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _even_positions(seq: list[str]) -> dict[str, float]:
    if not seq:
        return {}
    if len(seq) == 1:
        return {seq[0]: 0.5}
    return {item: i / (len(seq) - 1) for i, item in enumerate(seq)}


def _short_paper_id(paper_id: str) -> str:
    return paper_id.split(".pdf")[0] if ".pdf" in paper_id else paper_id[:18]
