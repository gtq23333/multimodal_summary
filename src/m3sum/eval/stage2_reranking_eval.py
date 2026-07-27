from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from m3sum.clients.openai_embedder import OpenAIEmbedder
from m3sum.config import PipelineConfig, resolve_api_credentials
from m3sum.eval.stage2_rerank_metrics import (
    average_precision,
    compute_mrr,
    image_precision_at_k,
    image_recall_at_k,
    image_recall_at_ks,
    jaccard_at_k,
    maxsim_at_k,
    r_precision,
)
from m3sum.stage2_rerank.baselines.base import RankedFigure, Stage2Ranker
from m3sum.stage2_rerank.baselines.caption_bm25 import CaptionBM25Ranker
from m3sum.stage2_rerank.baselines.caption_dense import CaptionDenseRanker
from m3sum.stage2_rerank.baselines.layout_order import LayoutOrderRanker
from m3sum.stage2_rerank.baselines.proposed import ProposedRanker
from m3sum.stage2_rerank.baselines.proposed_v2 import ProposedV2Ranker
from m3sum.stage2_rerank.baselines.qwen3_vl_rerank import (
    Qwen3VLRerankImgCapLinkRanker,
    Qwen3VLRerankImgCapRanker,
    Qwen3VLRerankImgRanker,
    build_vl_rerank_client,
)
from m3sum.stage2_rerank.figure_link_context import build_figure_link_context_selector
from m3sum.stage2_rerank.baselines.zeroshot_clip import ZeroshotClipRanker
from m3sum.stage2_rerank.clip_utils import ClipImageEmbeddingCache, load_clip_model
from m3sum.stage2_rerank.figure_number import parse_figure_number
from m3sum.stage2_rerank.sample_loader import load_all_stage2_samples

logger = logging.getLogger(__name__)

VL_RERANK_METHODS = frozenset(
    {
        "Qwen3-VL-Rerank-Img",
        "Qwen3-VL-Rerank-ImgCap",
        "Qwen3-VL-Rerank-ImgCap+Link",
        "Qwen3-VL-Rerank",
    }
)

METHOD_LABELS = {
    "Proposed": "Proposed",
    "Proposed-v2": "Proposed-v2",
    "Qwen3-VL-Rerank-Img": "Qwen3-VL-Rerank-Img",
    "Qwen3-VL-Rerank-ImgCap+Link": "Qwen3-VL-Rerank-ImgCap+Link",
    "Qwen3-VL-Rerank-ImgCap": "Qwen3-VL-Rerank-ImgCap",
    "Layout-Order": "Layout-Order",
    "Caption-BM25": "Caption-BM25",
    "Caption-Dense-v4": "Caption-Dense-v4",
    "Zero-shot-CLIP": "Zero-shot-CLIP",
}

ZH_COLUMNS = {
    "paper_id": "论文ID",
    "method_name": "方法名称",
    "r_precision": "R-Precision",
    "ip@3": "Image Precision (IP@3)",
    "ir@3": "Image Recall (IR@3)",
    "jaccard@3": "Jaccard@3",
    "maxsim@3": "MaxSim@3",
    "map": "AP/MAP",
    "mrr": "MRR",
}


def _recall_k_column(k: int) -> str:
    return f"ir@{k}"


def _build_zh_columns(recall_ks: list[int]) -> dict[str, str]:
    cols = dict(ZH_COLUMNS)
    for k in recall_ks:
        col = _recall_k_column(k)
        cols[col] = f"Image Recall (IR@{k})"
    return cols


def _is_flowchart_caption(caption: str) -> bool:
    keywords = ("流程图", "flowchart", "flow chart", "示意图")
    lower = caption.lower()
    return any(k in caption or k in lower for k in keywords)


def _figure_lookup(sample) -> dict[str, Any]:
    return {f.image_hash: f for f in sample.figures}


def resolve_stage2_eval_methods(
    config: PipelineConfig,
    methods: list[str] | None = None,
    *,
    vl_rerank_only: bool = False,
) -> list[str]:
    """解析本次评估要运行的方法列表（保持 config 中的顺序）。"""
    configured = list(config.stage2_eval_methods)
    if vl_rerank_only:
        selected = {m for m in configured if m in VL_RERANK_METHODS}
        if not selected:
            raise ValueError(
                "配置 stage2_eval.methods 中未包含任何 Qwen3-VL-Rerank 方法"
            )
        return [m for m in configured if m in selected]

    if methods is None:
        return configured

    requested = set(methods)
    unknown = requested - set(configured)
    if unknown:
        raise ValueError(
            f"未在配置 stage2_eval.methods 中声明的方法: {sorted(unknown)}"
        )
    return [m for m in configured if m in requested]


def build_stage2_rankers(
    config: PipelineConfig,
    skip_clip: bool,
    active_methods: list[str] | None = None,
) -> tuple[dict[str, Stage2Ranker], ClipImageEmbeddingCache | None, Any]:
    """构建 method_name -> ranker 映射，并初始化 CLIP 缓存（若需要）。"""
    methods = set(
        active_methods if active_methods is not None else config.stage2_eval_methods
    )
    return _build_rankers(config, skip_clip=skip_clip, active_methods=methods)


def _build_rankers(
    config: PipelineConfig,
    skip_clip: bool,
    active_methods: set[str] | None = None,
) -> tuple[dict[str, Stage2Ranker], ClipImageEmbeddingCache | None, Any]:
    """构建 method_name -> ranker 映射，并初始化 CLIP 缓存（若需要）。"""
    methods = active_methods or set(config.stage2_eval_methods)
    rankers: dict[str, Stage2Ranker] = {}
    dry_run = config.dry_run
    clip_cache: ClipImageEmbeddingCache | None = None
    clip_encoder = None

    need_clip_cache = not skip_clip and (
        "Zero-shot-CLIP" in methods
        or "Proposed" in methods
        or "Proposed-v2" in methods
        or bool(methods & VL_RERANK_METHODS)
    )

    if need_clip_cache:
        if dry_run:
            clip_cache = ClipImageEmbeddingCache(
                config.stage2_eval_clip_cache_dir,
                clip_encoder=None,
                dry_run=True,
            )
        else:
            clip_encoder = load_clip_model(config.stage2_eval_clip_model)
            clip_cache = ClipImageEmbeddingCache(
                config.stage2_eval_clip_cache_dir,
                clip_encoder=clip_encoder,
                dry_run=False,
            )

    if "Layout-Order" in methods:
        rankers["Layout-Order"] = LayoutOrderRanker()
    if "Caption-BM25" in methods:
        rankers["Caption-BM25"] = CaptionBM25Ranker(
            query_use_keywords=config.query_use_keywords,
        )

    if "Proposed" in methods:
        rankers["Proposed"] = ProposedRanker(
            config,
            dry_run=dry_run,
            image_cache=clip_cache,
        )

    if "Proposed-v2" in methods:
        rankers["Proposed-v2"] = ProposedV2Ranker(
            config,
            dry_run=dry_run,
            image_cache=clip_cache,
        )

    if "Caption-Dense-v4" in methods:
        embedder = None
        if not dry_run:
            creds = resolve_api_credentials(config)
            embedder = OpenAIEmbedder(config.embed_model, creds)
        rankers["Caption-Dense-v4"] = CaptionDenseRanker(
            config.stage2_eval_text_cache_dir,
            embedder,
            dry_run=dry_run,
            query_use_keywords=config.query_use_keywords,
        )

    if "Zero-shot-CLIP" in methods and not skip_clip:
        rankers["Zero-shot-CLIP"] = ZeroshotClipRanker(
            clip_encoder=clip_encoder,
            image_cache_dir=config.stage2_eval_clip_cache_dir,
            dry_run=dry_run,
            query_use_keywords=config.query_use_keywords,
        )

    vl_kw = {"query_use_keywords": config.query_use_keywords}
    vl_img_methods = {"Qwen3-VL-Rerank-Img", "Qwen3-VL-Rerank"}
    if vl_img_methods & methods:
        vl_client = build_vl_rerank_client(config, dry_run=dry_run, img_cap=False)
        rankers["Qwen3-VL-Rerank-Img"] = Qwen3VLRerankImgRanker(
            config.stage2_eval_vl_rerank_cache_dir,
            vl_client,
            dry_run=dry_run,
            **vl_kw,
        )

    if "Qwen3-VL-Rerank-ImgCap+Link" in methods:
        vl_link_client = build_vl_rerank_client(
            config, dry_run=dry_run, img_cap_link=True
        )
        context_selector = build_figure_link_context_selector(config, dry_run=dry_run)
        rankers["Qwen3-VL-Rerank-ImgCap+Link"] = Qwen3VLRerankImgCapLinkRanker(
            config.stage2_eval_vl_rerank_cache_dir,
            vl_link_client,
            context_selector,
            dry_run=dry_run,
            **vl_kw,
        )

    if "Qwen3-VL-Rerank-ImgCap" in methods:
        vl_cap_client = build_vl_rerank_client(config, dry_run=dry_run, img_cap=True)
        rankers["Qwen3-VL-Rerank-ImgCap"] = Qwen3VLRerankImgCapRanker(
            config.stage2_eval_vl_rerank_cache_dir,
            vl_cap_client,
            dry_run=dry_run,
            **vl_kw,
        )

    return rankers, clip_cache, clip_encoder


def _load_diagnostics_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _merge_eval_results(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    methods_run: set[str],
) -> pd.DataFrame:
    if existing.empty:
        return new
    kept = existing[~existing["method_name"].isin(methods_run)]
    return pd.concat([kept, new], ignore_index=True)


def _merge_diagnostics_records(
    existing: list[dict[str, Any]],
    new: list[dict[str, Any]],
    methods_run: set[str],
) -> list[dict[str, Any]]:
    kept = [r for r in existing if r.get("method_name") not in methods_run]
    return kept + new


def _detect_diagnostic_flags(
    method_name: str,
    ranked: list[RankedFigure],
    sample,
    metrics: dict[str, float],
    score_by_id: dict[str, float],
    clip_scores_by_paper: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """检测疑似 failure case 并返回诊断记录。"""
    flags: list[dict[str, Any]] = []
    hash_to_fig = _figure_lookup(sample)
    gold = sample.ground_truth_ids
    jaccard_k = metrics.get("jaccard@3", 0.0)
    maxsim_k = metrics.get("maxsim@3", 0.0)

    if maxsim_k > 0.5 and jaccard_k < 0.3:
        flags.append(
            {
                "type": "high_maxsim_low_jaccard",
                "message": "MaxSim@3 较高但 Jaccard@3 较低，可能存在视觉等价图",
                "maxsim@3": maxsim_k,
                "jaccard@3": jaccard_k,
            }
        )

    if method_name == "Zero-shot-CLIP":
        top5 = ranked[:5]
        top5_ids = {r.figure_id for r in top5}
        for fig in sample.figures:
            if _is_flowchart_caption(fig.caption) and fig.image_hash not in top5_ids:
                rank_pos = next(
                    (r.rank for r in ranked if r.figure_id == fig.image_hash),
                    None,
                )
                flags.append(
                    {
                        "type": "flowchart_ranked_low",
                        "message": "Zero-shot CLIP 将 flowchart 类 figure 排在 top-5 之外",
                        "figure_id": fig.image_hash,
                        "caption": fig.caption[:80],
                        "rank": rank_pos,
                        "in_ground_truth": fig.image_hash in gold,
                    }
                )

    if method_name in ("Caption-BM25", "Caption-Dense-v4"):
        paper_clip = clip_scores_by_paper.get(sample.paper_id, {})
        for fig in sample.figures:
            if fig.image_hash not in gold:
                continue
            cap_score = score_by_id.get(fig.image_hash, 0.0)
            clip_score = paper_clip.get(fig.image_hash, 0.0)
            if cap_score > clip_score + 0.05:
                flags.append(
                    {
                        "type": "caption_beats_clip",
                        "message": "Caption baseline 在 caption-heavy GT figure 上优于 CLIP",
                        "figure_id": fig.image_hash,
                        "caption": fig.caption[:80],
                        "caption_score": cap_score,
                        "clip_score": clip_score,
                    }
                )

    return flags


def _log_clip_diagnostics(
    sample,
    ranked: list[RankedFigure],
    score_by_id: dict[str, float],
) -> list[dict[str, Any]]:
    """Zero-shot CLIP 额外诊断：top-5 详情。"""
    hash_to_fig = _figure_lookup(sample)
    gold = sample.ground_truth_ids
    queries = [q.query for q in sample.sub_queries]
    entries: list[dict[str, Any]] = []

    logger.info("  [CLIP 诊断] paper_id=%s", sample.paper_id)
    logger.info("  query list: %s", queries)

    for item in ranked[:5]:
        fig = hash_to_fig.get(item.figure_id)
        caption = fig.caption if fig else ""
        fig_num = parse_figure_number(caption) if caption else None
        source_type = fig.source_type if fig else ""
        img_path = fig.abs_image_path if fig else ""
        hit = item.figure_id in gold
        logger.info(
            "    top-%d | figure_id=%s | figure_number=%s | caption=%s | "
            "score=%.4f | source_type=%s | hit_gt=%s | path=%s",
            item.rank,
            item.figure_id[:16] + "...",
            fig_num,
            (caption[:40] + "...") if len(caption) > 40 else caption,
            item.score,
            source_type,
            hit,
            img_path,
        )
        entries.append(
            {
                "rank": item.rank,
                "figure_id": item.figure_id,
                "figure_number": fig_num,
                "caption_snippet": caption[:80],
                "image_path": img_path,
                "clip_score": item.score,
                "source_type": source_type,
                "hit_ground_truth": hit,
                "is_flowchart_heuristic": _is_flowchart_caption(caption),
            }
        )
    return entries


def run_stage2_reranking_eval(
    config: PipelineConfig,
    skip_clip: bool = False,
    *,
    methods: list[str] | None = None,
    vl_rerank_only: bool = False,
    merge_results: bool | None = None,
    run_secondary: bool | None = None,
) -> pd.DataFrame:
    """
    对 trial 样本运行 Stage-2 baselines + Proposed 评估。
    返回英文列名 DataFrame，并写入 eval 目录 CSV。

    增量评估：指定 methods 或 vl_rerank_only 时，默认只重跑对应方法，
    并与已有 stage2_reranking_eval_results.csv / diagnostics.jsonl 合并。
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    methods_to_run = resolve_stage2_eval_methods(
        config,
        methods=methods,
        vl_rerank_only=vl_rerank_only,
    )
    partial_eval = set(methods_to_run) != set(config.stage2_eval_methods)
    if merge_results is None:
        merge_results = partial_eval
    if run_secondary is None:
        run_secondary = not partial_eval

    if partial_eval:
        logger.info(
            "增量评估：仅运行 %d 个方法 %s",
            len(methods_to_run),
            methods_to_run,
        )
        if merge_results:
            logger.info("将与已有 eval 结果合并（--no-merge-results 可关闭）")
    else:
        logger.info(
            "检索 query 配置：query_use_keywords=%s",
            config.query_use_keywords,
        )

    samples = load_all_stage2_samples(config)
    if not samples:
        logger.warning("无有效 Stage-2 样本，评估终止")
        return pd.DataFrame()

    rankers, clip_cache, _ = _build_rankers(
        config,
        skip_clip=skip_clip,
        active_methods=set(methods_to_run),
    )
    jaccard_k = config.stage2_eval_jaccard_k
    maxsim_k = config.stage2_eval_maxsim_k
    recall_ks = config.stage2_eval_recall_ks
    zh_columns = _build_zh_columns(recall_ks)

    rows: list[dict[str, Any]] = []
    diag_records: list[dict[str, Any]] = []
    diagnostics_path = config.eval_dir / "stage2_reranking_diagnostics.jsonl"
    config.eval_dir.mkdir(parents=True, exist_ok=True)
    methods_run_set = set(methods_to_run)

    clip_ranker = rankers.get("Zero-shot-CLIP")
    clip_scores_by_paper: dict[str, dict[str, float]] = {}
    if clip_ranker is not None:
        for sample in samples:
            clip_ranked = clip_ranker.rank(sample)
            clip_scores_by_paper[sample.paper_id] = {
                r.figure_id: r.score for r in clip_ranked
            }

    for method_name in methods_to_run:
        if method_name not in rankers:
            if method_name == "Zero-shot-CLIP" and skip_clip:
                logger.info("跳过方法 Zero-shot-CLIP（--skip-clip）")
            else:
                logger.warning("未知或未启用的方法: %s", method_name)
            continue

        ranker = rankers[method_name]
        logger.info("======== 方法: %s ========", method_name)

        for sample in samples:
            ranked = ranker.rank(sample)
            ranked_ids = [r.figure_id for r in ranked]
            score_by_id = {r.figure_id: r.score for r in ranked}
            gold = sample.ground_truth_ids

            rp = r_precision(ranked_ids, gold)
            ip = image_precision_at_k(ranked_ids, gold, k=jaccard_k)
            ir = image_recall_at_k(ranked_ids, gold, k=jaccard_k)
            ir_by_k = image_recall_at_ks(ranked_ids, gold, recall_ks)
            jac = jaccard_at_k(ranked_ids, gold, k=jaccard_k)
            ap = average_precision(ranked_ids, gold)
            mrr_val = compute_mrr(ranked_ids, gold)

            if skip_clip or clip_cache is None:
                ms = float("nan")
            else:
                ms = maxsim_at_k(
                    ranked_ids,
                    gold,
                    sample.figures,
                    clip_cache,
                    sample.paper_id,
                    k=maxsim_k,
                )

            top3 = ranked_ids[:3]
            ir_extra = " | ".join(
                f"IR@{k}={ir_by_k[k]:.4f}" for k in recall_ks
            )
            logger.info(
                "  paper_id=%s | method=%s | 候选数=%d | GT=%s | top3=%s | "
                "R-Precision=%.4f | IP@%d=%.4f | IR@%d=%.4f | %s | Jaccard@%d=%.4f | MaxSim@%d=%s | AP=%.4f | MRR=%.4f",
                sample.paper_id,
                method_name,
                len(sample.figures),
                [h[:12] + "..." for h in sorted(gold)],
                [h[:12] + "..." for h in top3],
                rp,
                jaccard_k,
                ip,
                jaccard_k,
                ir,
                ir_extra,
                jaccard_k,
                jac,
                maxsim_k,
                f"{ms:.4f}" if ms == ms else "N/A",
                ap,
                mrr_val,
            )

            clip_diag: list[dict[str, Any]] = []
            if method_name == "Zero-shot-CLIP":
                clip_diag = _log_clip_diagnostics(sample, ranked, score_by_id)

            flags = _detect_diagnostic_flags(
                method_name,
                ranked,
                sample,
                {"jaccard@3": jac, "maxsim@3": ms},
                score_by_id,
                clip_scores_by_paper,
            )
            for flag in flags:
                logger.warning("  [疑似 case] %s", flag.get("message", flag))

            diag_metrics: dict[str, Any] = {
                "r_precision": rp,
                f"ip@{jaccard_k}": ip,
                f"ir@{jaccard_k}": ir,
                f"jaccard@{jaccard_k}": jac,
                f"maxsim@{maxsim_k}": ms if ms == ms else None,
                "map": ap,
                "mrr": mrr_val,
            }
            for k, score in ir_by_k.items():
                diag_metrics[_recall_k_column(k)] = score

            diag_record = {
                "paper_id": sample.paper_id,
                "method_name": method_name,
                "metrics": diag_metrics,
                "top3_predicted": top3,
                "ground_truth": list(gold),
                "clip_top5": clip_diag,
                "flags": flags,
            }
            debug_for_sample = getattr(ranker, "debug_for_sample", None)
            if callable(debug_for_sample):
                diag_record["feature_breakdown"] = debug_for_sample(sample)
            diag_records.append(diag_record)

            row: dict[str, Any] = {
                "paper_id": sample.paper_id,
                "method_name": method_name,
                "r_precision": round(rp, 6),
                "ip@3": round(ip, 6),
                "ir@3": round(ir, 6),
                "jaccard@3": round(jac, 6),
                "maxsim@3": round(ms, 6) if ms == ms else None,
                "map": round(ap, 6),
                "mrr": round(mrr_val, 6),
            }
            for k, score in ir_by_k.items():
                row[_recall_k_column(k)] = round(score, 6)
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("评估未产生任何结果行")
        return df

    en_path = config.eval_dir / "stage2_reranking_eval_results.csv"
    zh_path = config.eval_dir / "stage2_reranking_eval_results_zh.csv"

    if merge_results:
        existing_df = pd.read_csv(en_path) if en_path.is_file() else pd.DataFrame()
        df = _merge_eval_results(existing_df, df, methods_run_set)
        existing_diag = _load_diagnostics_records(diagnostics_path)
        diag_records = _merge_diagnostics_records(
            existing_diag,
            diag_records,
            methods_run_set,
        )

    df.to_csv(en_path, index=False, encoding="utf-8-sig")
    df.rename(columns=zh_columns).to_csv(zh_path, index=False, encoding="utf-8-sig")

    with diagnostics_path.open("w", encoding="utf-8") as diag_file:
        for record in diag_records:
            diag_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "评估完成：共 %d 行（%d 样本 × 配置 %d 方法；本次运行 %d 方法）",
        len(df),
        len(samples),
        len(config.stage2_eval_methods),
        len(methods_to_run),
    )
    logger.info("结果已保存: %s", en_path)
    logger.info("中文列名版本: %s", zh_path)
    logger.info("诊断日志: %s", diagnostics_path)

    ablation_df = pd.DataFrame()
    grid_df = pd.DataFrame()
    if run_secondary:
        try:
            from m3sum.eval.stage2_ablation_eval import run_stage2_ablation_eval

            ablation_df, grid_df = run_stage2_ablation_eval(
                config,
                samples,
                clip_cache,
                skip_clip=skip_clip,
            )
        except Exception as exc:
            logger.warning("ClusterPrior 消融评估失败（不影响主结果）: %s", exc)

    try:
        from m3sum.eval.stage2_reranking_viz import export_stage2_reranking_visuals

        viz_paths = export_stage2_reranking_visuals(
            df,
            config.eval_dir,
            ablation_df=ablation_df,
            grid_df=grid_df,
        )
        logger.info("可视化报告: %s", viz_paths.get("html_report"))
    except Exception as exc:
        logger.warning("可视化生成失败（不影响 CSV 结果）: %s", exc)

    if run_secondary:
        try:
            from m3sum.eval.legacy_compare_eval import run_legacy_compare_eval

            run_legacy_compare_eval(config, skip_clip=skip_clip, force_legacy_rerun=False)
        except Exception as exc:
            logger.warning("Legacy 对比评估失败（不影响主结果）: %s", exc)

        if config.raw.get("case_study_export", False):
            try:
                import importlib.util

                case_study_root = Path(__file__).resolve().parents[3] / "case_study"
                cs_config = case_study_root / "config.yaml"
                export_script = case_study_root / "scripts" / "export_case_study_data.py"
                if cs_config.is_file() and export_script.is_file():
                    spec = importlib.util.spec_from_file_location(
                        "export_case_study_data",
                        export_script,
                    )
                    mod = importlib.util.module_from_spec(spec)
                    assert spec.loader is not None
                    spec.loader.exec_module(mod)
                    mod.export_case_study_data(
                        cs_config,
                        trial_config_override=config.config_path,
                    )
                    logger.info("Case Study 数据已导出: %s", case_study_root / "data")
                else:
                    logger.warning(
                        "case_study_export 已启用但未找到配置或脚本: %s",
                        case_study_root,
                    )
            except Exception as exc:
                logger.warning("Case Study 导出失败（不影响 eval 结果）: %s", exc)

    return df
