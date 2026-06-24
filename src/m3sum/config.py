from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ApiCredentials:
    base_url: str
    api_key: str


@dataclass
class PipelineConfig:
    root: Path
    config_path: Path
    annotation_config: Path
    manifest: Path
    ground_truth_dir: Path
    problem_mds_root: Path
    output_dir: Path
    acceptance_csv: Path
    llm_model: str
    embed_model: str
    vlm_model: str
    api_base_url: str
    api_key: str
    top_p: int
    bm25_weight: float
    vector_weight: float
    alpha: float
    distance_tiers: list[float]
    caption_patterns: list[str]
    gt_mode: str
    sample_ids: list[str]
    stage: str
    mode: str
    vlm_mode: str
    force_rerun: bool
    use_cache: bool
    sample_id: str | None
    smoke_test: bool
    auto_prepare_manifest: bool
    auto_sanity_check: bool
    auto_eval: bool
    sanity_stage: str
    init_acceptance_csv: bool
    export_html_report: bool
    export_csv_summary: bool
    export_markdown_summary: bool
    stage2_eval_jaccard_k: int
    stage2_eval_maxsim_k: int
    stage2_eval_clip_model: str
    stage2_eval_methods: list[str]
    cluster_prior_enabled: bool
    cluster_prior_path: Path
    cluster_prior_clip_model: str
    cluster_prior_threshold_mode: str
    cluster_prior_tau_grid: list[float]
    cluster_prior_beta_grid: list[float]
    cluster_prior_margin_tau: float
    cluster_prior_fusion_modes: list[str]
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def dry_run(self) -> bool:
        return self.mode == "dry_run"

    @property
    def stage1_dir(self) -> Path:
        return self.output_dir / "stage1"

    @property
    def stage2_dir(self) -> Path:
        return self.output_dir / "stage2"

    @property
    def stage3_dir(self) -> Path:
        return self.output_dir / "stage3"

    @property
    def embed_cache_dir(self) -> Path:
        return self.output_dir / "cache" / "embeddings"

    @property
    def eval_dir(self) -> Path:
        return self.output_dir / "eval"

    @property
    def stage2_eval_text_cache_dir(self) -> Path:
        return self.output_dir / "cache" / "stage2_eval" / "text_embeddings"

    @property
    def stage2_eval_clip_cache_dir(self) -> Path:
        return self.output_dir / "cache" / "stage2_eval" / "clip_embeddings"

    @property
    def stage2_eval_vl_rerank_cache_dir(self) -> Path:
        return self.output_dir / "cache" / "stage2_eval" / "vl_rerank"

    @property
    def cluster_prior_cache_dir(self) -> Path:
        return self.output_dir / "cache" / "stage2_eval" / "cluster_prior_clip"

    def resolved_sample_ids(self) -> list[str]:
        if self.sample_id:
            return [self.sample_id]
        if self.smoke_test:
            return [self.sample_ids[0]] if self.sample_ids else []
        return self.sample_ids

    def resolved_sanity_stage(self) -> int:
        if self.sanity_stage != "auto":
            return int(self.sanity_stage)
        mapping = {"1": 1, "2": 2, "3": 3, "all": 3}
        return mapping.get(self.stage, 0)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path) -> PipelineConfig:
    config_path = config_path.resolve()
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    local_path = config_path.with_name(config_path.stem + ".local.yaml")
    if local_path.is_file():
        with open(local_path, encoding="utf-8") as f:
            local_raw = yaml.safe_load(f) or {}
        raw = _deep_merge(raw, local_raw)

    base = config_path.parent
    paths = raw["paths"]
    run = raw.get("run", {})
    sanity = raw.get("sanity_check", {})
    ev = raw.get("eval", {})
    api = raw.get("api", {})
    s2ev = raw.get("stage2_eval", {})
    cp = raw.get("cluster_prior", {})

    def resolve(p: str | None) -> Path | None:
        if not p:
            return None
        return (base / p).resolve()

    base_url = (api.get("base_url") or "").strip()
    api_key = (api.get("api_key") or "").strip()

    return PipelineConfig(
        root=base.parent.parent,
        config_path=config_path,
        annotation_config=resolve(paths["annotation_config"]),
        manifest=resolve(paths["manifest"]),
        ground_truth_dir=resolve(paths["ground_truth_dir"]),
        problem_mds_root=resolve(paths["problem_mds_root"]),
        output_dir=resolve(paths["output_dir"]),
        acceptance_csv=resolve(paths.get("acceptance_csv"))
        or (base.parent.parent / "data" / "trial_10" / "acceptance_review.csv"),
        llm_model=raw["models"]["llm"],
        embed_model=raw["models"]["embed"],
        vlm_model=raw["models"]["vlm"],
        api_base_url=base_url,
        api_key=api_key,
        top_p=raw["retrieval"]["top_p"],
        bm25_weight=raw["retrieval"]["bm25_weight"],
        vector_weight=raw["retrieval"]["vector_weight"],
        alpha=raw["rerank"]["alpha"],
        distance_tiers=raw["rerank"]["distance_tiers"],
        caption_patterns=raw["caption_regex"]["patterns"],
        gt_mode=raw["trial"]["gt_mode"],
        sample_ids=raw["trial"]["sample_ids"],
        stage=run.get("stage", "all"),
        mode=run.get("mode", "dry_run"),
        vlm_mode=run.get("vlm_mode", "caption"),
        force_rerun=bool(run.get("force_rerun", False)),
        use_cache=bool(run.get("use_cache", True)),
        sample_id=run.get("sample_id"),
        smoke_test=bool(run.get("smoke_test", False)),
        auto_prepare_manifest=bool(run.get("auto_prepare_manifest", True)),
        auto_sanity_check=bool(run.get("auto_sanity_check", True)),
        auto_eval=bool(run.get("auto_eval", True)),
        sanity_stage=str(sanity.get("stage", "auto")),
        init_acceptance_csv=bool(ev.get("init_acceptance_csv", True)),
        export_html_report=bool(ev.get("export_html_report", True)),
        export_csv_summary=bool(ev.get("export_csv_summary", True)),
        export_markdown_summary=bool(ev.get("export_markdown_summary", True)),
        stage2_eval_jaccard_k=int(s2ev.get("jaccard_k", 3)),
        stage2_eval_maxsim_k=int(s2ev.get("maxsim_k", 3)),
        stage2_eval_clip_model=str(
            s2ev.get("clip_model", "OFA-Sys/chinese-clip-vit-base-patch16")
        ),
        stage2_eval_methods=list(
            s2ev.get(
                "methods",
                [
                    "Proposed",
                    "Qwen3-VL-Rerank-ImgCap+Link",
                    "Qwen3-VL-Rerank-ImgCap",
                    "Qwen3-VL-Rerank-Img",
                    "Layout-Order",
                    "Caption-BM25",
                    "Caption-Dense-v4",
                    "Zero-shot-CLIP",
                ],
            )
        ),
        cluster_prior_enabled=bool(cp.get("enabled", True)),
        cluster_prior_path=resolve(cp.get("path", "../m3sum/cluster_prior.json"))
        or (base.parent / "m3sum" / "cluster_prior.json"),
        cluster_prior_clip_model=str(
            cp.get("clip_model", "OFA-Sys/chinese-clip-vit-base-patch16")
        ),
        cluster_prior_threshold_mode=str(cp.get("threshold_mode", "top1_margin")),
        cluster_prior_tau_grid=[float(x) for x in cp.get("tau_grid", [0.72, 0.75, 0.78])],
        cluster_prior_beta_grid=[
            float(x) for x in cp.get("beta_grid", [0.15, 0.25, 0.35])
        ],
        cluster_prior_margin_tau=float(cp.get("margin_tau", 0.03)),
        cluster_prior_fusion_modes=list(
            cp.get("fusion_modes", ["additive", "multiplicative"])
        ),
        raw=raw,
    )


def resolve_api_credentials(config: PipelineConfig) -> ApiCredentials:
    """Read base_url + api_key from config file only."""
    if config.dry_run:
        return ApiCredentials(base_url=config.api_base_url, api_key=config.api_key)

    if not config.api_base_url:
        raise RuntimeError(
            "未配置 api.base_url。请在 configs/trial_10.yaml 的 api 段填写 OpenAI 兼容平台地址。"
        )
    if not config.api_key:
        raise RuntimeError(
            "未配置 api.api_key。请在 configs/trial_10.yaml 的 api 段直接粘贴 API Key。"
        )

    base_url = config.api_base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"

    return ApiCredentials(base_url=base_url, api_key=config.api_key)
