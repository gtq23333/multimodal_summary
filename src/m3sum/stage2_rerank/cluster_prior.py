from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ClusterPriorDebug:
    cluster_top1_label: str | None
    cluster_top1_sim: float
    cluster_top2_label: str | None
    cluster_top2_sim: float
    cluster_margin: float
    cluster_prior_raw: float
    cluster_prior: float
    cluster_gate_passed: bool
    cluster_fusion_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_top1_label": self.cluster_top1_label,
            "cluster_top1_sim": round(self.cluster_top1_sim, 6),
            "cluster_top2_label": self.cluster_top2_label,
            "cluster_top2_sim": round(self.cluster_top2_sim, 6),
            "cluster_margin": round(self.cluster_margin, 6),
            "cluster_prior_raw": round(self.cluster_prior_raw, 6),
            "cluster_prior": round(self.cluster_prior, 6),
            "cluster_gate_passed": self.cluster_gate_passed,
            "cluster_fusion_mode": self.cluster_fusion_mode,
        }


@dataclass
class ClusterPrototype:
    label: str
    weight: float
    centroid: np.ndarray


class ClusterPriorScorer:
    """基于 Chinese-CLIP 图像 embedding 与领域聚类质心的弱先验打分器。"""

    def __init__(
        self,
        prototypes: list[ClusterPrototype],
        model_name: str,
        tau: float = 0.78,
        margin_tau: float = 0.03,
        threshold_mode: str = "top1_margin",
    ):
        self.prototypes = prototypes
        self.model_name = model_name
        self.tau = tau
        self.margin_tau = margin_tau
        self.threshold_mode = threshold_mode

    @classmethod
    def from_json(
        cls,
        path: Path,
        tau: float = 0.78,
        margin_tau: float = 0.03,
        threshold_mode: str = "top1_margin",
    ) -> ClusterPriorScorer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model_name = data.get("embedding_model", {}).get(
            "name", "OFA-Sys/chinese-clip-vit-base-patch16"
        )
        normalized_weights = data.get("weighting", {}).get("normalized_weights", {})

        prototypes: list[ClusterPrototype] = []
        for proto in data.get("prototypes", []):
            label = str(proto.get("cluster_label"))
            vector = np.array(proto.get("centroid_vector", []), dtype=np.float32)
            if vector.size == 0:
                continue
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            weight = float(
                normalized_weights.get(
                    label,
                    proto.get("human_weight_normalized", 0.0),
                )
            )
            prototypes.append(
                ClusterPrototype(label=label, weight=weight, centroid=vector)
            )

        if not prototypes:
            raise ValueError(f"cluster prior 文件未包含可用 prototypes: {path}")
        return cls(
            prototypes=prototypes,
            model_name=model_name,
            tau=tau,
            margin_tau=margin_tau,
            threshold_mode=threshold_mode,
        )

    def score(self, image_embedding: np.ndarray | None) -> tuple[float, ClusterPriorDebug]:
        """返回门控后的 cluster prior 与 debug 信息。"""
        if image_embedding is None:
            debug = ClusterPriorDebug(
                cluster_top1_label=None,
                cluster_top1_sim=0.0,
                cluster_top2_label=None,
                cluster_top2_sim=0.0,
                cluster_margin=0.0,
                cluster_prior_raw=0.0,
                cluster_prior=0.0,
                cluster_gate_passed=False,
            )
            return 0.0, debug

        emb = np.array(image_embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm == 0:
            return self.score(None)
        emb = emb / norm

        sims = [
            (proto.label, float(np.dot(emb, proto.centroid)), proto.weight)
            for proto in self.prototypes
        ]
        sims.sort(key=lambda x: x[1], reverse=True)
        top1_label, top1_sim, _ = sims[0]
        top2_label, top2_sim, _ = sims[1] if len(sims) > 1 else (None, 0.0, 0.0)
        margin = top1_sim - top2_sim
        prior_raw = sum(weight * sim for _, sim, weight in sims)

        gate_passed = top1_sim >= self.tau
        dilution = 1.0
        if self.threshold_mode == "top1_margin":
            if self.margin_tau > 0:
                dilution = min(1.0, max(0.0, margin / self.margin_tau))
        elif self.threshold_mode == "weighted_sum_threshold":
            gate_passed = prior_raw >= self.tau
        elif self.threshold_mode == "softmax_temperature":
            # 当前仅保留同一接口；温度 softmax 可后续扩展。
            pass

        prior = prior_raw * dilution if gate_passed else 0.0
        debug = ClusterPriorDebug(
            cluster_top1_label=top1_label,
            cluster_top1_sim=top1_sim,
            cluster_top2_label=top2_label,
            cluster_top2_sim=top2_sim,
            cluster_margin=margin,
            cluster_prior_raw=prior_raw,
            cluster_prior=prior,
            cluster_gate_passed=gate_passed,
        )
        return prior, debug
