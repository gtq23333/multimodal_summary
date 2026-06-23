from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


FusionMode = Literal["none", "additive", "multiplicative"]


@dataclass(frozen=True)
class FusionConfig:
    method_name: str
    use_direct: bool = True
    use_link: bool = True
    use_layout: bool = True
    use_type: bool = True
    use_cluster: bool = False
    use_local_window: bool = True
    cluster_fusion_mode: FusionMode = "none"
    beta: float = 0.0


def _link_value(item: dict[str, Any], use_local_window: bool) -> float:
    s_link = float(item.get("s_link", item.get("s_co", 0.0)) or 0.0)
    if use_local_window:
        return s_link

    link_debug = item.get("debug", {}).get("link", {})
    if link_debug.get("evidence_source") in {"local_prev", "local_next"}:
        return 0.0
    return s_link


def compute_fused_score(
    item: dict[str, Any],
    config: FusionConfig,
    alpha: float,
    cluster_prior: float = 0.0,
) -> float:
    """根据模块开关计算消融/融合分数。"""
    s_direct = float(item.get("s_direct", 0.0) or 0.0) if config.use_direct else 0.0
    s_link = _link_value(item, config.use_local_window) if config.use_link else 0.0

    if config.use_direct and config.use_link:
        base = alpha * s_direct + (1.0 - alpha) * s_link
    elif config.use_direct:
        base = s_direct
    elif config.use_link:
        base = s_link
    else:
        base = float(item.get("score", 0.0) or 0.0)

    if config.use_layout:
        base *= float(item.get("p_layout", 1.0) or 1.0)
    if config.use_type:
        base *= float(item.get("p_type", 1.0) or 1.0)

    if config.use_cluster and config.cluster_fusion_mode == "additive":
        base = base + config.beta * cluster_prior
    elif config.use_cluster and config.cluster_fusion_mode == "multiplicative":
        base = base * (1.0 + config.beta * cluster_prior)

    return float(base)


def incremental_configs(beta: float = 0.0, fusion_mode: FusionMode = "none") -> list[FusionConfig]:
    """核心递增式消融配置。"""
    configs = [
        FusionConfig("DirectOnly", use_link=False, use_layout=False, use_type=False),
        FusionConfig("Direct+Link", use_layout=False, use_type=False),
        FusionConfig("Direct+Link+Layout", use_type=False),
        FusionConfig("Direct+Link+Layout+Type"),
        FusionConfig("LG-JSSF"),
    ]
    if fusion_mode == "additive":
        configs.append(
            FusionConfig(
                "LG-JSSF+ClusterAdd",
                use_cluster=True,
                cluster_fusion_mode="additive",
                beta=beta,
            )
        )
    elif fusion_mode == "multiplicative":
        configs.append(
            FusionConfig(
                "LG-JSSF+ClusterMul",
                use_cluster=True,
                cluster_fusion_mode="multiplicative",
                beta=beta,
            )
        )
    return configs


def drop_one_configs(beta: float, fusion_mode: Literal["additive", "multiplicative"]) -> list[FusionConfig]:
    """Drop-one 消融配置。"""
    full_name = "FullClusterAdd" if fusion_mode == "additive" else "FullClusterMul"
    suffix = "Add" if fusion_mode == "additive" else "Mul"
    return [
        FusionConfig(
            full_name,
            use_cluster=True,
            cluster_fusion_mode=fusion_mode,
            beta=beta,
        ),
        FusionConfig(
            f"w/o S_link ({suffix})",
            use_link=False,
            use_cluster=True,
            cluster_fusion_mode=fusion_mode,
            beta=beta,
        ),
        FusionConfig(
            f"w/o P_layout ({suffix})",
            use_layout=False,
            use_cluster=True,
            cluster_fusion_mode=fusion_mode,
            beta=beta,
        ),
        FusionConfig(
            f"w/o P_type ({suffix})",
            use_type=False,
            use_cluster=True,
            cluster_fusion_mode=fusion_mode,
            beta=beta,
        ),
        FusionConfig(f"w/o ClusterPrior ({suffix})"),
        FusionConfig(
            f"w/o LocalWindow ({suffix})",
            use_local_window=False,
            use_cluster=True,
            cluster_fusion_mode=fusion_mode,
            beta=beta,
        ),
    ]
