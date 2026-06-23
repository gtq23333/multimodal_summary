from __future__ import annotations


def distance_weight(d: int, tiers: list[float] | None = None) -> float:
    """Discrete layout distance weight: d<=1 -> 1.0, 2-5 -> 0.7, >5 -> 0.4."""
    tiers = tiers or [1.0, 0.7, 0.4]
    if d <= 1:
        return tiers[0]
    if d <= 5:
        return tiers[1]
    return tiers[2]


def block_distance(pos_a: int, pos_b: int) -> int:
    if pos_a < 0 or pos_b < 0:
        return 999
    return abs(pos_a - pos_b)
