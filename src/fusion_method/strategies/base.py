from __future__ import annotations

from typing import Protocol

from fusion_method.types import FusionInput


class FusionStrategy(Protocol):
    name: str

    def fuse(self, fusion_input: FusionInput) -> list[tuple[str, float]]: ...
