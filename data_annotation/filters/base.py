from __future__ import annotations

from abc import ABC, abstractmethod

from models.paper import ImageCandidate, Paper


class ImageFilterStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def filter(self, candidates: list[ImageCandidate], paper: Paper | None) -> list[ImageCandidate]:
        raise NotImplementedError
