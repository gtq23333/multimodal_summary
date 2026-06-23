from __future__ import annotations

import importlib
from typing import Any

from filters.base import ImageFilterStrategy
from filters.body_with_caption import BodyWithCaptionFilter

_REGISTRY: dict[str, type[ImageFilterStrategy]] = {
    BodyWithCaptionFilter.name: BodyWithCaptionFilter,
}


def get_filter_strategy(config: dict[str, Any]) -> ImageFilterStrategy:
    strategy_name = config.get("strategy", "body_with_caption")
    module_path = config.get("strategy_module")
    class_name = config.get("strategy_class")

    if module_path and class_name:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls()

    cls = _REGISTRY.get(strategy_name)
    if cls is None:
        raise ValueError(f"Unknown image filter strategy: {strategy_name}")
    return cls()
