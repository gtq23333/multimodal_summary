from __future__ import annotations

from m3sum.clients.openai_vlm import OpenAIVLMClient
from m3sum.data.schema import FigureMeta


def describe_figures(
    figures: list[FigureMeta],
    vlm: OpenAIVLMClient | None = None,
    mode: str = "vlm",
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for fig in figures:
        if mode == "caption":
            desc = fig.caption or f"图表 {fig.image_hash[:8]}"
        else:
            if vlm is None:
                raise ValueError("VLM client required when mode=vlm")
            desc = vlm.describe_image(fig.abs_image_path, fig.caption)
        results.append({"image_hash": fig.image_hash, "description": desc})
    return results
