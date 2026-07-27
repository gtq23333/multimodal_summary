from __future__ import annotations

import base64
import logging
import time
from collections import defaultdict
from enum import Enum
from http import HTTPStatus
from io import BytesIO
from pathlib import Path

import dashscope
from PIL import Image

from m3sum.data.schema import FigureMeta

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCT = (
    "Given a query describing information needs for a mathematical modeling paper "
    "summary, retrieve the most relevant figures."
)
DEFAULT_INSTRUCT_IMG_CAP = (
    "Given a query describing information needs for a mathematical modeling paper "
    "summary, retrieve the most relevant figure entries. Each entry includes the "
    "figure image and its caption text."
)
DEFAULT_INSTRUCT_IMG_CAP_LINK = (
    "Given a query describing information needs for a mathematical modeling paper "
    "summary, retrieve the most relevant figure entries. Each entry includes the "
    "figure image, its caption, and the most relevant surrounding text chunk that "
    "discusses or references the figure."
)
MAX_IMAGES_PER_REQUEST = 40
MAX_TEXT_DOCS_PER_REQUEST = 100
MAX_IMAGE_BYTES = 2 * 1024 * 1024
API_MAX_RETRIES = 5
API_RETRY_BASE_SEC = 3.0


class DocumentMode(str, Enum):
    IMAGE_ONLY = "image_only"
    IMAGE_CAPTION = "image_caption"
    IMAGE_CAPTION_CONTEXT = "image_caption_context"


def encode_image_base64(path: Path) -> str:
    """Encode a local image as a data URI for DashScope qwen3-vl-rerank."""
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    raw = path.read_bytes()
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix or "jpeg"

    if len(raw) > MAX_IMAGE_BYTES:
        img = Image.open(BytesIO(raw))
        img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        raw = buf.getvalue()
        mime = "jpeg"

    data = base64.b64encode(raw).decode("utf-8")
    return f"data:image/{mime};base64,{data}"


def _max_figures_per_chunk(mode: DocumentMode) -> int:
    if mode == DocumentMode.IMAGE_ONLY:
        return MAX_IMAGES_PER_REQUEST
    return min(MAX_IMAGES_PER_REQUEST, MAX_TEXT_DOCS_PER_REQUEST // 2)


def _build_documents(
    figures: list[FigureMeta],
    mode: DocumentMode,
    context_by_figure: dict[str, str] | None = None,
) -> tuple[list[dict], list[int]]:
    """Build API documents and doc_index -> local_figure_index mapping."""
    documents: list[dict] = []
    doc_to_fig: list[int] = []
    ctx_map = context_by_figure or {}

    for fig_idx, fig in enumerate(figures):
        image_uri = encode_image_base64(Path(fig.abs_image_path))
        if mode == DocumentMode.IMAGE_ONLY:
            documents.append({"image": image_uri})
            doc_to_fig.append(fig_idx)
        elif mode == DocumentMode.IMAGE_CAPTION:
            documents.append({"text": fig.caption.strip()})
            doc_to_fig.append(fig_idx)
            documents.append({"image": image_uri})
            doc_to_fig.append(fig_idx)
        elif mode == DocumentMode.IMAGE_CAPTION_CONTEXT:
            caption = fig.caption.strip()
            context = ctx_map.get(fig.image_hash, "").strip()
            text_parts = [f"Caption: {caption}"]
            if context:
                text_parts.append(f"Context: {context}")
            documents.append({"text": "\n\n".join(text_parts)})
            doc_to_fig.append(fig_idx)
            documents.append({"image": image_uri})
            doc_to_fig.append(fig_idx)
        else:
            raise ValueError(f"Unsupported document mode: {mode}")

    return documents, doc_to_fig


def _aggregate_doc_scores(
    doc_scores: list[tuple[int, float]],
    doc_to_fig: list[int],
    n_figures: int,
) -> list[tuple[int, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for doc_idx, score in doc_scores:
        grouped[doc_to_fig[doc_idx]].append(score)

    results: list[tuple[int, float]] = []
    for fig_idx in range(n_figures):
        vals = grouped.get(fig_idx, [])
        if vals:
            results.append((fig_idx, float(sum(vals) / len(vals))))
        else:
            results.append((fig_idx, 0.0))
    return results


class DashScopeVLRerankClient:
    """DashScope qwen3-vl-rerank client for text-query / multimodal-document reranking."""

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-vl-rerank",
        instruct: str = DEFAULT_INSTRUCT,
    ):
        self.model = model
        self.instruct = instruct
        dashscope.api_key = api_key

    def rerank_figures(
        self,
        query_text: str,
        figures: list[FigureMeta],
        *,
        mode: DocumentMode = DocumentMode.IMAGE_ONLY,
        top_n: int | None = None,
        context_by_figure: dict[str, str] | None = None,
    ) -> list[tuple[int, float]]:
        """
        Rerank figure documents against a text query.

        Returns list of (figure_index, relevance_score) aligned with `figures`.
        """
        if not figures:
            return []

        chunk_size = _max_figures_per_chunk(mode)
        if len(figures) <= chunk_size:
            return self._rerank_figure_chunk(
                query_text,
                figures,
                mode=mode,
                top_n=top_n,
                context_by_figure=context_by_figure,
            )

        merged: dict[int, list[float]] = {i: [] for i in range(len(figures))}
        for start in range(0, len(figures), chunk_size):
            chunk_figs = figures[start : start + chunk_size]
            chunk_ctx = None
            if context_by_figure is not None:
                chunk_ctx = {
                    fig.image_hash: context_by_figure.get(fig.image_hash, "")
                    for fig in chunk_figs
                }
            chunk_scores = self._rerank_figure_chunk(
                query_text,
                chunk_figs,
                mode=mode,
                top_n=top_n or len(chunk_figs),
                context_by_figure=chunk_ctx,
            )
            normalized = _normalize_scores(chunk_scores)
            for local_idx, score in normalized:
                merged[start + local_idx].append(score)

        return [
            (idx, float(sum(vals) / len(vals)))
            for idx, vals in merged.items()
            if vals
        ]

    def _rerank_figure_chunk(
        self,
        query_text: str,
        figures: list[FigureMeta],
        *,
        mode: DocumentMode,
        top_n: int | None,
        context_by_figure: dict[str, str] | None = None,
    ) -> list[tuple[int, float]]:
        documents, doc_to_fig = _build_documents(
            figures,
            mode,
            context_by_figure=context_by_figure,
        )
        effective_top_n = top_n if top_n is not None else len(documents)

        last_error: Exception | None = None
        for attempt in range(1, API_MAX_RETRIES + 1):
            try:
                resp = dashscope.TextReRank.call(
                    model=self.model,
                    query={"text": query_text},
                    documents=documents,
                    top_n=effective_top_n,
                    return_documents=False,
                    instruct=self.instruct,
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt >= API_MAX_RETRIES:
                    raise
                wait_sec = API_RETRY_BASE_SEC * attempt
                logger.warning(
                    "  [VL-Rerank] API 请求失败 (attempt %d/%d): %s; %.0fs 后重试",
                    attempt,
                    API_MAX_RETRIES,
                    exc,
                    wait_sec,
                )
                time.sleep(wait_sec)
        else:
            raise last_error  # pragma: no cover

        if resp.status_code != HTTPStatus.OK:
            request_id = getattr(resp, "request_id", "")
            code = getattr(resp, "code", "") or getattr(resp, "status_code", "")
            message = getattr(resp, "message", str(resp))
            raise RuntimeError(
                f"DashScope qwen3-vl-rerank failed: code={code} message={message} "
                f"request_id={request_id}"
            )

        output = getattr(resp, "output", None) or {}
        results = output.get("results", []) if isinstance(output, dict) else []
        doc_scores = [(int(item["index"]), float(item["relevance_score"])) for item in results]

        usage = getattr(resp, "usage", None)
        if usage:
            logger.info(
                "  [VL-Rerank] mode=%s query=%r figures=%d docs=%d tokens=%s",
                mode.value,
                query_text[:60],
                len(figures),
                len(documents),
                usage,
            )

        return _aggregate_doc_scores(doc_scores, doc_to_fig, len(figures))


def _normalize_scores(scores: list[tuple[int, float]]) -> list[tuple[int, float]]:
    if not scores:
        return []
    values = [s for _, s in scores]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [(idx, 1.0) for idx, _ in scores]
    return [(idx, (score - lo) / (hi - lo)) for idx, score in scores]
