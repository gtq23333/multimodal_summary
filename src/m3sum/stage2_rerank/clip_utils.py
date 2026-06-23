from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from m3sum.data.schema import FigureMeta

logger = logging.getLogger(__name__)

DEFAULT_CLIP_MODEL = "OFA-Sys/chinese-clip-vit-base-patch16"


def _as_feature_tensor(features) -> "torch.Tensor":
    """将 model 输出统一转为 2D tensor。"""
    import torch

    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "text_embeds") and features.text_embeds is not None:
        return features.text_embeds
    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        return features.image_embeds
    if hasattr(features, "last_hidden_state"):
        return features.last_hidden_state[:, 0]
    raise TypeError(f"无法从 {type(features)} 提取 embedding tensor")


def load_clip_model(model_name: str = DEFAULT_CLIP_MODEL, device: str | None = None):
    """
    解耦的 CLIP 模型加载入口，便于后续替换模型或本地路径。
    返回 ChineseCLIPWrapper 实例。
    """
    return ChineseCLIPWrapper.load(model_name=model_name, device=device)


class ChineseCLIPWrapper:
    """Chinese-CLIP 图文编码封装，评估生命周期内单例复用。"""

    def __init__(self, model, processor, device: str, model_name: str):
        self.model = model
        self.processor = processor
        self.device = device
        self.model_name = model_name

    @classmethod
    def load(cls, model_name: str = DEFAULT_CLIP_MODEL, device: str | None = None) -> ChineseCLIPWrapper:
        import torch
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("加载 Chinese-CLIP 模型 %s，设备=%s", model_name, resolved_device)
        processor = ChineseCLIPProcessor.from_pretrained(model_name)
        model = ChineseCLIPModel.from_pretrained(model_name)
        model.to(resolved_device)
        model.eval()
        return cls(model=model, processor=processor, device=resolved_device, model_name=model_name)

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> list[np.ndarray]:
        """批量编码文本，返回 L2 归一化向量列表。"""
        if not texts:
            return []
        import torch

        all_embs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                inputs = self.processor(
                    text=chunk,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                ).to(self.device)
                features = self.model.get_text_features(**inputs)
                features = _as_feature_tensor(features)
                features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                all_embs.extend(f.cpu().numpy().astype(np.float32) for f in features)
        return all_embs

    def encode_images(self, image_paths: list[str], batch_size: int = 16) -> list[np.ndarray | None]:
        """
        批量编码图片路径；无法读取的路径对应位置返回 None。
        """
        if not image_paths:
            return []
        from PIL import Image
        import torch

        all_embs: list[np.ndarray | None] = []
        batch_images: list = []
        batch_indices: list[int] = []
        placeholder: dict[int, np.ndarray | None] = {}

        def flush_batch() -> None:
            if not batch_images:
                return
            with torch.no_grad():
                inputs = self.processor(
                    images=batch_images,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                features = self.model.get_image_features(**inputs)
                features = _as_feature_tensor(features)
                features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                for idx, feat in zip(batch_indices, features):
                    placeholder[idx] = feat.cpu().numpy().astype(np.float32)
            batch_images.clear()
            batch_indices.clear()

        for idx, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
            except (OSError, FileNotFoundError) as exc:
                logger.warning("无法读取图片 %s: %s", path, exc)
                placeholder[idx] = None
                continue
            batch_images.append(img)
            batch_indices.append(idx)
            if len(batch_images) >= batch_size:
                flush_batch()
        flush_batch()

        for i in range(len(image_paths)):
            all_embs.append(placeholder.get(i))
        return all_embs


class ClipImageEmbeddingCache:
    """CLIP 图像 embedding 磁盘缓存（NPZ）。"""

    def __init__(self, cache_dir: Path, clip_encoder: ChineseCLIPWrapper | None = None, dry_run: bool = False):
        self.cache_dir = cache_dir
        self.clip_encoder = clip_encoder
        self.dry_run = dry_run
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, paper_id: str) -> Path:
        return self.cache_dir / f"{paper_id}.npz"

    def load_or_compute(
        self,
        paper_id: str,
        figures: list[FigureMeta],
    ) -> dict[str, np.ndarray | None]:
        cache_path = self._cache_path(paper_id)
        fig_ids = [f.image_hash for f in figures]

        if cache_path.is_file():
            data = np.load(cache_path, allow_pickle=True)
            cached: dict[str, np.ndarray | None] = {}
            hit = True
            for fid in fig_ids:
                key = f"img_{fid}"
                if key not in data:
                    hit = False
                    break
                val = data[key]
                cached[fid] = None if val is None or (isinstance(val, float) and np.isnan(val)) else val
            if hit and len(cached) == len(fig_ids):
                return cached

        if self.dry_run:
            dim = 512
            return {f.image_hash: np.random.randn(dim).astype(np.float32) for f in figures}

        if self.clip_encoder is None:
            raise RuntimeError("ClipImageEmbeddingCache 需要 clip_encoder 或 dry_run=True")

        paths = [f.abs_image_path for f in figures]
        vectors = self.clip_encoder.encode_images(paths)
        result = {f.image_hash: vec for f, vec in zip(figures, vectors)}

        save_dict = {}
        for fid, vec in result.items():
            if vec is None:
                save_dict[f"img_{fid}"] = np.array(float("nan"), dtype=np.float32)
            else:
                save_dict[f"img_{fid}"] = vec
        np.savez(cache_path, **save_dict)
        return result
