from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap


def load_full_pixmap(path: str) -> QPixmap:
    """Load image at full resolution."""
    return QPixmap(path)


def scale_pixmap_for_display(
    source: QPixmap,
    max_width: int,
    max_height: int,
    device_pixel_ratio: float = 1.0,
) -> QPixmap:
    """Scale pixmap to fit display area while preserving aspect ratio."""
    if source.isNull() or max_width <= 0 or max_height <= 0:
        return source

    target_w = max(1, int(max_width * device_pixel_ratio))
    target_h = max(1, int(max_height * device_pixel_ratio))
    scaled = source.scaled(
        target_w,
        target_h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    if device_pixel_ratio > 1.0:
        scaled.setDevicePixelRatio(device_pixel_ratio)
    return scaled
