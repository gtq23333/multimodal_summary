from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

MAX_IMAGE_WIDTH = 420
MAX_IMAGE_HEIGHT = 280


def _load_pixmap(image_path: str) -> QPixmap | None:
    if not image_path:
        return None
    path = Path(image_path)
    if not path.is_file():
        return None
    pix = QPixmap(str(path))
    if pix.isNull():
        return None
    return pix.scaled(
        MAX_IMAGE_WIDTH,
        MAX_IMAGE_HEIGHT,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class GtSequencePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._scroll.setWidget(self._container)

        root = QVBoxLayout(self)
        title = QLabel("GT 标注序列")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        root.addWidget(title)
        root.addWidget(self._scroll, 1)

    def _clear_blocks(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def set_sequence(self, sequence: list[dict[str, Any]]) -> None:
        self._clear_blocks()
        for item in sequence:
            block_type = item.get("type", "text")
            if block_type == "text":
                text = item.get("text") or item.get("content") or ""
                editor = QTextEdit()
                editor.setReadOnly(True)
                editor.setPlainText(text.strip())
                editor.setMaximumHeight(min(220, max(80, len(text) // 4 + 60)))
                self._layout.addWidget(editor)
            elif block_type == "image":
                card = QFrame()
                card.setFrameShape(QFrame.Shape.StyledPanel)
                card.setStyleSheet(
                    "QFrame { border: 2px solid #2ecc71; border-radius: 6px; background: #f8fff9; }"
                )
                card_layout = QVBoxLayout(card)

                badge = QLabel("GT")
                badge.setStyleSheet(
                    "background: #2ecc71; color: white; padding: 2px 8px; "
                    "border-radius: 4px; font-weight: bold; max-width: 40px;"
                )
                card_layout.addWidget(badge, alignment=Qt.AlignmentFlag.AlignLeft)

                image_path = item.get("image_path") or item.get("img_path") or ""
                pix = _load_pixmap(image_path)
                img_label = QLabel()
                img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                if pix is not None:
                    img_label.setPixmap(pix)
                else:
                    img_label.setText(f"[图片不可用]\n{image_path}")
                    img_label.setWordWrap(True)
                img_label.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Preferred,
                )
                card_layout.addWidget(img_label)

                caption = item.get("caption") or item.get("figure_caption") or ""
                if caption:
                    cap_label = QLabel(caption.strip())
                    cap_label.setWordWrap(True)
                    cap_label.setStyleSheet("color: #333; font-size: 11px;")
                    card_layout.addWidget(cap_label)

                self._layout.addWidget(card)
        self._layout.addStretch(1)
