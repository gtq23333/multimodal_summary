from __future__ import annotations

from PyQt6.QtCore import Qt, QMimeData, pyqtSignal
from PyQt6.QtGui import QDrag
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from ui.utils.image_utils import load_full_pixmap, scale_pixmap_for_display

MIME_IMAGE_HASH = "application/x-image-hash"

CAPTION_AREA_HEIGHT = 56
CARD_PADDING = 12
MIN_IMAGE_HEIGHT = 120
MAX_IMAGE_HEIGHT = 480


class DraggableImageCard(QWidget):
    clicked = pyqtSignal(str)

    def __init__(self, candidate, card_width: int = 220, parent=None):
        super().__init__(parent)
        self.candidate = candidate
        self._card_width = card_width
        self._source_pixmap = load_full_pixmap(candidate.abs_image_path)
        self._drag_start_pos = None

        self.setAcceptDrops(False)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            "DraggableImageCard { border: 1px solid #ccc; border-radius: 6px; background: #fafafa; }"
            "DraggableImageCard:hover { border-color: #4a90d9; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        type_label = "[表]" if candidate.is_table else "[图]"
        self.title = QLabel(f"{type_label} {candidate.caption}")
        self.title.setWordWrap(True)
        self.title.setStyleSheet("font-size: 11px; font-weight: bold;")
        self.title.setFixedHeight(CAPTION_AREA_HEIGHT)

        self.thumb = QLabel()
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setStyleSheet("background: #eee; border: 1px solid #ddd;")
        self.thumb.setScaledContents(False)

        layout.addWidget(self.title)
        layout.addWidget(self.thumb)

        self.set_display_width(card_width)

    def set_display_width(self, card_width: int) -> None:
        self._card_width = max(80, card_width)
        image_width = self._card_width - CARD_PADDING
        image_height = self._compute_image_height(image_width)

        self.setFixedSize(self._card_width, CAPTION_AREA_HEIGHT + image_height + CARD_PADDING)
        self.thumb.setFixedSize(image_width, image_height)
        self._update_thumbnail()

    def _compute_image_height(self, image_width: int) -> int:
        if self._source_pixmap.isNull():
            return MIN_IMAGE_HEIGHT

        src_w = self._source_pixmap.width()
        src_h = self._source_pixmap.height()
        if src_w <= 0:
            return MIN_IMAGE_HEIGHT

        height = int(image_width * src_h / src_w)
        return max(MIN_IMAGE_HEIGHT, min(MAX_IMAGE_HEIGHT, height))

    def _update_thumbnail(self) -> None:
        if self._source_pixmap.isNull():
            self.thumb.setText(self.candidate.image_hash[:8])
            return

        dpr = self.devicePixelRatioF()
        scaled = scale_pixmap_for_display(
            self._source_pixmap,
            self.thumb.width(),
            self.thumb.height(),
            device_pixel_ratio=dpr,
        )
        self.thumb.setPixmap(scaled)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if self._drag_start_pos is None:
            return
        if (event.pos() - self._drag_start_pos).manhattanLength() < 10:
            return

        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(MIME_IMAGE_HASH, self.candidate.image_hash.encode("utf-8"))
        drag.setMimeData(mime)

        if not self._source_pixmap.isNull():
            drag_pixmap = scale_pixmap_for_display(
                self._source_pixmap,
                min(240, self.thumb.width()),
                min(180, self.thumb.height()),
            )
            drag.setPixmap(drag_pixmap)
        drag.exec(Qt.DropAction.MoveAction)
