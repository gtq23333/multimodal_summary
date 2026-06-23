from __future__ import annotations

from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtWidgets import QGridLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from ui.widgets.draggable_image_card import DraggableImageCard

COL_COUNT = 3
GRID_SPACING = 10
GRID_MARGIN = 8


class ImagePoolPanel(QWidget):
    """Right panel showing available image candidates."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards: dict[str, DraggableImageCard] = {}
        self._candidate_map: dict = {}
        self._ordered_hashes: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel("候选图片（拖拽到左侧摘要）")
        self.title.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.count_label = QLabel("0 张可选")
        layout.addWidget(self.title)
        layout.addWidget(self.count_label)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.viewport().installEventFilter(self)
        self.container = QWidget()
        self.grid = QGridLayout(self.container)
        self.grid.setContentsMargins(GRID_MARGIN, GRID_MARGIN, GRID_MARGIN, GRID_MARGIN)
        self.grid.setHorizontalSpacing(GRID_SPACING)
        self.grid.setVerticalSpacing(GRID_SPACING)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll, 1)

        self.setAcceptDrops(True)

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(50)
        self._resize_timer.timeout.connect(self._update_card_sizes)

    def set_candidates(self, candidates: list, used_hashes: set[str] | None = None):
        used_hashes = used_hashes or set()
        self._clear_cards()
        self._candidate_map = {c.image_hash: c for c in candidates}

        available = [c for c in candidates if c.image_hash not in used_hashes]
        self._ordered_hashes = [c.image_hash for c in available]
        self.count_label.setText(f"{len(available)} 张可选 / 共 {len(candidates)} 张")

        card_width = self._compute_card_width()
        for i, candidate in enumerate(available):
            card = DraggableImageCard(candidate, card_width=card_width)
            self._cards[candidate.image_hash] = card
            row, col = divmod(i, COL_COUNT)
            self.grid.addWidget(card, row, col)

    def remove_candidate(self, image_hash: str):
        card = self._cards.pop(image_hash, None)
        if card:
            self.grid.removeWidget(card)
            card.deleteLater()
        if image_hash in self._ordered_hashes:
            self._ordered_hashes.remove(image_hash)
        self._relayout_grid()
        available = len(self._cards)
        total = len(self._candidate_map)
        self.count_label.setText(f"{available} 张可选 / 共 {total} 张")

    def restore_candidate(self, image_hash: str):
        candidate = self._candidate_map.get(image_hash)
        if candidate is None or image_hash in self._cards:
            return

        if image_hash not in self._ordered_hashes:
            self._ordered_hashes.append(image_hash)

        card_width = self._compute_card_width()
        card = DraggableImageCard(candidate, card_width=card_width)
        self._cards[image_hash] = card
        self._relayout_grid()
        available = len(self._cards)
        total = len(self._candidate_map)
        self.count_label.setText(f"{available} 张可选 / 共 {total} 张")

    def get_candidate(self, image_hash: str):
        return self._candidate_map.get(image_hash)

    def _clear_cards(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._cards.clear()
        self._ordered_hashes.clear()

    def schedule_layout_update(self):
        self._resize_timer.start()

    def _compute_card_width(self) -> int:
        viewport_w = self.scroll.viewport().width()
        if viewport_w <= 0:
            viewport_w = max(360, self.width())
        usable = viewport_w - GRID_MARGIN * 2 - GRID_SPACING * (COL_COUNT - 1)
        return max(100, usable // COL_COUNT)

    def _relayout_grid(self):
        for i in reversed(range(self.grid.count())):
            item = self.grid.itemAt(i)
            if item and item.widget():
                self.grid.removeWidget(item.widget())

        card_width = self._compute_card_width()
        for i, image_hash in enumerate(self._ordered_hashes):
            card = self._cards.get(image_hash)
            if card is None:
                continue
            row, col = divmod(i, COL_COUNT)
            self.grid.addWidget(card, row, col)
            card.set_display_width(card_width)

    def _update_card_sizes(self):
        if not self._cards:
            return
        card_width = self._compute_card_width()
        for card in self._cards.values():
            card.set_display_width(card_width)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_timer.start()

    def eventFilter(self, obj, event):
        if obj is self.scroll.viewport() and event.type() == QEvent.Type.Resize:
            self._resize_timer.start()
        return super().eventFilter(obj, event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-image-hash"):
            event.acceptProposedAction()

    def dropEvent(self, event):
        raw = event.mimeData().data("application/x-image-hash")
        if raw:
            image_hash = bytes(raw).decode("utf-8")
            self.restore_candidate(image_hash)
            event.acceptProposedAction()
