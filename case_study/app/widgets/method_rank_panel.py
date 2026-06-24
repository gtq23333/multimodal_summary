from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

MAX_IMAGE_WIDTH = 360
MAX_IMAGE_HEIGHT = 220


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


class _FigureCard(QFrame):
    def __init__(
        self,
        item: dict[str, Any],
        *,
        show_debug: bool,
        parent=None,
    ):
        super().__init__(parent)
        is_gt = bool(item.get("is_gt"))
        border = "#2ecc71" if is_gt else "#bdc3c7"
        bg = "#f8fff9" if is_gt else "#fafafa"
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"QFrame {{ border: 2px solid {border}; border-radius: 6px; background: {bg}; }}"
        )

        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        rank = item.get("rank", "?")
        score = item.get("score", 0)
        header.addWidget(QLabel(f"#{rank}"))
        header.addWidget(QLabel(f"score={float(score):.4f}"))
        if is_gt:
            gt_badge = QLabel("✓ GT")
            gt_badge.setStyleSheet("color: #27ae60; font-weight: bold;")
            header.addWidget(gt_badge)
        header.addStretch(1)
        layout.addLayout(header)

        image_path = item.get("image_path") or ""
        pix = _load_pixmap(image_path)
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if pix is not None:
            img_label.setPixmap(pix)
        else:
            img_label.setText(f"[图片不可用]\n{image_path}")
            img_label.setWordWrap(True)
        layout.addWidget(img_label)

        caption = (item.get("caption") or "").strip()
        cap = QLabel(caption[:240] + ("..." if len(caption) > 240 else ""))
        cap.setWordWrap(True)
        cap.setStyleSheet("font-size: 11px; color: #444;")
        layout.addWidget(cap)

        if show_debug and item.get("debug"):
            debug = item["debug"]
            debug_lines = [
                f"s_direct={debug.get('s_direct')}",
                f"s_link={debug.get('s_link')}",
                f"p_layout={debug.get('p_layout')}",
                f"cluster_prior={debug.get('cluster_prior')}",
            ]
            debug_label = QLabel("\n".join(debug_lines))
            debug_label.setStyleSheet(
                "font-family: Consolas, monospace; font-size: 10px; color: #555;"
            )
            debug_label.setWordWrap(True)
            layout.addWidget(debug_label)


class _MethodTab(QWidget):
    def __init__(self, method_name: str, parent=None):
        super().__init__(parent)
        self.method_name = method_name
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._container = QWidget()
        self._grid = QGridLayout(self._container)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._scroll.setWidget(self._container)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._scroll)

    def set_ranked(self, ranked: list[dict[str, Any]], k: int) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        show_debug = self.method_name == "Proposed"
        for idx, row in enumerate(ranked[:k]):
            card = _FigureCard(row, show_debug=show_debug)
            self._grid.addWidget(card, idx // 2, idx % 2)


class MethodRankPanel(QWidget):
    def __init__(self, methods: list[str], parent=None):
        super().__init__(parent)
        self._tabs = QTabWidget()
        self._method_tabs: dict[str, _MethodTab] = {}
        for method in methods:
            tab = _MethodTab(method)
            self._method_tabs[method] = tab
            self._tabs.addTab(tab, method)

        layout = QVBoxLayout(self)
        title = QLabel("方法选图 (Top-K)")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title)
        layout.addWidget(self._tabs, 1)

    def set_bundle_methods(self, methods_data: dict[str, Any], k: int) -> None:
        for method_name, tab in self._method_tabs.items():
            method_block = methods_data.get(method_name, {})
            ranked = method_block.get("ranked_top10", [])
            tab.set_ranked(ranked, k)

    def current_method(self) -> str:
        widget = self._tabs.currentWidget()
        if isinstance(widget, _MethodTab):
            return widget.method_name
        return self._tabs.tabText(self._tabs.currentIndex())

    def set_default_method(self, method_name: str) -> None:
        tab = self._method_tabs.get(method_name)
        if tab is not None:
            self._tabs.setCurrentWidget(tab)

    def tab_widget(self) -> QTabWidget:
        return self._tabs
