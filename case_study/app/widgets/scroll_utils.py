from __future__ import annotations

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import QScrollArea, QSizePolicy, QWidget


def configure_vertical_scroll(scroll: QScrollArea, container: QWidget) -> None:
    """滚动区只纵向滚动，避免内容 sizeHint 把主窗口撑大。"""
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)


def sync_scroll_container_width(scroll: QScrollArea, container: QWidget) -> None:
    width = scroll.viewport().width()
    if width <= 0:
        return
    container.setMinimumWidth(width)
    container.setMaximumWidth(width)


class ScrollWidthSyncMixin:
    """将 scroll 内层容器宽度锁定为 viewport 宽度。"""

    _width_scroll: QScrollArea | None = None
    _width_container: QWidget | None = None

    def bind_scroll_width(self, scroll: QScrollArea, container: QWidget) -> None:
        configure_vertical_scroll(scroll, container)
        self._width_scroll = scroll
        self._width_container = container
        scroll.viewport().installEventFilter(self)
        sync_scroll_container_width(scroll, container)

    def sync_scroll_width(self) -> None:
        if self._width_scroll is None or self._width_container is None:
            return
        sync_scroll_container_width(self._width_scroll, self._width_container)

    def eventFilter(self, obj, event):  # type: ignore[override]
        if (
            self._width_scroll is not None
            and obj is self._width_scroll.viewport()
            and event.type() == QEvent.Type.Resize
        ):
            self.sync_scroll_width()
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):  # type: ignore[override]
        super().resizeEvent(event)
        self.sync_scroll_width()
