from __future__ import annotations

import copy

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QPlainTextEdit, QScrollArea, QVBoxLayout, QWidget

from models.paper import ImageCandidate
from models.segments import (
    ImageSegment,
    Segment,
    TextSegment,
    insert_image_at_index,
    merge_adjacent_text_segments,
    remove_image_segment,
    split_text_segment,
)
from ui.widgets.draggable_image_card import MIME_IMAGE_HASH
from ui.widgets.inserted_image_block import InsertedImageBlock


class DroppableTextEdit(QPlainTextEdit):
    image_dropped = pyqtSignal(str, int)

    def __init__(self, segment_index: int, parent=None):
        super().__init__(parent)
        self.segment_index = segment_index
        self.setAcceptDrops(True)
        self.setMinimumHeight(80)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(MIME_IMAGE_HASH):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(MIME_IMAGE_HASH):
            event.acceptProposedAction()

    def dropEvent(self, event):
        raw = event.mimeData().data(MIME_IMAGE_HASH)
        if not raw:
            return
        image_hash = bytes(raw).decode("utf-8")
        cursor = self.cursorForPosition(event.position().toPoint())
        offset = cursor.position()
        self.image_dropped.emit(image_hash, offset)
        event.acceptProposedAction()


class AbstractEditor(QWidget):
    segments_changed = pyqtSignal()
    image_inserted = pyqtSignal(str)
    image_removed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.segments: list[Segment] = []
        self._candidate_lookup: dict[str, ImageCandidate] = {}
        self._undo_stack: list[list[Segment]] = []
        self._redo_stack: list[list[Segment]] = []
        self._block_sync = False
        self._text_edit_pending_undo = False

        outer = QVBoxLayout(self)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.container)
        outer.addWidget(self.scroll)

        self._text_widgets: list[DroppableTextEdit] = []
        self._image_widgets: list[InsertedImageBlock] = []

        undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_shortcut.activated.connect(self.undo)

    def set_candidate_lookup(self, candidates: list[ImageCandidate]):
        self._candidate_lookup = {c.image_hash: c for c in candidates}

    def load_segments(self, segments: list[Segment], push_undo: bool = False):
        if push_undo and self.segments:
            self._push_undo()
        self.segments = merge_adjacent_text_segments(copy.deepcopy(segments))
        if not self.segments:
            self.segments = [TextSegment(content="")]
        self._rebuild_ui()

    def get_segments(self) -> list[Segment]:
        self._sync_from_widgets()
        return copy.deepcopy(self.segments)

    def reset_to_abstract(self, abstract_text: str):
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.segments = [TextSegment(content=abstract_text)]
        self._rebuild_ui()

    def undo(self):
        if not self._undo_stack:
            return
        self._sync_from_widgets()
        self._redo_stack.append(copy.deepcopy(self.segments))
        self.segments = self._undo_stack.pop()
        self._text_edit_pending_undo = False
        self._rebuild_ui()

    def _push_undo(self):
        self._sync_from_widgets()
        self._undo_stack.append(copy.deepcopy(self.segments))
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def insert_image(self, image_hash: str, segment_index: int, offset: int = 0):
        candidate = self._candidate_lookup.get(image_hash)
        if candidate is None:
            return
        self._push_undo()
        self._sync_from_widgets()

        seg = self.segments[segment_index]
        if seg.type != "text":
            return

        if offset <= 0:
            insert_at = segment_index
        elif offset >= len(seg.content):
            insert_at = segment_index + 1
        else:
            self.segments, insert_at = split_text_segment(self.segments, segment_index, offset)

        self.segments = insert_image_at_index(self.segments, insert_at, candidate)
        self.segments = merge_adjacent_text_segments(self.segments)
        self._rebuild_ui()
        self.image_inserted.emit(image_hash)
        self.segments_changed.emit()

    def remove_image(self, image_hash: str):
        self._push_undo()
        self._sync_from_widgets()
        for i, seg in enumerate(self.segments):
            if seg.type == "image" and seg.candidate.image_hash == image_hash:
                self.segments, _ = remove_image_segment(self.segments, i)
                break
        self.segments = merge_adjacent_text_segments(self.segments)
        if not self.segments:
            self.segments = [TextSegment(content="")]
        self._rebuild_ui()
        self.image_removed.emit(image_hash)
        self.segments_changed.emit()

    def _sync_from_widgets(self):
        if self._block_sync:
            return
        new_segments: list[Segment] = []
        text_idx = 0
        image_idx = 0
        for seg in self.segments:
            if seg.type == "text":
                if text_idx < len(self._text_widgets):
                    content = self._text_widgets[text_idx].toPlainText()
                    new_segments.append(TextSegment(content=content))
                    text_idx += 1
            else:
                new_segments.append(ImageSegment(candidate=seg.candidate))
                image_idx += 1
        self.segments = merge_adjacent_text_segments(new_segments)

    def _rebuild_ui(self):
        self._block_sync = True
        self._text_edit_pending_undo = False
        while self.container_layout.count():
            item = self.container_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._text_widgets.clear()
        self._image_widgets.clear()

        for i, seg in enumerate(self.segments):
            if seg.type == "text":
                editor = DroppableTextEdit(segment_index=i)
                editor.setPlainText(seg.content)
                editor.textChanged.connect(self._on_text_changed)
                editor.image_dropped.connect(self._on_image_dropped)
                self.container_layout.addWidget(editor)
                self._text_widgets.append(editor)
            else:
                block = InsertedImageBlock(seg.candidate)
                block.remove_requested.connect(self.remove_image)
                block.setAcceptDrops(False)
                self.container_layout.addWidget(block)
                self._image_widgets.append(block)

        self._block_sync = False
        self.segments_changed.emit()

    def _on_text_changed(self):
        if self._block_sync:
            return
        if not self._text_edit_pending_undo:
            self._undo_stack.append(copy.deepcopy(self.segments))
            if len(self._undo_stack) > 50:
                self._undo_stack.pop(0)
            self._redo_stack.clear()
            self._text_edit_pending_undo = True
        self._sync_from_widgets()
        self.segments_changed.emit()

    def _on_image_dropped(self, image_hash: str, offset: int):
        sender = self.sender()
        if not isinstance(sender, DroppableTextEdit):
            return
        seg_index = self._text_widgets.index(sender)
        real_index = 0
        text_count = 0
        for i, seg in enumerate(self.segments):
            if seg.type == "text":
                if text_count == seg_index:
                    real_index = i
                    break
                text_count += 1
        self.insert_image(image_hash, real_index, offset)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(MIME_IMAGE_HASH):
            event.acceptProposedAction()
