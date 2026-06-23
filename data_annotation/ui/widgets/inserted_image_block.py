from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class InsertedImageBlock(QWidget):
    remove_requested = pyqtSignal(str)

    def __init__(self, candidate, parent=None):
        super().__init__(parent)
        self.candidate = candidate
        self.setStyleSheet(
            "InsertedImageBlock { border: 2px dashed #4a90d9; border-radius: 6px; background: #f0f7ff; margin: 4px 0; }"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        thumb = QLabel()
        thumb.setFixedSize(100, 80)
        pixmap = QPixmap(candidate.abs_image_path)
        if not pixmap.isNull():
            thumb.setPixmap(
                pixmap.scaled(100, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
        else:
            thumb.setText(candidate.image_hash[:8])
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)

        info = QVBoxLayout()
        type_label = "[表]" if candidate.is_table else "[图]"
        cap = QLabel(f"{type_label} {candidate.caption}")
        cap.setWordWrap(True)
        hash_label = QLabel(candidate.image_hash[:16] + "...")
        hash_label.setStyleSheet("font-size: 9px; color: #888;")
        info.addWidget(cap)
        info.addWidget(hash_label)

        remove_btn = QPushButton("移除")
        remove_btn.setFixedWidth(60)
        remove_btn.clicked.connect(lambda: self.remove_requested.emit(self.candidate.image_hash))

        layout.addWidget(thumb)
        layout.addLayout(info, 1)
        layout.addWidget(remove_btn)
