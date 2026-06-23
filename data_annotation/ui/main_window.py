from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from export.annotation_exporter import export_annotation
from export.annotation_loader import restore_from_file, used_image_hashes
from loaders.paper_loader import PaperLoader
from storage.progress_tracker import ProgressTracker
from ui.abstract_editor import AbstractEditor
from ui.image_pool_panel import ImagePoolPanel


class MainWindow(QMainWindow):
    def __init__(self, config_path: Path):
        super().__init__()
        self.config_path = config_path
        self.loader = PaperLoader(config_path)
        self.config = self.loader.config

        base = config_path.parent
        paths = self.config["paths"]
        self.annotation_output_dir = (base / paths["annotation_output_dir"]).resolve()
        self.progress_file = (base / paths["progress_file"]).resolve()
        self.context_window_chars = self.config["annotation"]["context_window_chars"]
        self.tool_version = self.config.get("tool", {}).get("version", "0.1.0")
        self.filter_strategy_name = self.config.get("image_filter", {}).get("strategy", "body_with_caption")

        self.paper_ids = self.loader.list_paper_ids()
        self.progress = ProgressTracker.load_or_create(
            self.progress_file,
            str(self.loader.md_corpus_root),
            self.paper_ids,
        )
        self.current_index = min(self.progress.current_index, max(0, len(self.paper_ids) - 1))
        self.current_paper = None
        self.original_abstract = ""

        self.setWindowTitle("多模态摘要标注工具")
        self.resize(1400, 900)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("上一篇")
        self.next_btn = QPushButton("下一篇")
        self.progress_label = QLabel()
        self.paper_label = QLabel()
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        nav.addWidget(self.progress_label, 1)
        nav.addWidget(self.paper_label, 2)
        root.addLayout(nav)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.editor = AbstractEditor()
        self.pool = ImagePoolPanel()
        splitter.addWidget(self.editor)
        splitter.addWidget(self.pool)
        splitter.setSizes([700, 700])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.splitterMoved.connect(lambda _pos, _index: self.pool.schedule_layout_update())
        root.addWidget(splitter, 1)

        actions = QHBoxLayout()
        self.undo_btn = QPushButton("撤销 (Ctrl+Z)")
        self.reset_btn = QPushButton("重置本篇")
        self.skip_btn = QPushButton("跳过")
        self.confirm_btn = QPushButton("确定标注并下一篇")
        self.confirm_btn.setStyleSheet("font-weight: bold;")
        actions.addWidget(self.undo_btn)
        actions.addWidget(self.reset_btn)
        actions.addStretch()
        actions.addWidget(self.skip_btn)
        actions.addWidget(self.confirm_btn)
        root.addLayout(actions)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.prev_btn.clicked.connect(self.go_prev)
        self.next_btn.clicked.connect(self.go_next)
        self.undo_btn.clicked.connect(self.editor.undo)
        self.reset_btn.clicked.connect(self.reset_current)
        self.skip_btn.clicked.connect(self.skip_current)
        self.confirm_btn.clicked.connect(self.confirm_and_next)
        self.editor.image_inserted.connect(self._on_image_inserted)
        self.editor.image_removed.connect(self._on_image_removed)
        self.editor.segments_changed.connect(self._sync_pool_from_segments)

        if self.paper_ids:
            self.load_paper_at(self.current_index)
        else:
            QMessageBox.warning(self, "警告", "未找到任何 MD 论文文件")

    def load_paper_at(self, index: int):
        if not self.paper_ids or index < 0 or index >= len(self.paper_ids):
            return

        self.current_index = index
        self.progress.set_current_index(index)
        paper_id = self.paper_ids[index]
        paper, err = self.loader.load_paper(paper_id)

        if err:
            self.progress.mark_skipped(paper_id, err.reason)
            self.status.showMessage(f"跳过 {paper_id}: {err.reason}", 5000)
            if index + 1 < len(self.paper_ids):
                self.load_paper_at(index + 1)
            return

        self.current_paper = paper
        self.original_abstract = paper.abstract_text
        self.editor.set_candidate_lookup(paper.filtered_candidates)

        ann_path = self.progress.annotation_path(self.annotation_output_dir, paper_id)
        if ann_path.is_file():
            segments = restore_from_file(paper, ann_path)
            self.editor.load_segments(segments)
            self.status.showMessage(f"已加载已有标注: {ann_path.name}", 3000)
        else:
            self.editor.reset_to_abstract(paper.abstract_text)

        self.pool.set_candidates(paper.filtered_candidates, used_image_hashes(self.editor.get_segments()))
        self._update_nav_labels()

    def _update_nav_labels(self):
        completed = len(self.progress.completed)
        total = len(self.paper_ids)
        idx = self.current_index + 1
        pid = self.current_paper.paper_id if self.current_paper else "-"
        self.progress_label.setText(f"进度: {idx}/{total} | 已完成: {completed}")
        self.paper_label.setText(f"paper_id: {pid}")

    def _on_image_inserted(self, image_hash: str):
        self.pool.remove_candidate(image_hash)

    def _on_image_removed(self, image_hash: str):
        self.pool.restore_candidate(image_hash)

    def _sync_pool_from_segments(self):
        if not self.current_paper:
            return
        used = used_image_hashes(self.editor.get_segments())
        self.pool.set_candidates(self.current_paper.filtered_candidates, used)

    def go_prev(self):
        if self.current_index > 0:
            self.load_paper_at(self.current_index - 1)

    def go_next(self):
        if self.current_index + 1 < len(self.paper_ids):
            self.load_paper_at(self.current_index + 1)

    def reset_current(self):
        if not self.current_paper:
            return
        reply = QMessageBox.question(self, "确认", "重置将清除本篇所有插入，是否继续？")
        if reply == QMessageBox.StandardButton.Yes:
            self.editor.reset_to_abstract(self.original_abstract)
            self.pool.set_candidates(self.current_paper.filtered_candidates, set())

    def skip_current(self):
        if not self.current_paper:
            return
        self.progress.mark_skipped(self.current_paper.paper_id, "user_skipped")
        self.go_next()

    def confirm_and_next(self):
        if not self.current_paper:
            return

        segments = self.editor.get_segments()
        out_path = export_annotation(
            output_dir=self.annotation_output_dir,
            paper=self.current_paper,
            segments=segments,
            original_abstract=self.original_abstract,
            filter_strategy=self.filter_strategy_name,
            context_window_chars=self.context_window_chars,
            tool_version=self.tool_version,
        )
        self.progress.mark_completed(self.current_paper.paper_id)
        self.status.showMessage(f"已保存: {out_path.name}", 5000)

        next_index = self.current_index + 1
        if next_index < len(self.paper_ids):
            self.load_paper_at(next_index)
        else:
            QMessageBox.information(self, "完成", "所有论文已处理完毕")
            self._update_nav_labels()
