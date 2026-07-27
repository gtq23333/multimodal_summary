from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.data_loader import (
    CaseStudyConfig,
    load_case_study_config,
    load_manifest,
    load_paper_bundle,
)
from app.widgets.gt_sequence_panel import GtSequencePanel
from app.widgets.metrics_bar import MetricsBar
from app.widgets.method_rank_panel import MethodRankPanel


class MainWindow(QMainWindow):
    def __init__(self, config_path: Path):
        super().__init__()
        self.config_path = config_path.resolve()
        self.cs_config: CaseStudyConfig = load_case_study_config(self.config_path)
        self.manifest = load_manifest(self.cs_config.data_dir)
        self.current_k = self.cs_config.default_k
        self.current_bundle: dict | None = None

        self.setWindowTitle("Stage-2 Case Study")
        self.resize(1400, 900)
        self.setMinimumSize(1024, 640)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("论文:"))
        self.paper_combo = QComboBox()
        self.paper_combo.setMinimumWidth(260)
        for paper in self.manifest.papers:
            self.paper_combo.addItem(paper["short_id"], paper["paper_id"])
        toolbar.addWidget(self.paper_combo)

        self.refresh_btn = QPushButton("刷新数据")
        toolbar.addWidget(self.refresh_btn)

        toolbar.addWidget(QLabel("Top-K:"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, self.manifest.max_rank)
        self.k_spin.setValue(self.current_k)
        if self.cs_config.k_options:
            self.k_spin.setSingleStep(
                self.cs_config.k_options[1] - self.cs_config.k_options[0]
                if len(self.cs_config.k_options) > 1
                else 1
            )
        toolbar.addWidget(self.k_spin)
        toolbar.addStretch(1)
        root.addLayout(toolbar)

        self.metrics_bar = MetricsBar()
        root.addWidget(self.metrics_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.gt_panel = GtSequencePanel()
        self.method_panel = MethodRankPanel(self.manifest.methods)
        splitter.addWidget(self.gt_panel)
        splitter.addWidget(self.method_panel)
        splitter.setSizes([700, 700])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        self.paper_combo.currentIndexChanged.connect(self._on_paper_changed)
        self.refresh_btn.clicked.connect(self._reload_manifest_and_paper)
        self.k_spin.valueChanged.connect(self._on_k_changed)
        self.method_panel.tab_widget().currentChanged.connect(self._update_metrics)

        if self.paper_combo.count() > 0:
            self.method_panel.set_default_method(self.cs_config.default_method)
            self._load_paper(self.paper_combo.currentData())
        else:
            QMessageBox.warning(self, "警告", "manifest 中无论文数据，请先运行 export 脚本")

    def _reload_manifest_and_paper(self) -> None:
        try:
            self.manifest = load_manifest(self.cs_config.data_dir)
            current_id = self.paper_combo.currentData()
            self.paper_combo.blockSignals(True)
            self.paper_combo.clear()
            for paper in self.manifest.papers:
                self.paper_combo.addItem(paper["short_id"], paper["paper_id"])
            self.paper_combo.blockSignals(False)
            if current_id:
                idx = self.paper_combo.findData(current_id)
                if idx >= 0:
                    self.paper_combo.setCurrentIndex(idx)
            if self.paper_combo.count() > 0:
                self._load_paper(self.paper_combo.currentData())
        except FileNotFoundError as exc:
            QMessageBox.warning(self, "数据缺失", str(exc))

    def _on_paper_changed(self, _index: int) -> None:
        paper_id = self.paper_combo.currentData()
        if paper_id:
            self._load_paper(paper_id)

    def _on_k_changed(self, value: int) -> None:
        self.current_k = value
        if self.current_bundle:
            self.method_panel.set_bundle_methods(
                self.current_bundle.get("methods", {}),
                self.current_k,
            )

    def _load_paper(self, paper_id: str) -> None:
        try:
            bundle = load_paper_bundle(self.cs_config.data_dir, paper_id)
        except FileNotFoundError as exc:
            QMessageBox.warning(self, "数据缺失", str(exc))
            return

        saved_size = self.size()
        self.current_bundle = bundle
        self.gt_panel.set_sequence(bundle.get("multimodal_sequence", []))
        self.method_panel.set_bundle_methods(bundle.get("methods", {}), self.current_k)
        self._update_metrics()
        self.resize(saved_size)

    def _update_metrics(self) -> None:
        if not self.current_bundle:
            return
        method_name = self.method_panel.current_method()
        method_block = self.current_bundle.get("methods", {}).get(method_name, {})
        self.metrics_bar.update_metrics(
            paper_short_id=self.current_bundle.get("short_id", ""),
            method_name=method_name,
            metrics=method_block.get("metrics", {}),
            n_gt=int(self.current_bundle.get("n_ground_truth", 0)),
            n_candidates=int(self.current_bundle.get("n_candidates", 0)),
            flags=method_block.get("flags"),
        )
