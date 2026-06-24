from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget


METRIC_LABELS = [
    ("ip@3", "IP@3"),
    ("ir@3", "IR@3"),
    ("jaccard@3", "Jaccard@3"),
    ("map", "MAP"),
    ("mrr", "MRR"),
    ("maxsim@3", "MaxSim@3"),
]


class MetricsBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._title = QLabel("指标")
        self._title.setStyleSheet("font-weight: bold; font-size: 13px;")
        self._subtitle = QLabel()
        self._subtitle.setStyleSheet("color: #666; font-size: 11px;")
        self._value_labels: dict[str, QLabel] = {}

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        for col, (key, label) in enumerate(METRIC_LABELS):
            name = QLabel(label)
            name.setStyleSheet("color: #555; font-size: 11px;")
            value = QLabel("-")
            value.setStyleSheet("font-size: 14px; font-weight: bold;")
            self._value_labels[key] = value
            grid.addWidget(name, 0, col)
            grid.addWidget(value, 1, col)

        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setLayout(grid)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._title)
        layout.addWidget(self._subtitle)
        layout.addWidget(card)

    def update_metrics(
        self,
        *,
        paper_short_id: str,
        method_name: str,
        metrics: dict[str, Any],
        n_gt: int,
        n_candidates: int,
        flags: list[dict[str, Any]] | None = None,
    ) -> None:
        self._title.setText(f"{paper_short_id} · {method_name}")
        flag_text = ""
        if flags:
            names = [f.get("name", str(f)) for f in flags]
            flag_text = f" | flags: {', '.join(names)}"
        self._subtitle.setText(
            f"GT={n_gt} · 候选={n_candidates}{flag_text}"
        )
        for key, label in self._value_labels.items():
            val = metrics.get(key)
            if val is None:
                label.setText("-")
            else:
                label.setText(f"{float(val):.3f}")
