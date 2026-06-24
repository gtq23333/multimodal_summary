from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

CASE_STUDY_ROOT = Path(__file__).resolve().parents[1]
if str(CASE_STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(CASE_STUDY_ROOT))

from app.main_window import MainWindow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 Case Study UI")
    parser.add_argument(
        "--config",
        default=str(CASE_STUDY_ROOT / "config.yaml"),
        help="case_study/config.yaml 路径",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        print(f"配置文件不存在: {config_path}", file=sys.stderr)
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(config_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
