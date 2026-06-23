from __future__ import annotations

import argparse
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(base_dir))

from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main():
    parser = argparse.ArgumentParser(description="多模态摘要标注工具")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（默认 data_annotation/config.yaml）",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else base_dir / "config.yaml"

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
