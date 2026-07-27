from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.io import load_pipeline_config  # noqa: E402
from core.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="MD 预清洗管线")
    parser.add_argument(
        "--config",
        default=str(ROOT / "config.yaml"),
        help="config.yaml 路径",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="覆盖 config 中的 profile",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只验证与报告，不写输出文件",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        print(f"配置文件不存在: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_pipeline_config(config_path)
    if args.profile:
        config.profile = args.profile

    run_pipeline(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
