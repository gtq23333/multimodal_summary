#!/usr/bin/env python3
"""一次性：将国赛赛题目录下的 full.md 拷贝到 usable_data/problem_mds，以文件夹名命名。"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE_ROOT = Path(
    r"C:/Users/32780/Desktop/数模论文信息/minerU提取文件语料库/部分赛题/国赛"
)
DEFAULT_DEST_ROOT = ROOT.parent / "usable_data" / "problem_mds"


def copy_problem_mds(
    source_root: Path,
    dest_root: Path,
    *,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """返回 (copied, skipped, missing_full_md)。"""
    if not source_root.is_dir():
        raise FileNotFoundError(f"源目录不存在: {source_root}")

    dest_root.mkdir(parents=True, exist_ok=True)

    copied = skipped = missing = 0

    for folder in sorted(source_root.iterdir()):
        if not folder.is_dir():
            continue

        src_md = folder / "full.md"
        dest_md = dest_root / f"{folder.name}.md"

        if dest_md.exists():
            skipped += 1
            print(f"SKIP  {dest_md.name}")
            continue

        if not src_md.is_file():
            missing += 1
            print(f"MISS  {folder.name} (无 full.md)")
            continue

        if dry_run:
            print(f"DRY   {src_md} -> {dest_md}")
        else:
            shutil.copy2(src_md, dest_md)
            print(f"COPY  {dest_md.name}")

        copied += 1

    return copied, skipped, missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="拷贝国赛赛题 full.md 到 usable_data/problem_mds（同名跳过）"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"赛题根目录（默认: {DEFAULT_SOURCE_ROOT}）",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST_ROOT,
        help=f"输出目录（默认: {DEFAULT_DEST_ROOT}）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要执行的操作，不写入文件",
    )
    args = parser.parse_args()

    copied, skipped, missing = copy_problem_mds(
        args.source.resolve(),
        args.dest.resolve(),
        dry_run=args.dry_run,
    )

    action = "将拷贝" if args.dry_run else "已拷贝"
    print(
        f"\n{action} {copied} 个，跳过 {skipped} 个，缺少 full.md {missing} 个"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
