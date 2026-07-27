#!/usr/bin/env python3
"""对比 national_competition 清洗结果与 gold corpus。"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiles.national_competition import NationalCompetitionProfile  # noqa: E402

DEFAULT_RAW_ROOT = Path(
    r"C:/Users/32780/Desktop/数模论文信息/minerU提取文件语料库/国赛"
)
DEFAULT_GOLD_ROOT = ROOT.parent / "usable_data" / "cleaned_excellent_paper_mds"
SEP = "##############"


def _plain(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _split_gold(text: str) -> tuple[str, str]:
    if SEP not in text:
        return text.strip(), ""
    a, b = text.split(SEP, 1)
    return a.strip(), b.strip()


def _prefix_match(a: str, b: str, n: int = 200) -> bool:
    def norm(t: str) -> str:
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        while lines and lines[0].startswith("#"):
            lines = lines[1:]
        return _plain("\n".join(lines))

    pa, pb = norm(a)[:n], norm(b)[:n]
    if not pa or not pb:
        return False
    m = min(len(pa), len(pb), 80)
    return pa[:m] == pb[:m]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--gold-root", default=str(DEFAULT_GOLD_ROOT))
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    gold_root = Path(args.gold_root)
    profile = NationalCompetitionProfile()

    if not gold_root.is_dir():
        print(f"gold 目录不存在: {gold_root}", file=sys.stderr)
        sys.exit(1)

    total = ok = fail = missing_raw = 0
    for gold_path in sorted(gold_root.glob("*.md")):
        paper_id = gold_path.stem
        raw_path = raw_root / paper_id / "full.md"
        if not raw_path.is_file():
            missing_raw += 1
            continue

        total += 1
        gold_abs, gold_body = _split_gold(gold_path.read_text(encoding="utf-8"))
        result = profile.clean(
            raw_path.read_text(encoding="utf-8"),
            paper_id=paper_id,
            separator=SEP,
        )
        if not result.success:
            fail += 1
            print(f"FAIL {paper_id[:24]}: {result.reason}")
            continue

        abs_ok = _prefix_match(result.abstract, gold_abs) or _plain(result.abstract)[
            :100
        ] in _plain(gold_abs)
        body_ok = _prefix_match(result.body, gold_body, n=200)
        if abs_ok and body_ok:
            ok += 1
        else:
            fail += 1
            print(
                f"DIFF {paper_id[:24]}: abs_ok={abs_ok} body_ok={body_ok} "
                f"pred_body={result.body[:60]!r}"
            )

    rate = (ok / total * 100) if total else 0
    print(
        f"evaluated={total} ok={ok} fail={fail} missing_raw={missing_raw} "
        f"match_rate={rate:.1f}%"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
