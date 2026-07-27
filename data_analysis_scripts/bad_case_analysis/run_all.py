#!/usr/bin/env python3
"""Run full bad-case complementarity analysis pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _run(script: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(SCRIPT_DIR / script), *extra_args]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bad-case analysis pipeline")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--trial", default="trial_31")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--with-fusion", action="store_true", help="Also run fusion method eval + viz")
    parser.add_argument("--with-union-pool", action="store_true", help="Also run union pool overlap analysis")
    args = parser.parse_args()

    common: list[str] = ["--trial", args.trial]
    if args.config:
        common.extend(["--config", str(args.config)])
    if args.output_dir:
        common.extend(["--output-dir", str(args.output_dir)])

    if not args.skip_export:
        export_args = list(common)
        if args.force_export:
            export_args.append("--force")
        if args.skip_clip:
            export_args.append("--skip-clip")
        _run("export_rankings.py", export_args)

    _run("analyze_complementarity.py", common)
    _run("analyze_ablation_modules.py", common)

    if args.with_union_pool:
        _run("analyze_union_pool.py", common)

    report_dir = SCRIPT_DIR / "reports" / args.trial

    if args.with_fusion:
        src_root = SCRIPT_DIR.parents[1] / "src"
        fusion_args = [
            str(src_root / "scripts" / "evaluate_fusion_methods.py"),
            "--trial",
            args.trial,
        ]
        if args.config:
            fusion_args.extend(["--config", str(args.config)])
        if args.output_dir:
            fusion_args.extend(["--output-dir", str(args.output_dir)])
        print(">>", " ".join([sys.executable, *fusion_args]))
        subprocess.run([sys.executable, *fusion_args], check=True)

        plot_args = [
            str(src_root / "scripts" / "plot_fusion_methods.py"),
            str(report_dir / "fusion"),
        ]
        print(">>", " ".join([sys.executable, *plot_args]))
        subprocess.run([sys.executable, *plot_args], check=True)

    _build_final_report(report_dir)
    print(f"Done. Reports: {report_dir}")


def _build_final_report(report_dir: Path) -> None:
    parts = []
    for name in ("complementarity_summary.md", "ablation_summary.md", "union_pool/union_pool_summary.md"):
        path = report_dir / name
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8"))

    if not parts:
        return

    md = "# Stage-2 Bad Case Complementarity Report\n\n" + "\n".join(parts)
    (report_dir / "report.md").write_text(md, encoding="utf-8")

    html_parts = [
        "<html><head><meta charset='utf-8'><title>Bad Case Report</title>",
        "<style>body{font-family:sans-serif;max-width:960px;margin:2em auto;line-height:1.5}</style>",
        "</head><body>",
        "<h1>Stage-2 Bad Case Complementarity Report</h1>",
    ]
    for name in ("complementarity_summary.md", "ablation_summary.md", "union_pool/union_pool_summary.md"):
        path = report_dir / name
        if path.is_file():
            html_parts.append(f"<pre>{path.read_text(encoding='utf-8')}</pre>")
    for fig_subdir in ("figures", "union_pool/figures"):
        fig_dir = report_dir / fig_subdir
        if fig_dir.is_dir():
            html_parts.append(f"<h2>Figures ({fig_subdir})</h2>")
            for png in sorted(fig_dir.glob("*.png")):
                html_parts.append(
                    f"<h3>{png.stem}</h3><img src='{fig_subdir}/{png.name}' style='max-width:100%'/>"
                )
    html_parts.append("</body></html>")
    (report_dir / "report.html").write_text("\n".join(html_parts), encoding="utf-8")


if __name__ == "__main__":
    main()
