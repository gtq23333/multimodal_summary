#!/usr/bin/env python3
"""Anchor git commits with OpenTimestamps (Bitcoin-backed, third-party verifiable).

This does NOT replace legally certified TSA timestamps (RFC 3161 / 国内可信时间戳).
It provides a cryptographic proof that a file digest existed no later than the
Bitcoin block time embedded in the upgraded .ots proof.
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

from opentimestamps.calendar import RemoteCalendar
from opentimestamps.core.op import OpAppend, OpSHA256
from opentimestamps.core.serialize import StreamSerializationContext
from opentimestamps.core.timestamp import DetachedTimestampFile, make_merkle_tree

DEFAULT_CALENDARS = [
    "https://a.pool.opentimestamps.org",
    "https://b.pool.opentimestamps.org",
    "https://a.pool.eternitywall.com",
    "https://ots.btc.catallaxy.com",
]

REPO_URL = "https://github.com/gtq23333/multimodal_summary.git"


def git_lines(*args: str, cwd: Path) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def commit_record(commit: str, repo_root: Path) -> str:
    meta = git_lines(
        "show",
        "-s",
        "--format=%H%n%T%n%an <%ae>%n%ci%n%s",
        commit,
        cwd=repo_root,
    )
    if len(meta) < 5:
        raise RuntimeError(f"Unexpected git show output for {commit}: {meta!r}")
    full_hash, tree_hash, author, committer_date, subject = meta[:5]
    return (
        f"repository: {REPO_URL}\n"
        f"commit: {full_hash}\n"
        f"tree: {tree_hash}\n"
        f"author: {author}\n"
        f"committer_date: {committer_date}\n"
        f"subject: {subject}\n"
    )


def submit_to_calendars(
    merkle_tip,
    calendar_urls: list[str],
    *,
    timeout: int,
    min_replies: int,
) -> None:
    q: Queue = Queue()
    for url in calendar_urls:
        threading.Thread(
            target=_submit_one,
            args=(url, merkle_tip.msg, q, timeout),
            daemon=True,
        ).start()

    start = time.time()
    merged = 0
    for _ in calendar_urls:
        remaining = max(0, timeout - (time.time() - start))
        try:
            result = q.get(block=True, timeout=remaining)
        except Empty:
            continue
        if isinstance(result, Exception):
            print(f"  calendar error: {result}", file=sys.stderr)
            continue
        merkle_tip.merge(result)
        merged += 1

    if merged < min_replies:
        raise RuntimeError(
            f"Need at least {min_replies} calendar reply(s), got {merged}"
        )


def _submit_one(url: str, msg: bytes, q: Queue, timeout: int) -> None:
    try:
        calendar = RemoteCalendar(url, user_agent="multimodal-summary-ots/1.0")
        q.put(calendar.submit(msg, timeout=timeout))
    except Exception as exc:
        q.put(exc)


def stamp_file(path: Path, calendar_urls: list[str], *, timeout: int, min_replies: int) -> None:
    ots_path = path.with_suffix(path.suffix + ".ots")
    if ots_path.exists():
        print(f"skip (exists): {ots_path.name}")
        return

    with path.open("rb") as fd:
        file_timestamp = DetachedTimestampFile.from_fd(OpSHA256(), fd)

    nonce_appended = file_timestamp.timestamp.ops.add(OpAppend(os.urandom(16)))
    merkle_tip = nonce_appended.ops.add(OpSHA256())

    print(f"stamping: {path.name}")
    submit_to_calendars(
        merkle_tip,
        calendar_urls,
        timeout=timeout,
        min_replies=min_replies,
    )

    with ots_path.open("xb") as out_fd:
        ctx = StreamSerializationContext(out_fd)
        file_timestamp.serialize(ctx)
    print(f"  wrote: {ots_path}")


def upgrade_pending(ots_path: Path, calendar_urls: list[str], *, timeout: int) -> bool:
    from opentimestamps.core.notary import PendingAttestation
    from opentimestamps.core.serialize import StreamDeserializationContext
    from opentimestamps.core.timestamp import Timestamp

    data = ots_path.read_bytes()
    ctx = StreamDeserializationContext(io.BytesIO(data))
    detached = DetachedTimestampFile.deserialize(ctx)
    stamp = detached.timestamp
    changed = False

    for msg, attestation in stamp.all_attestations():
        if not isinstance(attestation, PendingAttestation):
            continue
        uri = attestation.uri
        if uri not in calendar_urls:
            calendar_urls = [uri, *calendar_urls]
        calendar = RemoteCalendar(uri, user_agent="multimodal-summary-ots/1.0")
        try:
            upgraded = calendar.get_timestamp(msg, timeout=timeout)
        except Exception:
            continue
        stamp.merge(upgraded)
        changed = True

    if changed:
        buf = io.BytesIO()
        ctx_out = StreamSerializationContext(buf)
        detached.serialize(ctx_out)
        ots_path.write_bytes(buf.getvalue())
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "timestamps" / "commits",
    )
    parser.add_argument(
        "--ref",
        default="HEAD",
        help="Git ref to stamp (default: HEAD). Use --all-commits for full history.",
    )
    parser.add_argument(
        "--all-commits",
        action="store_true",
        help="Stamp every commit reachable from --ref.",
    )
    parser.add_argument(
        "--calendar",
        action="append",
        dest="calendars",
        default=[],
        help="OpenTimestamps calendar URL (repeatable).",
    )
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--min-replies", type=int, default=2)
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Try upgrading pending .ots proofs against calendars.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    calendars = args.calendars or list(DEFAULT_CALENDARS)

    if args.all_commits:
        commits = git_lines("rev-list", args.ref, cwd=repo_root)
    else:
        commits = git_lines("rev-parse", args.ref, cwd=repo_root)

    manifest_index = out_dir.parent / "manifest.txt"
    manifest_lines = [
        f"generated_at_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"repository: {REPO_URL}",
        f"calendar_servers: {', '.join(calendars)}",
        "commits:",
    ]

    for commit in commits:
        short = commit[:12]
        record_path = out_dir / f"{short}.commit"
        record_path.write_text(commit_record(commit, repo_root), encoding="utf-8")
        manifest_lines.append(f"  - {commit}  file={record_path.name}")
        try:
            stamp_file(
                record_path,
                calendars,
                timeout=args.timeout,
                min_replies=args.min_replies,
            )
        except RuntimeError as exc:
            print(f"FAILED {short}: {exc}", file=sys.stderr)
            sys.exit(1)

    manifest_index.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"wrote manifest: {manifest_index}")

    if args.upgrade:
        print("upgrading pending proofs (may take minutes until Bitcoin confirms)...")
        for ots_path in sorted(out_dir.glob("*.commit.ots")):
            if upgrade_pending(ots_path, calendars, timeout=args.timeout):
                print(f"  upgraded: {ots_path.name}")


if __name__ == "__main__":
    main()
