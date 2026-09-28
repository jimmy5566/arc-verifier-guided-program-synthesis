#!/usr/bin/env python3
"""Create and publish periodic Gold-authorized V5 analysis snapshots.

The watcher is intentionally outside the generator path.  It opens only the
local recovered state database in SQLite read-only mode, waits for a fixed
number of completed heavy cells, and then invokes the existing analysis-only
snapshot tool.  It never launches, stops, or changes a V5 worker.
"""
from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time


def heavy_done(db_path: Path) -> tuple[int, int]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        completed = connection.execute(
            """SELECT COUNT(*) FROM cells
                 WHERE status = 'DONE'
                   AND execution_engine = 'INDEPENDENT_SINGLE_PER_GPU_REPAIR'"""
        ).fetchone()[0]
        pending = connection.execute(
            """SELECT COUNT(*) FROM cells
                 WHERE status IN ('RETRY_HEAVY_OOM', 'RETRY_TRANSIENT')"""
        ).fetchone()[0]
    finally:
        connection.close()
    return int(completed), int(pending)


def push(repo: Path, report_dir: Path, message: str) -> None:
    subprocess.run(["git", "-C", str(repo), "add", "-f", str(report_dir)], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", message], check=True)
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise RuntimeError("GH_TOKEN is required to publish the checkpoint")
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    subprocess.run(
        ["git", "-C", str(repo), "-c", f"http.extraheader=AUTHORIZATION: Basic {basic}",
         "push", "origin", "HEAD:transfer/turbodfs-opt-v2-repair"],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--greedy-parquet", type=Path, required=True)
    parser.add_argument("--initial-heavy-done", type=int, required=True)
    parser.add_argument("--step", type=int, default=25)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    repo = args.repo.resolve()
    analyzer = repo / "scripts" / "analyze_eval60_v5_nonblind_interim.py"
    report_root = repo / "reports" / "eval60_v5_nonblind_interim" / "checkpoints"
    target = args.initial_heavy_done + args.step
    while True:
        completed, pending = heavy_done(args.state_db)
        if completed >= target or (pending == 0 and completed >= args.initial_heavy_done):
            label = f"HEAVY_{completed:03d}_DONE"
            report_dir = report_root / label
            if report_dir.exists():
                target += args.step
                continue
            subprocess.run(
                [sys.executable, str(analyzer), "--run", str(args.run),
                 "--state-db", str(args.state_db), "--solutions", str(args.solutions),
                 "--greedy-parquet", str(args.greedy_parquet), "--repo", str(repo),
                 "--report-dir", str(report_dir)],
                check=True,
            )
            push(repo, report_dir, f"analysis: checkpoint nonblind V5 heavy {completed}")
            if pending == 0:
                return
            target += args.step
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
