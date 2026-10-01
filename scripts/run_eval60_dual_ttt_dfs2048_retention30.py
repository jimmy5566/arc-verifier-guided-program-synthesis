#!/usr/bin/env python3
"""Thin DFS2048 identity wrapper around the proven Retention30 runner."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_eval60_dual_ttt_dfs1024_retention30 as shared


EXPERIMENT_ID = "EVAL60_DUAL_TTT_DFS2048_RETENTION30_V1"
EXPECTED_MAX_EXPANDED_NODES = 2048


def configure_shared_runner() -> None:
    shared.EXPERIMENT_ID = EXPERIMENT_ID
    shared.EXPECTED_BENCHMARK_ID = EXPERIMENT_ID
    shared.EXPECTED_MAX_EXPANDED_NODES = EXPECTED_MAX_EXPANDED_NODES
    shared.MANIFEST_EXPERIMENT_ID = "EVAL60_DUAL_TTT_DFS1024_RETENTION30_V1"
    shared.DEFAULT_SOURCE_BRANCH = "experiment/eval60-dual-ttt-dfs2048-retention30-v1"
    shared.EXPECTED_SCIENTIFIC_CONFIG_SHA256 = "db4efbe965a97111b0378b24956fd4fb78aa7dac6b077b3bd1ab3ece82b92fcf"


if __name__ == "__main__":
    configure_shared_runner()
    shared.main()
