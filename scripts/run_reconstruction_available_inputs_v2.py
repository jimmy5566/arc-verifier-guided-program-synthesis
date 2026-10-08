#!/usr/bin/env python3
"""Detached-job launcher guard for the distinct available-input reconstruction condition.

It validates a final read-only preflight and consumes a Director-bound nonce
before handing control to the frozen worker command.  It does not inherit the
closed Foundation-V2 V1 launch contract.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from reconstruction_available_inputs_v2_authorization import read_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--directive", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--consumption-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    preflight, _ = read_json(args.preflight)
    contract, contract_sha = read_json(args.contract)
    if preflight.get("status") != "PASS" or preflight.get("binding_sha256") != contract.get("launch_binding_sha256"):
        raise RuntimeError("FINAL_PREFLIGHT_REQUIRED")
    if not args.command or args.command[0] != "--":
        parser.error("worker command must follow --")
    consume = [sys.executable, str(Path(__file__).with_name("consume_reconstruction_available_inputs_v2_authorization.py")), "--contract", str(args.contract), "--gate", str(args.gate), "--directive", str(args.directive), "--consumption-dir", str(args.consumption_dir)]
    subprocess.run(consume, check=True)
    # Receipt/accounting wrappers are invoked by the detached worker wrapper
    # named in the frozen binding.  This guard intentionally does no model work.
    return subprocess.run(args.command[1:], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
