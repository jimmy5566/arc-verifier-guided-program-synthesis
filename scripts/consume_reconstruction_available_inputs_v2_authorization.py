#!/usr/bin/env python3
"""Consume a qualifying V2 launch authorization once, immediately before dispatch."""
from __future__ import annotations

import argparse
from pathlib import Path

from reconstruction_available_inputs_v2_authorization import consume_once, read_json, validate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--directive", type=Path, required=True)
    parser.add_argument("--consumption-dir", type=Path, required=True)
    args = parser.parse_args()
    contract, contract_sha = read_json(args.contract)
    directive, directive_sha = read_json(args.directive)
    gate, _ = read_json(args.gate)
    validate(contract, contract_sha, directive, directive_sha, gate)
    receipt = consume_once(args.consumption_dir, contract, contract_sha, directive_sha)
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
