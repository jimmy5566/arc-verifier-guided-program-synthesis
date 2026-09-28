#!/usr/bin/env python3
"""Render the compact evidence handoff after V5 generation and Gold attach."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--global-root", type=Path, default=Path("/workspace/arc2"))
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args(); root=args.run_root.resolve(); global_root=args.global_root.resolve()
    freeze=read(root/"TURBODFS_GENERATION_FROZEN.flag")
    if freeze.get("status") != "FROZEN" or int(freeze.get("primary_cells",0)) != 1068:
        raise RuntimeError("V5 generation freeze is incomplete")
    calibration=read(root/"turbodfs_v5_full_calibration.json")
    global_manifest=read(global_root/"turbodfs_v5"/"GLOBAL_ASSET_MANIFEST.json")
    gold=read(root/"turbodfs_v5_gold_summary.json") if (root/"turbodfs_v5_gold_summary.json").is_file() else {}
    db=sqlite3.connect(root/"run_state.sqlite")
    blocks=db.execute("SELECT status,COUNT(*) FROM blocks GROUP BY status").fetchall(); db.close()
    output=root/"TURBODFS_V5_HANDOFF.md"
    lines=[
        "# TurboDFS V5 handoff", "",
        "- Decoder: `TURBODFS_OPT_V5_FRONTIER_FLOOR` (explicit `SEARCH_SEMANTICS_EXTENSION`, not public-reference parity).",
        f"- Final decoder SHA256: `{read(root/'run_manifest.json').get('final_decoder_sha256')}`.",
        f"- Source commit for final collection: `{args.source_commit}`.",
        f"- Calibration: {calibration['complete_valid_cells']}/{calibration['cells']} complete-valid; zero-complete={calibration['zero_complete_rate']:.2%}; median={calibration['median_seconds_per_cell']:.3f}s; p90={calibration['p90_seconds_per_cell']:.3f}s.",
        f"- Frontier-floor activation: {calibration['frontier_floor_activation_rate']:.2%}; valid completions requiring it={calibration['fraction_valid_completion_requiring_fallback']:.2%}.",
        f"- Blocks in transactional state DB: `{blocks}`.",
        f"- Generation freeze SHA: `{freeze['analysis_ready_manifest_sha256']}`.",
        f"- Global asset manifest: `{global_root/'turbodfs_v5/GLOBAL_ASSET_MANIFEST.json'}`.",
        f"- Global asset source commit: `{global_manifest['source_commit']}`.",
        f"- Gold attached only after freeze: `{gold.get('gold_attached_after_turbodfs_freeze', False)}`.",
    ]
    if gold:
        lines += ["", "## Post-freeze outcome", ""]
        for key in ("GREEDY_ORACLE","TURBODFS_ORACLE","UNION_ORACLE","TURBODFS_RESCUED_OUTPUTS","DEPTH12_TURBO_ANYK","DEPTH24_TURBO_ANYK","DEPTH48_TURBO_ANYK","IDENTITY_TURBO_ANYK","FLIP_UD_TURBO_ANYK","TRANSPOSE_TURBO_ANYK","ANTI_TRANSPOSE_TURBO_ANYK"):
            lines.append(f"- {key}: `{gold.get(key)}`")
    output.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
