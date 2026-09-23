#!/usr/bin/env python3
"""Create a portable, solution-free D1 review bundle with SHA256 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release" / "TTT24_TTT48_4PLUS4_D1_BASELINE_V1"
D1 = ROOT / "artifacts" / "eval60_4plus4_d1_selector"
LOG = ROOT / "artifacts" / "kaggle_downloads" / "ttt48_missing_cross_scores_v2" / "version2_logs.json"
REFERENCE = ROOT.parents[2] / "ARC2_TTT24_TTT48_4plus4_selector_handoff_20260924.zip"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT.parents[2] / "ARC2_D1_BASELINE_V1_REVIEW_20260924.zip")
    parser.add_argument("--reference-zip", type=Path, default=REFERENCE)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    files = {
        **{f"release/{path.name}": path for path in RELEASE.iterdir() if path.is_file()},
        "source/src/inference/selector_d1.py": ROOT / "src" / "inference" / "selector_d1.py",
        "source/src/inference/d1_release_contract.py": ROOT / "src" / "inference" / "d1_release_contract.py",
        "source/scripts/run_eval60_4plus4_d1_selector.py": ROOT / "scripts" / "run_eval60_4plus4_d1_selector.py",
        "source/scripts/audit_d1_baseline_release.py": ROOT / "scripts" / "audit_d1_baseline_release.py",
        "source/scripts/run_d1_release_4gpu.py": ROOT / "scripts" / "run_d1_release_4gpu.py",
        "source/scripts/build_d1_release_submission.py": ROOT / "scripts" / "build_d1_release_submission.py",
        "source/scripts/build_d1_release_kaggle.py": ROOT / "scripts" / "build_d1_release_kaggle.py",
        "source/scripts/replay_d1_release_contract.py": ROOT / "scripts" / "replay_d1_release_contract.py",
        "source/scripts/package_d1_release_review.py": ROOT / "scripts" / "package_d1_release_review.py",
        "source/tests/test_selector_d1.py": ROOT / "tests" / "test_selector_d1.py",
        "source/tests/test_audit_d1_baseline_release.py": ROOT / "tests" / "test_audit_d1_baseline_release.py",
        "source/tests/test_release_checklist_cpu_behavior.py": ROOT / "tests" / "test_release_checklist_cpu_behavior.py",
        "source/tests/test_release_contract_mutations.py": ROOT / "tests" / "test_release_contract_mutations.py",
        "evidence/D1_REPLAY_REPORT.json": D1 / "D1_REPLAY_REPORT.json",
        "evidence/D1_REPLAY_REPORT.md": D1 / "D1_REPLAY_REPORT.md",
        "evidence/d1_config_frozen.json": D1 / "d1_config_frozen.json",
        "evidence/baseline_replay_frozen.json": D1 / "baseline_replay_frozen.json",
        "evidence/d1_rankings_frozen.json": D1 / "d1_rankings_frozen.json",
        "evidence/d1_predictions_frozen.json": D1 / "d1_predictions_frozen.json",
        "evidence/D1_RELEASE_RUNTIME_CONFIG.json": RELEASE / "D1_RELEASE_RUNTIME_CONFIG.json",
        "evidence/NEW_CONTRACT_REPLAY.json": ROOT / "artifacts" / "d1_release_contract_replay" / "NEW_CONTRACT_REPLAY.json",
    }
    if args.reference_zip.is_file():
        files["evidence/reference_selector_handoff.zip"] = args.reference_zip
    if LOG.is_file():
        files["evidence/failed_ttt48_cross_score_version2_logs.json"] = LOG
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("required review inputs missing: " + ", ".join(missing))
    forbidden = [name for name in files if "solution" in name.lower() or "credential" in name.lower()]
    if forbidden:
        raise ValueError(f"forbidden review-bundle paths: {forbidden}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {name: _sha256(path) for name, path in sorted(files.items())}
    with tempfile.TemporaryDirectory(prefix="arc2-d1-review-") as temporary:
        stage = Path(temporary)
        for name, source in files.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        (stage / "MANIFEST.sha256.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not args.reference_zip.is_file():
            (stage / "evidence" / "REFERENCE_SELECTOR_HANDOFF_NOT_AVAILABLE.md").write_text(
                "The historical selector handoff ZIP was not present when this review package was built. "
                "The frozen D1 replay artifacts in this package remain the authoritative CPU replay evidence.\n",
                encoding="utf-8",
            )
        with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage).as_posix())
    print(json.dumps({"event": "D1_RELEASE_REVIEW_BUNDLE_COMPLETE", "path": str(args.output), "sha256": _sha256(args.output), "file_count": len(manifest) + 1}, sort_keys=True))


if __name__ == "__main__":
    main()
