"""Stage the isolated Aug8 strict-coverage Kaggle diagnostic assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "8bf757bc51c8250ad3431e30a7a28febbad89d64"
DATASET_SLUG = "arc2-aug8-strict-zero-lb-source"
KERNEL_SLUG = "arc2-arc-prize-2026-aug8-strict-diagnostic"
COMPETITION = "arc-prize-2026-arc-agi-2"
MODEL_SOURCE = "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"
DOCKER_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--owner", default="jimmy5566")
    parser.add_argument("--source-commit", default="HEAD")
    args = parser.parse_args()
    if args.stage.exists():
        raise FileExistsError(f"refusing to overwrite staging directory: {args.stage}")
    if _git("rev-parse", BASELINE_COMMIT) != BASELINE_COMMIT:
        raise RuntimeError("the requested Frozen60 baseline commit is unavailable")
    source_commit = _git("rev-parse", args.source_commit)
    dataset = args.stage / "dataset"
    kernel = args.stage / "kernel"
    dataset.mkdir(parents=True)
    kernel.mkdir(parents=True)
    archive = dataset / "ARC2.tar.gz"
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "--prefix=ARC2/", "--output", str(archive), source_commit],
        cwd=ROOT,
        check=True,
    )
    with tarfile.open(archive, "r:gz") as value:
        names = value.getnames()
    bad = [name for name in names if "solution" in Path(name).name.lower()]
    if bad or not any(name.endswith("scripts/build_strict_public_lb_submission.py") for name in names):
        raise RuntimeError(f"source archive violates strict-source requirements: {bad}")
    dataset_metadata = {
        "title": "ARC2 Aug8 Strict Zero-LB Source",
        "subtitle": "Private strict-coverage diagnostic source; no ARC solutions",
        "description": "Exact V38/V8 core with Aug8-only and B-only strict finalization. No solutions or fallback submission transport.",
        "id": f"{args.owner}/{DATASET_SLUG}",
        "licenses": [{"name": "other"}],
    }
    _write_json(dataset / "dataset-metadata.json", dataset_metadata)
    notebook = kernel / f"{KERNEL_SLUG}.ipynb"
    subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "build_aug8_strict_zero_lb_notebook.py"),
            "--base-builder", str(ROOT / "scripts" / "build_final_arc_prize_2026_submission_notebook.py"),
            "--output", str(notebook),
        ],
        cwd=ROOT,
        check=True,
    )
    code = "".join(json.loads(notebook.read_text(encoding="utf-8"))["cells"][0]["source"])
    for forbidden in ("KAGGLE_IS_COMPETITION_RERUN", "FAST_COMMIT_MODE", "--allow-deadline-partial", '"--external-augmentation-count", "32"'):
        if forbidden in code:
            raise RuntimeError(f"notebook retains forbidden behavior: {forbidden}")
    for required in ("AUG8_ALWAYS_PRODUCTION_MODE", "STRICT_HIDDEN_COVERAGE_FAILED", '"--external-augmentation-count", "8"', "build_strict_public_lb_submission.py"):
        if required not in code:
            raise RuntimeError(f"notebook lacks required behavior: {required}")
    kernel_metadata = {
        "id": f"{args.owner}/{KERNEL_SLUG}",
        "title": "ARC2 Aug8 Strict Zero-LB Diagnostic",
        "code_file": notebook.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [f"{args.owner}/{DATASET_SLUG}"],
        "kernel_sources": [],
        "competition_sources": [COMPETITION],
        "model_sources": [MODEL_SOURCE],
        "docker_image": DOCKER_IMAGE,
        "machine_shape": "NvidiaL4",
    }
    _write_json(kernel / "kernel-metadata.json", kernel_metadata)
    identity = {
        "frozen_baseline_commit": BASELINE_COMMIT,
        "diagnostic_source_commit": source_commit,
        "source_archive_sha256": _sha256(archive),
        "notebook_sha256": _sha256(notebook),
        "augmentation_count": 8,
        "search_beams": 1,
        "generation_micro_batch_size": 1,
        "likelihood_micro_batch_size": 1,
        "worker_count": 4,
        "model_source": MODEL_SOURCE,
        "competition": COMPETITION,
        "strict_coverage": {
            "expected_tasks": 240,
            "model_tasks": 240,
            "b_selected": 240,
            "a_fallback": 0,
            "identity_fallback": 0,
            "failed": 0,
            "unfinished": 0,
        },
    }
    _write_json(args.stage / "AUG8_STRICT_IDENTITY.json", identity)
    print(json.dumps({"dataset": dataset_metadata["id"], "kernel": kernel_metadata["id"], **identity}, sort_keys=True))


if __name__ == "__main__":
    main()
