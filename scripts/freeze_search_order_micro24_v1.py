"""Freeze a compact, target-blind archive of terminal Micro24 E1 generation.

The P0/P1/P2 raw outputs remain on the RunPod because normal GitHub Git
transport is not a raw-output store.  This tool first re-verifies each full
generation ledger (including the raw outputs), then copies the compact
receipts, checkpoints, telemetry and ledgers into a Git-sized archive.  The
archive includes its own copy-parity record and SHA-256 ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


EXPERIMENT = "SEARCH_ORDER_MICRO24_V1_E1"
POLICIES = {"P0": "CURRENT_DFS", "P1": "FAIR_DFS_Q64", "P2": "REGRET_BAND_FAIR_Q64"}
COMPACT_DIRECTORIES = ("OUTPUT_RECEIPTS", "OUTPUT_CHECKPOINTS")
CONTROL_RECEIPTS = frozenset({"ARCHIVE_HASHES.json", "ARCHIVE_HASH_VERIFICATION.json", "COMPACT_ARCHIVE_FREEZE.json"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_canonical(value), encoding="utf-8")
    temporary.replace(path)


def _bytes_in(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _verify_source(label: str, source: Path) -> dict[str, Any]:
    generation_path = source / "GENERATION_HASHES.json"
    verification_path = source / "GENERATION_HASH_VERIFICATION.json"
    freeze_path = source / "GENERATION_FREEZE.json"
    for path in (generation_path, verification_path, freeze_path):
        if not path.is_file():
            raise RuntimeError(f"{label}_MISSING_REQUIRED_RECEIPT:{path.name}")

    generation = _read(generation_path)
    freeze = _read(freeze_path)
    stored_verification = _read(verification_path)
    expected = generation.get("files")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError(f"{label}_INVALID_GENERATION_LEDGER")
    if freeze.get("status") != "FROZEN" or freeze.get("target_blind") is not True or freeze.get("gold_loaded") is not False:
        raise RuntimeError(f"{label}_INVALID_FREEZE_RECEIPT")
    if stored_verification.get("status") != "PASS":
        raise RuntimeError(f"{label}_STORED_GENERATION_HASH_FAIL")
    mismatches = [
        {"path": relative, "expected": digest, "actual": _sha256(source / relative) if (source / relative).is_file() else None}
        for relative, digest in sorted(expected.items())
        if not (source / relative).is_file() or _sha256(source / relative) != digest
    ]
    if mismatches:
        raise RuntimeError(f"{label}_GENERATION_HASH_FAIL:{_canonical(mismatches[:3]).strip()}")

    raw_count = len(list((source / "RAW_OUTPUTS").glob("*.json")))
    if raw_count != int(freeze.get("raw_count", -1)):
        raise RuntimeError(f"{label}_RAW_COUNT_DRIFT:{raw_count}")
    if label in {"P1", "P2"} and raw_count != 24:
        raise RuntimeError(f"{label}_EXPECTED_24_OUTPUTS:{raw_count}")
    # The controller's freeze receipt records the stable stage label (P0,
    # P1, or P2); POLICY_CONFIG.json carries the corresponding policy name.
    if freeze.get("policy") != label:
        raise RuntimeError(f"{label}_FREEZE_LABEL_DRIFT:{freeze.get('policy')}")
    policy_config = _read(source / "POLICY_CONFIG.json")
    if policy_config.get("logical_search_order_policy") != POLICIES[label]:
        raise RuntimeError(f"{label}_POLICY_DRIFT:{policy_config.get('logical_search_order_policy')}")

    result = {
        "label": label,
        "source": str(source),
        "policy": freeze.get("policy"),
        "raw_count": raw_count,
        "generation_ledger_sha256": _sha256(generation_path),
        "generation_hashes_checked": len(expected),
        "generation_hash_status": "PASS",
        "stored_generation_hash_verification": stored_verification,
        "raw_output_bytes": _bytes_in(source / "RAW_OUTPUTS"),
    }
    if label == "P0":
        semantic = _read(source / "P0_SEMANTIC_PARITY.json")
        if semantic.get("P0_ENGINE_PARITY") != "PASS_SEMANTIC_24_OF_24":
            raise RuntimeError("P0_SEMANTIC_PARITY_FAIL")
        result["p0_semantic_parity"] = semantic["P0_ENGINE_PARITY"]
    return result


def _copy_file(source: Path, destination: Path, copied: list[dict[str, Any]], source_root: Path, archive_root: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    source_hash = _sha256(source)
    archive_hash = _sha256(destination)
    if source_hash != archive_hash:
        raise RuntimeError(f"COPY_HASH_MISMATCH:{source}")
    copied.append({
        "source_relative_path": str(source.relative_to(source_root)).replace("\\", "/"),
        "archive_relative_path": str(destination.relative_to(archive_root)).replace("\\", "/"),
        "byte_size": source.stat().st_size,
        "sha256": source_hash,
    })


def _copy_compact(label: str, source: Path, archive: Path) -> list[dict[str, Any]]:
    destination = archive / label
    copied: list[dict[str, Any]] = []
    for path in sorted(source.iterdir()):
        if path.is_file():
            _copy_file(path, destination / path.name, copied, source, archive)
    for directory_name in COMPACT_DIRECTORIES:
        directory = source / directory_name
        if not directory.is_dir():
            raise RuntimeError(f"{label}_MISSING_COMPACT_DIRECTORY:{directory_name}")
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                _copy_file(path, destination / path.relative_to(source), copied, source, archive)
    return copied


def _archive_ledger(archive: Path) -> dict[str, str]:
    return {
        str(path.relative_to(archive)).replace("\\", "/"): _sha256(path)
        for path in sorted(archive.rglob("*"))
        if path.is_file() and path.name not in CONTROL_RECEIPTS
    }


def freeze(p0: Path, p1: Path, p2: Path, archive: Path) -> dict[str, Any]:
    if archive.exists():
        raise RuntimeError(f"ARCHIVE_ALREADY_EXISTS:{archive}")
    sources = {"P0": p0, "P1": p1, "P2": p2}
    verification = {label: _verify_source(label, path) for label, path in sources.items()}

    archive.mkdir(parents=True)
    copied = {label: _copy_compact(label, path, archive) for label, path in sources.items()}
    _write(archive / "SOURCE_GENERATION_VERIFICATION.json", {
        "experiment": EXPERIMENT,
        "status": "PASS",
        "target_blind": True,
        "gold_loaded": False,
        "sources": verification,
    })
    _write(archive / "COPY_PARITY.json", {
        "status": "PASS",
        "copied_file_count": sum(len(rows) for rows in copied.values()),
        "files": copied,
    })
    _write(archive / "ARCHIVE_SCOPE.json", {
        "experiment": EXPERIMENT,
        "status": "FROZEN",
        "target_blind": True,
        "gold_loaded": False,
        "policies": POLICIES,
        "included": ["all top-level compact receipts", *COMPACT_DIRECTORIES],
        "excluded": {
            "RAW_OUTPUTS": "Retained on the RunPod; every raw file is covered by the verified source GENERATION_HASHES.json ledger.",
            "ADAPTER_STAGE": "Empty staging directory; no result payload.",
        },
        "source_runs": verification,
    })
    hashes = _archive_ledger(archive)
    _write(archive / "ARCHIVE_HASHES.json", {"status": "PASS", "files": hashes, "excluded_control_receipts": sorted(CONTROL_RECEIPTS)})
    mismatches = [relative for relative, expected in hashes.items() if _sha256(archive / relative) != expected]
    verification_receipt = {"status": "PASS" if not mismatches else "FAIL", "checked": len(hashes), "mismatches": mismatches,
                            "excluded_control_receipts": sorted(CONTROL_RECEIPTS)}
    _write(archive / "ARCHIVE_HASH_VERIFICATION.json", verification_receipt)
    if mismatches:
        raise RuntimeError("ARCHIVE_HASH_VERIFICATION_FAIL")
    _write(archive / "COMPACT_ARCHIVE_FREEZE.json", {
        "experiment": EXPERIMENT,
        "status": "FROZEN",
        "target_blind": True,
        "gold_loaded": False,
        "source_generation_hashes": {label: value["generation_ledger_sha256"] for label, value in verification.items()},
        "archive_hash_ledger_sha256": _sha256(archive / "ARCHIVE_HASHES.json"),
        "archive_hash_status": "PASS",
    })
    return {"archive": str(archive), "source_verification": verification, "archive_verification": verification_receipt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0", type=Path, required=True)
    parser.add_argument("--p1", type=Path, required=True)
    parser.add_argument("--p2", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    print(_canonical(freeze(args.p0, args.p1, args.p2, args.archive)), end="")


if __name__ == "__main__":
    main()
