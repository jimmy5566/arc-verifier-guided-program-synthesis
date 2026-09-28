#!/usr/bin/env python3
"""Promote only immutable V5 runtime assets into established ARC2 Global Storage."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/"src")]


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def promote_file(source: Path, destination: Path) -> str:
    """Use hardlink, then reflink, then a normal copy; never move source."""
    destination.parent.mkdir(parents=True,exist_ok=True)
    expected=sha256(source)
    if destination.exists():
        if sha256(destination)!=expected: raise RuntimeError(f"existing global file hash mismatch:{destination}")
        return "existing_global"
    try:
        os.link(source,destination); method="hardlink"
    except OSError:
        try:
            subprocess.run(["cp","--reflink=always","--preserve=mode,timestamps",str(source),str(destination)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); method="reflink"
        except Exception:
            # Global Storage can reject chmod/utime metadata operations even
            # after accepting byte writes.  Artifact identity is the SHA256,
            # not POSIX timestamp/mode metadata, so use a byte-only copy.
            shutil.copyfile(source,destination); method="copy"
    if sha256(destination)!=expected: raise RuntimeError(f"promoted global file hash mismatch:{destination}")
    return method


def already_in_global(*, source: Path, global_root: Path) -> bool:
    """Whether an immutable authoritative asset is already durable in Global.

    Some RunPod layouts mount both the authoritative run and Global root under
    the same persistent volume.  Duplicating 180 one-GiB adapters in that case
    is neither a promotion nor a safety improvement; the canonical immutable
    source path itself is the Global path and is recorded as such.
    """
    try:
        source.resolve().relative_to(global_root.resolve())
        return True
    except ValueError:
        return False


def bundle_reference_assets(*, native_config_dir: Path, destination: Path) -> dict[str, Any]:
    """Create one verified archive for the non-Git native-tokenizer snapshot.

    The public tokenizer/config snapshot is small but contains enough individual
    files that copying it piecemeal to Global would violate the storage policy.
    A deterministic tar.zst bundle is the single optional immutable reference
    asset a clean Pod may need in addition to Git and the materialized model.
    """
    if not native_config_dir.is_dir():
        raise RuntimeError(f"native config directory missing:{native_config_dir}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".partial")
        if temporary.exists():
            temporary.unlink()
        # GNU tar is available on the validated Linux Pods.  Sort and fixed
        # metadata make repeat creation reproducible without copying a source tree.
        subprocess.run([
            "tar", "--zstd", "--sort=name", "--mtime=@0", "--owner=0", "--group=0",
            "--numeric-owner", "-C", str(native_config_dir.parent), "-cf", str(temporary),
            native_config_dir.name,
        ], check=True)
        temporary.replace(destination)
    return {
        "logical_name": "reference_bundle/native_tokenizer_config",
        "absolute_path": str(destination),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "source_path": str(native_config_dir),
        "storage_method": "tar_zstd_bundle",
        "immutable": True,
    }


def previously_attested_adapters(*, v5root: Path, checkpoint_csv: Path) -> list[dict[str, str]] | None:
    """Return a compact prior attestation without reading adapter tensor bytes.

    The first globalization performs a full byte-level SHA256 comparison against
    the authoritative checkpoint manifest.  Repeating that 177-GiB scan on a
    later Pod/bootstrap is neither an additional scientific check nor a useful
    default.  A previous immutable global manifest plus the authoritative
    *small* checkpoint manifest is the durable attestation.  We still fail
    closed on missing paths, count/identity/size disagreement, and an explicit
    caller may force a fresh full audit with ``--reverify-adapters``.
    """
    manifest_path = v5root / "GLOBAL_ASSET_MANIFEST.json"
    adapter_manifest = v5root / "eval60_adapter_manifest.csv"
    if not manifest_path.is_file() or not adapter_manifest.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text())
        if len(payload.get("assets", [])) < 180:
            return None
        with checkpoint_csv.open(newline="", encoding="utf-8") as handle:
            expected = {
                (str(row["task_id"]), int(row["depth"])): str(row.get("checkpoint_sha256") or row.get("sha256") or "")
                for row in csv.DictReader(handle)
            }
        with adapter_manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, ValueError, KeyError):
        return None
    if len(expected) != 180 or len(rows) != 180:
        return None
    seen: set[tuple[str, int]] = set()
    for row in rows:
        try:
            key = (str(row["task_id"]), int(row["depth"]))
            path = Path(row["global_path"])
            if key in seen or expected.get(key) != row["sha256"] or not path.is_file():
                return None
            if path.stat().st_size != int(row["size"]):
                return None
            seen.add(key)
        except (KeyError, OSError, ValueError):
            return None
    return rows if seen == set(expected) else None


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--global-root",type=Path,default=Path("/workspace/arc2")); parser.add_argument("--model-source",type=Path,required=True); parser.add_argument("--authoritative-root",type=Path,required=True); parser.add_argument("--repo-root",type=Path,required=True); parser.add_argument("--source-commit",required=True); parser.add_argument("--native-config-dir",type=Path,required=True); parser.add_argument("--final-config",type=Path,required=True); parser.add_argument("--reverify-adapters",action="store_true",help="force a new 180-file byte-level SHA256 audit")
    args=parser.parse_args(); global_root=args.global_root.resolve(); model_source=args.model_source.resolve(); authoritative=args.authoritative_root.resolve()
    manifest_path=model_source/"model_manifest.json"
    if not manifest_path.is_file(): raise RuntimeError("materialized model manifest missing")
    model_manifest=json.loads(manifest_path.read_text()); model_target=global_root/"models"/"qwen3_4b_grids15_sft139"
    model_assets=[]
    for source in sorted(path for path in model_source.rglob("*") if path.is_file()):
        relative=source.relative_to(model_source); target=model_target/relative; method=promote_file(source,target)
        model_assets.append({"logical_name":f"model/{relative}","absolute_path":str(target),"size_bytes":target.stat().st_size,"sha256":sha256(target),"source_path":str(source),"storage_method":method,"immutable":True})
    checkpoint_csv=authoritative/"checkpoint_manifest.csv"
    if not checkpoint_csv.is_file(): raise RuntimeError("authoritative checkpoint manifest missing")
    adapter_rows=[]; adapter_assets=[]
    with checkpoint_csv.open(newline="",encoding="utf-8") as handle:
        source_rows=list(csv.DictReader(handle))
    if len(source_rows)!=180: raise RuntimeError(f"authoritative checkpoint manifest expected 180 rows, found {len(source_rows)}")
    v5root=global_root/"turbodfs_v5"; v5root.mkdir(parents=True,exist_ok=True)
    prior_rows = None if args.reverify_adapters else previously_attested_adapters(v5root=v5root, checkpoint_csv=checkpoint_csv)
    for row in (prior_rows if prior_rows is not None else source_rows):
        task_id,depth=str(row["task_id"]),int(row["depth"]); source=Path(row.get("checkpoint_path") or row.get("adapter_path") or row.get("original_path") or "")
        expected=row.get("checkpoint_sha256") or row.get("sha256")
        if prior_rows is not None:
            target, method = Path(row["global_path"]), "prior_global_attestation"
            if not target.is_file() or target.stat().st_size != int(row["size"]):
                raise RuntimeError(f"prior global adapter inaccessible:{target}")
            target_sha = expected
        else:
            source_sha = sha256(source) if source.is_file() else None
            if not source.is_file() or not expected or source_sha!=expected: raise RuntimeError(f"authoritative adapter identity failure:{task_id} d{depth}")
            if already_in_global(source=source, global_root=global_root):
                target, method = source, "authoritative_already_global"
                target_sha = source_sha
            else:
                target=global_root/"adapters"/"eval60_authoritative_greedy_v1"/task_id/f"depth_{depth:03d}"/source.name
                method=promote_file(source,target)
                target_sha = sha256(target)
        if target_sha!=expected: raise RuntimeError(f"global adapter hash mismatch:{target}")
        adapter_rows.append({"task_id":task_id,"depth":depth,"global_path":str(target),"original_path":str(source),"size":target.stat().st_size,"sha256":expected,"storage_method":method})
        adapter_assets.append({"logical_name":f"adapter/{task_id}/depth_{depth:03d}","absolute_path":str(target),"size_bytes":target.stat().st_size,"sha256":expected,"source_path":str(source),"storage_method":method,"immutable":True})
    reference_asset = bundle_reference_assets(
        native_config_dir=args.native_config_dir.resolve(),
        destination=global_root / "assets" / "reference_bundle.tar.zst",
    )
    adapter_manifest=v5root/"eval60_adapter_manifest.csv"
    with adapter_manifest.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["task_id","depth","global_path","original_path","size","sha256","storage_method"]); writer.writeheader(); writer.writerows(adapter_rows)
    payload={"global_root":str(global_root),"repository":"https://github.com/jimmy5566/arc-verifier-guided-program-synthesis.git","source_commit":args.source_commit,"assets":model_assets+adapter_assets+[reference_asset],"adapter_manifest":str(adapter_manifest),"model_manifest":str(model_target/"model_manifest.json")}
    (v5root/"GLOBAL_ASSET_MANIFEST.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    env={"ARC2_GLOBAL_ROOT":str(global_root),"ARC2_MODEL_PATH":str(model_target),"ARC2_ADAPTER_ROOT":str(global_root/"adapters"/"eval60_authoritative_greedy_v1"),"ARC2_ADAPTER_MANIFEST":str(adapter_manifest),"ARC2_GLOBAL_ASSET_MANIFEST":str(v5root/"GLOBAL_ASSET_MANIFEST.json"),"ARC2_REFERENCE_BUNDLE":reference_asset["absolute_path"],"ARC2_REPO_ROOT":str(args.repo_root.resolve()),"ARC2_NATIVE_CONFIG_DIR":str(args.native_config_dir.resolve()),"ARC2_FINAL_TURBODFS_CONFIG":str(args.final_config.resolve())}
    (v5root/"GLOBAL_RUNTIME_PATHS.env").write_text("\n".join(f"{k}={v}" for k,v in env.items())+"\n")
    # Each adapter was byte-verified in this invocation or validated against a
    # prior immutable attestation; never trigger a redundant second 177-GiB scan.
    verified=len(adapter_rows)
    result={"GLOBALIZATION":"PASS","global_root":str(global_root),"model_path":str(model_target),"model_files":len(model_assets),"adapters_verified":verified,"adapter_verification_mode":"prior_attestation" if prior_rows is not None else "full_byte_audit","adapter_manifest":str(adapter_manifest),"global_asset_manifest":str(v5root/"GLOBAL_ASSET_MANIFEST.json"),"global_runtime_paths":str(v5root/"GLOBAL_RUNTIME_PATHS.env")}
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__": main()
