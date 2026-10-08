#!/usr/bin/env python3
"""Freeze the reviewed executable and RunPod launch contract without launching it."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "foundation_v2_reconstruction_and_targeted_repair_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blob(path: str) -> str:
    return subprocess.check_output(["git", "rev-parse", f"HEAD:{path}"], cwd=ROOT, text=True).strip()


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--remote-root", required=True); p.add_argument("--output", type=Path, default=EXP / "RECONSTRUCTION_LAUNCH_BINDING_V1.json")
    a = p.parse_args(); remote = a.remote_root.rstrip("/")
    recipe = json.loads((EXP / "RECONSTRUCTED_V2_RECIPE_FREEZE.json").read_text(encoding="utf-8"))
    source_hashes = json.loads((EXP / "RECONSTRUCTION_SOURCE_ARTIFACT_HASHES.json").read_text(encoding="utf-8"))["artifacts"]
    base = recipe["base"]["base_files"]
    required = [{"path": f"/workspace/arc2/models/qwen3_4b_grids15_sft139/{name}", "sha256": value["sha256"]} for name, value in sorted(base.items())]
    for path, digest in sorted(source_hashes.items()): required.append({"path": f"{remote}/{path}", "sha256": digest})
    sources = ["scripts/run_capability_pilot_2m_v1.py", "src/capability_pilot_2m_v1/pilot.py", "src/capability_pilot_2m_v1/__init__.py", "scripts/preflight_foundation_v2_reconstruction_v1.py", "scripts/run_foundation_v2_reconstruction_v1.py", "scripts/record_targeted_capability_repair_gpu_time.py"]
    source_entries=[]
    for path in sources:
        local=ROOT/path; source_entries.append({"path":path,"git_blob_sha":blob(path),"file_sha256":sha(local),"containing_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()})
        required.append({"path":f"{remote}/{path}","sha256":sha(local)})
    train_manifest=ROOT/"artifacts/novel_training_data_v1_1/NOVEL_TRAIN_SHARD_MANIFEST.json"; val_manifest=ROOT/"artifacts/novel_training_data_v1_1/NOVEL_VAL_SHARD_MANIFEST.json"
    contracts=[]
    for root, manifest in ((f"{remote}/data/processed/novel_training_data_v1_1/train/novel",train_manifest),(f"{remote}/data/processed/novel_training_data_v1_1/validation/novel",val_manifest)):
        contracts.append({"root":root,"manifest":f"{remote}/{manifest.relative_to(ROOT).as_posix()}","manifest_key":"shards","manifest_content_sha256":hashlib.sha256(json.dumps(json.loads(manifest.read_text(encoding='utf-8')),sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()})
    replay=f"{remote}/data/processed/arc_training_v2_1/train/replay/replay-00000.parquet"
    required.append({"path":replay,"sha256":recipe["data"]["replay_shard_sha256"]})
    command=["python3",f"{remote}/scripts/run_capability_pilot_2m_v1.py","--mode","train","--model-path","/workspace/arc2/models/qwen3_4b_grids15_sft139","--novel-train-root",contracts[0]["root"],"--novel-validation-root",contracts[1]["root"],"--replay-shard",replay,"--freeze",f"/workspace/arc2/active_runs/reconstructed_foundation_v2_v1/freeze","--runtime",f"/workspace/arc2/active_runs/reconstructed_foundation_v2_v1/runtime","--checkpoints",f"/workspace/arc2/active_runs/reconstructed_foundation_v2_v1/checkpoints"]
    value={"schema_version":1,"status":"FROZEN_PENDING_PREFLIGHT","protocol_id":"FOUNDATION_V2_RECONSTRUCTION_AND_TARGETED_REPAIR_V1","source_provenance":{"historical_d7b489e7":"artifact provenance only; executable did not exist there","reviewed_executable_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"checked_out_root":remote,"entries":source_entries},"required_files":required,"dataset_contracts":contracts,"command":command,"output":{"parent":"/workspace/arc2/active_runs","run_root":"/workspace/arc2/active_runs/reconstructed_foundation_v2_v1","freeze":"/workspace/arc2/active_runs/reconstructed_foundation_v2_v1/freeze","runtime":"/workspace/arc2/active_runs/reconstructed_foundation_v2_v1/runtime","checkpoints":"/workspace/arc2/active_runs/reconstructed_foundation_v2_v1/checkpoints"},"reservation_seconds":7200,"cumulative_cap_seconds":28800,"no_gpu_training_started":True}
    a.output.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({"status":"FROZEN","output":str(a.output)}))
    return 0

if __name__=="__main__": raise SystemExit(main())
