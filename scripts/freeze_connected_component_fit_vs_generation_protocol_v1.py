"""Freeze the CPU-only no-update prelaunch protocol for the V7 RUN_002 follow-up.

This tool deliberately imports no model/runtime libraries and never opens OOD target
content.  It reads only frozen binding metadata, prompt metadata, raw predictions,
and recovered receipts to create review artifacts.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from collections import Counter
from pathlib import Path

PROTOCOL_ID = "CONNECTED_COMPONENT_COUNT_FIT_VS_GENERATION_DIAGNOSTIC_V1"
RUN_ID = "RUN_002_09ff2d41a20e291a5900c6f73897fed5"
ROOT = Path(__file__).resolve().parents[1]

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def write_json(path: Path, value: object) -> str:
    payload=(json.dumps(value, indent=2, sort_keys=True)+"\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_name(path.name+f".{os.getpid()}.tmp")
    temp.write_bytes(payload)
    os.replace(temp,path)
    return hashlib.sha256(payload).hexdigest()

def need(path: Path, expected: str | None = None) -> Path:
    if not path.is_file(): raise RuntimeError(f"MISSING:{path}")
    actual=sha(path)
    if expected and actual != expected: raise RuntimeError(f"SHA_MISMATCH:{path}:{actual}")
    return path

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--recovered-root",type=Path,required=True)
    ap.add_argument("--closure-out",type=Path,required=True)
    ap.add_argument("--protocol-out",type=Path,required=True)
    ap.add_argument("--validation-out",type=Path,required=True)
    args=ap.parse_args()
    pilot=ROOT/"experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1"
    binding_path=need(pilot/"execution_binding_v7_run_002/EXECUTION_BINDING_V7.json",
                      "457e50a9e21ea6ffc7dd535ab0f933c15989fcad9e6ea2602e1772c3a57ba354")
    binding=json.loads(binding_path.read_text(encoding="utf-8"))
    recovered=args.recovered_root
    manifest_path=need(recovered/"RECOVERY_MANIFEST.json","a0c9cf99efb00a3a44f010b245c62ff10d498f8433549cea4401bda252e72f2a")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt_paths={
        "raw_freeze": need(recovered/"RAW_FREEZE_RECEIPT.json","fbd741dcfcd13e6b92fa6a70e98a8546b5244562319ef9e8283a0c5a3f9c1450"),
        "terminal": need(recovered/"TERMINAL_RECEIPT.json","5cf2c4299e7b377c6456895560baf7c9bb8317de59a380ea0ed8864a8525bb2a"),
        "score": need(recovered/"SCORE_RECEIPT.json","35751ac8d0264ef32f837fd6ad2c47382518cc17beb3164a8c9cb51786288d6d"),
        "control_adapter_receipt": need(recovered/"CONTROL_NARROW_STRUCTURE/ADAPTER_RECEIPT.json","91c5a5484938a81d8b6d6ae5bbe20fbeb5ec5f22c0e8c3a1eb75bcf7efcbb5df"),
        "treatment_adapter_receipt": need(recovered/"TREATMENT_STRUCTURE_DIVERSE/ADAPTER_RECEIPT.json","46b7aeff895fb391cd79ef412acaeca4132246c46d9b88e5dc0b95dddbddc654"),
    }
    raw_freeze=json.loads(receipt_paths["raw_freeze"].read_text(encoding="utf-8"))
    score=json.loads(receipt_paths["score"].read_text(encoding="utf-8"))
    control_adapter=json.loads(receipt_paths["control_adapter_receipt"].read_text(encoding="utf-8"))
    treatment_adapter=json.loads(receipt_paths["treatment_adapter_receipt"].read_text(encoding="utf-8"))
    train_path=need(pilot/"measurement_contract_v1/TRAIN_FIT_INPUT_PROMPTS.json",
                    "708596c8715c5380903ab1af89852c32f05bb9176fb45cfd6c95f33e63dff981")
    train=json.loads(train_path.read_text(encoding="utf-8"))
    selected=[row for row in train["records"] if row.get("family")=="CONNECTED_COMPONENT_COUNT"]
    if len(selected)!=24: raise RuntimeError("TRAIN_COHORT_COUNT_NOT_24")
    if Counter(row.get("arm") for row in selected) != {"CONTROL_NARROW_STRUCTURE":12,"TREATMENT_STRUCTURE_DIVERSE":12}:
        raise RuntimeError("TRAIN_COHORT_ARM_BALANCE_INVALID")
    target_keys={"output","target","target_token_ids","label","expected","solution"}
    if any(set(row) & target_keys for row in selected): raise RuntimeError("TRAIN_INPUT_RECORD_SCHEMA_CHANGED")
    # This intentionally serializes only metadata, never task or target content.
    train_metadata=[{key:row[key] for key in ("arm","canonical_base_id","family","source_id","source_slot")} for row in selected]
    train_metadata_sha=hashlib.sha256(json.dumps(train_metadata,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    ood_path=need(pilot/"OOD_INPUT_PROMPTS.json","b50687c9d9493a3f2dc1ce6f8e369aa48915277b0818a71b22bec16a34f57a3b")
    ood_rows=json.loads(ood_path.read_text(encoding="utf-8"))
    if not isinstance(ood_rows,list) or len(ood_rows)!=48: raise RuntimeError("OOD_COHORT_COUNT_NOT_48")
    if any(set(row) & target_keys for row in ood_rows): raise RuntimeError("OOD_INPUT_TARGET_LEAK")
    if manifest["binding_sha256"] != sha(binding_path): raise RuntimeError("RECOVERY_BINDING_MISMATCH")
    if manifest["score_sha256"] != sha(receipt_paths["score"]): raise RuntimeError("RECOVERY_SCORE_MISMATCH")
    if raw_freeze.get("status")!="COMPLETE_RAW_FROZEN_NO_TARGETS" or raw_freeze.get("targets_accessed") is not False:
        raise RuntimeError("RAW_FREEZE_TARGET_BOUNDARY_INVALID")
    promotion=score.get("promotion",{})
    if promotion.get("decision")!="DO_NOT_PROMOTE": raise RuntimeError("CLOSURE_PROMOTION_DECISION_INVALID")
    closure={
      "schema_version":1,
      "protocol_id":binding["protocol_id"],
      "run_id":RUN_ID,
      "status":"CLOSED_VALID_NEGATIVE_RESULT_PRESERVED",
      "scientific_result":{
        "promotion":"DO_NOT_PROMOTE",
        "checkpoint_promotion":"REJECTED_BOTH_TRAINED_ARMS",
        "retry":"REJECTED",
        "same_budget_narrow_vs_diverse_followup":"REJECTED",
        "v7_reference_preserved":True,
        "interpretation":"Treatment minus Control structural-OOD exact delta was -8; the Connected Component Count family delta was -8. This is a valid negative result, not evidence of a general structural-diversity impossibility."
      },
      "immutable_identity":{
        "execution_binding_sha256":sha(binding_path),
        "execution_source_commit":binding["execution_source_commit"],
        "remote_output_root":binding["output"]["remote_output_root"],
        "raw_freeze_sha256":sha(receipt_paths["raw_freeze"]),
        "terminal_receipt_sha256":sha(receipt_paths["terminal"]),
        "deterministic_score_sha256":sha(receipt_paths["score"]),
        "recovery_manifest_sha256":sha(manifest_path),
        "raw_prediction_sha256":raw_freeze["raw_sha256"],
        "v7_adapter_model_sha256":binding["v7"]["adapter_model_sha256"],
        "v7_adapter_config_sha256":binding["v7"]["adapter_config_sha256"],
        "control_adapter_model_sha256":control_adapter["adapter_sha256"],
        "control_adapter_receipt_sha256":sha(receipt_paths["control_adapter_receipt"]),
        "treatment_adapter_model_sha256":treatment_adapter["adapter_sha256"],
        "treatment_adapter_receipt_sha256":sha(receipt_paths["treatment_adapter_receipt"])
      },
      "preservation":{
        "recovered_result_root":str(recovered.resolve()),
        "bounded_recovery":manifest["bounded_recovery"],
        "remote_adapters":"RETAINED_AT_EXISTING_RUN_OUTPUT; NOT_RELOADED_OR_COPIED_BY_THIS_CPU_STAGE",
        "files_preserved":sorted(manifest["files"]),
        "cpu_only":True,
        "model_loaded":False,
        "gpu_used":False,
        "targets_opened_by_this_closure":False
      }
    }
    closure_sha=write_json(args.closure_out,closure)
    protocol={
      "schema_version":1,
      "protocol_id":PROTOCOL_ID,
      "status":"CPU_ONLY_FROZEN_PROSPECTIVE_PROTOCOL_PENDING_DIRECTOR_REVIEW",
      "scope":"One no-optimizer-update diagnostic. It measures fixed V7, Control, and Treatment only; it does not retry RUN_002 or introduce a training variant.",
      "immutable_predecessor":{
        "closure_receipt_path":str(args.closure_out.resolve()),
        "closure_receipt_sha256":closure_sha,
        "run_002_binding_sha256":sha(binding_path),
        "run_002_execution_source_commit":binding["execution_source_commit"],
        "native_greedy_raw_freeze_sha256":sha(receipt_paths["raw_freeze"]),
        "deterministic_score_sha256":sha(receipt_paths["score"]),
        "recovery_manifest_sha256":sha(manifest_path)
      },
      "checkpoint_identities":{
        "V7":{
          "checkpoint_manifest_path":binding["v7"]["checkpoint_manifest_path"],
          "checkpoint_manifest_sha256":binding["v7"]["checkpoint_manifest_sha256"],
          "adapter_model_sha256":binding["v7"]["adapter_model_sha256"],
          "adapter_config_sha256":binding["v7"]["adapter_config_sha256"]
        },
        "CONTROL_NARROW_STRUCTURE":{
          "adapter_model_sha256":control_adapter["adapter_sha256"],
          "adapter_receipt_sha256":sha(receipt_paths["control_adapter_receipt"]),
          "optimizer_steps_historical":control_adapter["optimizer_steps"]
        },
        "TREATMENT_STRUCTURE_DIVERSE":{
          "adapter_model_sha256":treatment_adapter["adapter_sha256"],
          "adapter_receipt_sha256":sha(receipt_paths["treatment_adapter_receipt"]),
          "optimizer_steps_historical":treatment_adapter["optimizer_steps"]
        }
      },
      "frozen_cohorts":{
        "train_fit_connected_component_count":{
          "source_path":"experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/measurement_contract_v1/TRAIN_FIT_INPUT_PROMPTS.json",
          "source_sha256":sha(train_path),
          "selection":"family == CONNECTED_COMPONENT_COUNT",
          "records":24,
          "arm_counts":{"CONTROL_NARROW_STRUCTURE":12,"TREATMENT_STRUCTURE_DIVERSE":12},
          "metadata_identity_sha256":train_metadata_sha
        },
        "structural_ood":{
          "source_path":"experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/OOD_INPUT_PROMPTS.json",
          "source_sha256":sha(ood_path),
          "records":48
        },
        "unchanged_conditions":{
          "prompts":"exact input bytes from the named frozen cohorts",
          "decoder":"NATIVE_GREEDY",
          "max_new_tokens":192,
          "requested_batch_size":32,
          "fallback_ladder":[32,16,8,4,1],
          "fallback_cause":"CUDA_OOM_ONLY",
          "checkpoints":"the three hash-bound identities above",
          "no_new_cohorts":True
        }
      },
      "target_boundary":{
        "cpu_stage_target_access":False,
        "no_new_target_set":True,
        "ood_oracle_scorer_contract_path":"experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/measurement_contract_v1/ORACLE_AND_SCORER_CONTRACT.json",
        "ood_oracle_scorer_contract_sha256":binding["inputs"]["experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/measurement_contract_v1/ORACLE_AND_SCORER_CONTRACT.json"],
        "runtime_requirement":"Before any teacher-forced forward, a pre-existing exact correct-token sidecar for the same 24+48 cohort must be SHA-bound and match the frozen tokenizer serialization. No sidecar content, target grids, or target tokens are emitted in this protocol.",
        "current_status":"EXACT_PREEXISTING_CORRECT_TOKEN_SIDECAR_NOT_YET_BOUND; GPU LAUNCH MUST FAIL CLOSED UNTIL IT IS"
      },
      "measurements":{
        "teacher_forced":{
          "unit":"each correct target token at its fixed prefix under one causal forward",
          "per_family_reports":["correct_token_rank_distribution","Top1_rate_rank_eq_1","Top2_rate_rank_le_2","mean_correct_token_margin","token_weighted_NLL","sequence_normalized_NLL"],
          "margin":"logit(correct token) minus max logit(any incorrect token) at the same position",
          "token_weighted_NLL":"sum(-log p(correct token)) / all supervised correct tokens in the family",
          "sequence_normalized_NLL":"mean over records of [sum(-log p(correct token)) / correct-token-count]",
          "selection_prohibited":True
        },
        "free_running":{
          "decoder":"the same NATIVE_GREEDY decoder and max_new_tokens=192",
          "first_divergence":"first index at which generated token differs from the correct target token; if every shared token agrees and lengths differ, use shared-prefix length; otherwise null",
          "frozen_native_greedy_correlation":"join exactly by checkpoint, cohort, and record ID to the existing RUN_002 raw prediction hashes and score outcome where that existing raw row exists; retain new raw token hashes for V7 TRAIN rows because no V7 TRAIN row exists in RUN_002"
        },
        "semantic_error_taxonomy":{
          "independent_annotation_required":True,
          "categories":["CONNECTED_COMPONENT_COUNT","SELECTED_OBJECT_IDENTITY","GEOMETRY","COLOR","DIMENSIONS","PARSER_FAILURE","OUTPUT_CONSTRUCTION"],
          "method":"Use an independent deterministic semantic comparator after raw predictions are frozen. Record multi-label semantic axes plus exactly one exclusive primary code. PARSER_FAILURE is primary if unparsable; DIMENSIONS is primary for shape mismatch; otherwise choose the first present axis in CONNECTED_COMPONENT_COUNT, SELECTED_OBJECT_IDENTITY, GEOMETRY, COLOR, OUTPUT_CONSTRUCTION order. OUTPUT_CONSTRUCTION applies only when semantic selected-object/count/geometry/color axes agree but canonical output assembly differs.",
          "target_release":"forbidden"
        }
      },
      "decision_rule":{
        "TRAIN_RANKING_FAILURE":"If Treatment has poorer Connected Component Count TRAIN teacher-forced ranking, margin, or NLL than Control in the same direction as its existing exact-fit deficit, support insufficient effective fitting or optimization under the fixed diverse budget.",
        "FREE_RUNNING_SEQUENCE_FAILURE":"If Treatment has strong TRAIN teacher-forced ranking but poor native-Greedy exactness with early divergence or output-construction error, support a sequence-level or output-construction failure.",
        "STRUCTURAL_GENERALIZATION_FAILURE":"If Treatment has strong TRAIN ranking and exactness but weak structural-OOD ranking and exactness, support a representation or structural-generalization failure.",
        "invalid":"Any checkpoint, cohort, target-token, prompt, decoder, completeness, batch1 sensitivity, or paired-support failure is INVALID_NOT_INTERPRETABLE and selects no training mechanism."
      },
      "execution_shape_and_budget":{
        "checkpoint_record_teacher_forced_forwards":216,
        "checkpoint_record_free_running_generations":216,
        "formula":"3 checkpoints * (24 TRAIN_FIT Connected Component Count + 48 structural OOD)",
        "optimizer_steps":0,
        "backward_calls":0,
        "training":False,
        "reference":"RUN_002 raw generation froze 392 evaluations in 779.9362430069596 seconds, with two 96-step training arms also present.",
        "planning_estimate_seconds":3600,
        "hard_cap_seconds":10800,
        "no_update_gpu_estimate":"Bounded at 3,600 seconds planned and 10,800 seconds hard cap; this is conservative because the proposed diagnostic has no optimizer, backward, or parameter updates but adds teacher-forced forwards."
      },
      "fresh_execution_authorization_requirement":{
        "required":True,
        "why":"RUN_002 one-shot authorization is consumed and non-transferable; this is a distinct GPU diagnostic.",
        "must_bind_before_launch":["this protocol SHA256","a separately frozen execution binding and exact source commit","three checkpoint hashes above","24-record TRAIN metadata identity and 48-record OOD SHA256","exact pre-existing correct-token sidecar SHA256 and tokenizer identity","NATIVE_GREEDY decoder/max_new_tokens/batch policy","one detached job","retry=false","optimizer_steps=0","backward_calls=0","hard_cap_seconds=10800"],
        "must_refuse_when":["the correct-token sidecar is absent or mismatched","any checkpoint/cohort/prompt/decoder identity differs","an existing output root, terminal receipt, or active matching job exists","a request adds training, gradient, target release, or a new cohort"]
      },
      "forbidden":["model loading during this CPU stage","GPU use during this CPU stage","gradient computation","optimizer construction","training","generation during this CPU stage","new target construction or opening","Gold","dGold","FINAL_AUDIT","E05 targets","V4 targets","checkpoint promotion","RUN_002 retry"]
    }
    protocol_sha=write_json(args.protocol_out,protocol)
    validation={
      "schema_version":1,
      "protocol_id":PROTOCOL_ID,
      "status":"PASS_CPU_ONLY_FROZEN_PROTOCOL_VALIDATION",
      "protocol_sha256":protocol_sha,
      "checks":{
        "run_002_binding":"PASS",
        "recovered_raw_terminal_score_and_adapter_receipts":"PASS",
        "negative_closure_do_not_promote":"PASS",
        "train_connected_component_cohort_24_metadata_only":"PASS",
        "structural_ood_cohort_48_input_only":"PASS",
        "no_model_import_or_gpu_or_generation":"PASS",
        "no_target_content_opened_or_written":"PASS",
        "fresh_authorization_and_sidecar_fail_closed_rule":"PASS"
      },
      "observations":{
        "model_loaded":False,"gpu_used":False,"gradients_computed":False,
        "optimizer_constructed":False,"training_started":False,"generation_started":False,
        "new_targets_opened":False,"forbidden_targets_accessed":False,
        "preexisting_correct_token_sidecar_bound":False
      },
      "blocking_before_any_gpu":"A hash-bound pre-existing correct-token sidecar and a fresh binding-specific one-shot authorization are required."
    }
    validation_sha=write_json(args.validation_out,validation)
    print(json.dumps({"closure_sha256":closure_sha,"protocol_sha256":protocol_sha,"validation_sha256":validation_sha,"status":validation["status"]},sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())

