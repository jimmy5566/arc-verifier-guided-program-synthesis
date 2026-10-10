#!/usr/bin/env python3
"""Freeze the CPU receipt and Director prelaunch brief for matched-transfer V2."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/capability_repair_baseline_v1/matched_atomic_to_compositional_transfer_v2_rc1_rc5'
BRIEF = ROOT / 'orchestration/director/briefs/GOVERNOR_REVIEW_MATCHED_ATOMIC_TO_COMPOSITIONAL_TRANSFER_V2_RC1_RC5_PRELAUNCH.json'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(json.dumps(value, sort_keys=True, indent=2).encode('utf-8') + b'\n')
    os.replace(temporary, path)


def main():
    receipt = {
        'schema_version': 1,
        'protocol_id': 'MATCHED_ATOMIC_TO_COMPOSITIONAL_TRANSFER_V2_RC1_RC5',
        'status': 'CPU_FREEZE_COMPLETE_NOT_AUTHORIZED_FOR_MODEL_EXECUTION',
        'artifacts': {'cohort_manifest_sha256': sha(OUT / 'COHORT_MANIFEST.json'), 'protocol_sha256': sha(OUT / 'PROTOCOL.json'), 'generator_sha256': sha(ROOT / 'scripts/prepare_matched_atomic_to_compositional_transfer_v2.py'), 'test_sha256': sha(ROOT / 'tests/test_matched_atomic_to_compositional_transfer_v2.py')},
        'tests': {'command': 'py -3 -m unittest tests.test_matched_atomic_to_compositional_transfer_v1 tests.test_matched_atomic_to_compositional_transfer_v2 -q', 'status': 'PASS', 'count': 4},
        'scientific_repairs': {'RC1_deduplicated_canonical_latents': 48, 'RC2_functionally_sufficient_atoms': 'PASS_SELECT_THEN_RECOLOR_EQUALS_COMPOSITION_FOR_ALL_48', 'RC3_exposure_scope': 'WITHIN_SCHEMA_ONLY_TRAIN_STRUCTURAL_EXPOSURE_UNVERIFIED', 'RC4_primary_estimand': 'V7_OWN_CONDITIONED_RATE; COMPARATOR_COMMON_SUPPORT_OR_NOT_ESTIMABLE', 'RC5_top2_and_threshold': 'TOP2_REMOVED; CONDITIONAL_INTERVAL_DESCRIPTIVE_NO_UNJUSTIFIED_015_BOUNDARY'},
        'authorization': {'model_loading': False, 'gpu_inference': False, 'generation': False, 'optimizer': False, 'backward': False, 'training': False},
        'sealed_targets': {'sidecar_sha256': 'eb144c90d8f2100af662da0b570be8e80f24da8481cf2cb2d71054a2a5ea3296', 'opened': False, 'committed': False},
    }
    write(OUT / 'CPU_FREEZE_RECEIPT.json', receipt)
    brief = {
        'schema_version': 1,
        'brief_id': 'GOVERNOR_REVIEW_MATCHED_ATOMIC_TO_COMPOSITIONAL_TRANSFER_V2_RC1_RC5_PRELAUNCH',
        'scientific_question': 'On a deduplicated within-schema synthetic cohort, does V7 fail composition after two functionally sufficient atomic outputs are exact?',
        'context': {'rejected_predecessor': 'MATCHED_ATOMIC_TO_COMPOSITIONAL_TRANSFER_V1 is preserved NOT_AUTHORIZED: 48 rows collapsed to 19 latent combinations and used a non-equivalent recolor atom.', 'director_response_sha256': '0de10f6d4837b59c1ec4842238644992f9acc8a32402bfaa9c445bbe7cd4801d', 'successor_protocol_sha256': sha(OUT / 'PROTOCOL.json'), 'successor_cohort_sha256': sha(OUT / 'COHORT_MANIFEST.json'), 'cpu_freeze_receipt_sha256': sha(OUT / 'CPU_FREEZE_RECEIPT.json')},
        'repairs': receipt['scientific_repairs'],
        'frozen_design': {
            'cohort': '48 canonical-deduplicated tuples x ATOMIC_SELECT_WITH_CUE, ATOMIC_PARAMETERIZED_RECOLOR, COMPOSITION_SELECT_RECOLOR; 24 separate retention episodes; targets sealed locally.',
            'variation': 'colours, L4 or SQUARE4 shape, above or left selector relation, three layout families, and two distractor configurations.',
            'atomic_contract': 'select output retains selected source object plus one target-colour cue; parameterized recolor consumes that exact output; CPU proof reconstructs all composition targets.',
            'exposure': 'exact public prompt/observation/id checks passed against two frozen manifests. Synthetic TRAIN structural exposure is UNVERIFIED because those manifests lack canonical TRAIN task bytes; claim limited to within-schema composition.',
            'checkpoints': 'V7 primary; Family-Balanced descriptive comparator only.',
            'metrics': 'V7 own conditioned composition-failure rate primary. Family-Balanced descriptive. Cross-checkpoint effect only common support where both atoms exact for both, n>=12; else NOT_ESTIMABLE.',
            'uncertainty': '10,000 seed 20261010 bootstrap over inseparable canonical tuple units, strata selector_relation x component_shape. No ungrounded 0.15 mechanism boundary; descriptive interval only.',
            'inference': 'Greedy only; Batch16 primary; frozen 16->8->4->1 fallback; fixed 42-episode Batch1 validation; material drift invalidates; cap 1800 seconds.',
            'forbidden': 'No model/GPU before approval; no optimizer, update, backward, training, Beam, DFS, TTT, augmentation, external selector, Gold, dGold, FINAL_AUDIT.',
        },
        'director_question': 'Do RC1--RC5 repair construct validity sufficiently to authorize exactly one bounded no-update Greedy inference measurement under this frozen protocol? If not, name the smallest remaining scientific repair.',
        'allowed_director_outcomes': ['CONTINUE_CONTROLLER', 'REQUIRE_CHANGES', 'PAUSED'],
        'requested_authorization': 'If CONTINUE_CONTROLLER, state explicit next_stage, next_action, and whether model loading, GPU inference, and generation are authorized. Training and updates remain false.',
    }
    write(BRIEF, brief)
    print(json.dumps({'receipt_sha256': sha(OUT / 'CPU_FREEZE_RECEIPT.json'), 'brief_sha256': sha(BRIEF)}, sort_keys=True))


if __name__ == '__main__': main()
