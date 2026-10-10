#!/usr/bin/env python3
"""CPU-only RC1--RC5 successor freeze for matched atomic-to-composition transfer.

This successor deliberately leaves the rejected V1 cohort untouched.  It has
one scientific construct: can a checkpoint execute SELECT followed by a
parameterised RECOLOR on independently sampled latent tuples?  No target is
written to the public cohort and this module never imports a model runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'MATCHED_ATOMIC_TO_COMPOSITIONAL_TRANSFER_V2_RC1_RC5'
SEED = 20261012
TUPLES = 48
RETENTION = 24
V7 = 'experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json'
FB = 'experiments/capability_repair_baseline_v1/family_balanced_loss_control_v1/round_001/evaluation_v1/FINAL_CHECKPOINT_MANIFEST.json'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    return sha_bytes(path.read_bytes())


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(json.dumps(value, sort_keys=True, indent=2).encode('utf-8') + b'\n')
    os.replace(temporary, path)


def blank():
    return [[0] * 9 for _ in range(9)]


def clone(grid):
    return [row[:] for row in grid]


SHAPES = {
    'L4': ((0, 0), (1, 0), (1, 1), (2, 0)),
    'SQUARE4': ((0, 0), (0, 1), (1, 0), (1, 1)),
}


def put_shape(grid, y, x, colour, shape):
    for dy, dx in SHAPES[shape]:
        grid[y + dy][x + dx] = colour


def component_at(grid, y, x, colour):
    if grid[y][x] != colour:
        raise ValueError('SELECTED_COMPONENT_ANCHOR_INVALID')
    pending, seen, out = [(y, x)], {(y, x)}, []
    while pending:
        a, b = pending.pop()
        out.append((a, b))
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            na, nb = a + da, b + db
            if 0 <= na < 9 and 0 <= nb < 9 and grid[na][nb] == colour and (na, nb) not in seen:
                seen.add((na, nb)); pending.append((na, nb))
    return out


def layout(params, variant):
    # All coordinates have room for both frozen component shapes.  Layouts
    # vary independently of colours and include a relation-specific marker.
    if params['selector_relation'] == 'MARKER_ABOVE_ANCHOR':
        choices = ((2, 1, 5, 5), (3, 4, 1, 1), (4, 2, 1, 5))
        sy, sx, dy, dx = choices[(variant + params['layout_family']) % 3]
        return sy, sx, sy - 1, sx, dy, dx
    choices = ((2, 3, 5, 5), (4, 4, 1, 1), (5, 3, 1, 5))
    sy, sx, dy, dx = choices[(variant + params['layout_family']) % 3]
    return sy, sx, sy, sx - 1, dy, dx


def raw_scene(params, variant):
    sy, sx, my, mx, dy, dx = layout(params, variant)
    grid = blank()
    put_shape(grid, sy, sx, params['source_colour'], params['component_shape'])
    distractor_shape = 'SQUARE4' if params['distractor_configuration'] == 'SHAPE_CONTRAST' else params['component_shape']
    put_shape(grid, dy, dx, params['source_colour'], distractor_shape)
    if params['distractor_configuration'] == 'EXTRA_MARKER_COLOUR':
        grid[8][8] = params['distractor_colour']
    grid[my][mx] = params['target_colour']
    return grid, (sy, sx), (my, mx)


def select_projection(raw, params, anchor, marker):
    selected = component_at(raw, anchor[0], anchor[1], params['source_colour'])
    out = blank()
    for y, x in selected:
        out[y][x] = params['source_colour']
    # Preserve the parameter cue, so this output is directly valid recolor input.
    out[marker[0]][marker[1]] = params['target_colour']
    return out


def recolor_projection(selected_with_cue, params):
    marker_count = sum(cell == params['target_colour'] for row in selected_with_cue for cell in row)
    if marker_count != 1:
        raise ValueError('RECOLOR_CUE_NOT_UNAMBIGUOUS')
    out = blank()
    for y, row in enumerate(selected_with_cue):
        for x, cell in enumerate(row):
            if cell == params['source_colour']:
                out[y][x] = params['target_colour']
    return out


def task_for(params, role):
    pairs = []
    for variant in (0, 1):
        raw, anchor, marker = raw_scene(params, variant)
        selected = select_projection(raw, params, anchor, marker)
        composition = recolor_projection(selected, params)
        if role == 'ATOMIC_SELECT_WITH_CUE':
            pairs.append({'input': raw, 'output': selected})
        elif role == 'ATOMIC_PARAMETERIZED_RECOLOR':
            pairs.append({'input': selected, 'output': composition})
        elif role == 'COMPOSITION_SELECT_RECOLOR':
            pairs.append({'input': raw, 'output': composition})
        else:
            raise ValueError('UNKNOWN_ROLE')
    raw, anchor, marker = raw_scene(params, 2)
    selected = select_projection(raw, params, anchor, marker)
    composition = recolor_projection(selected, params)
    if role == 'ATOMIC_SELECT_WITH_CUE':
        test, answer = raw, selected
    elif role == 'ATOMIC_PARAMETERIZED_RECOLOR':
        test, answer = selected, composition
    else:
        test, answer = raw, composition
    return {'train': pairs, 'test': [{'input': test}]}, answer


def native(grid):
    return '\n'.join(''.join(str(cell) for cell in row) for row in grid)


def prompt(task):
    parts = []
    for pair in task['train']:
        parts.extend((f'<|im_start|>user\n{native(pair["input"])}<|im_end|>', f'<|im_start|>assistant\n{native(pair["output"])}<|im_end|>'))
    parts.extend((f'<|im_start|>user\n{native(task["test"][0]["input"])}<|im_end|>', '<|im_start|>assistant\n'))
    return ''.join(parts)


def retention_task(source, target, variant):
    grid = blank(); y, x = ((2, 2), (3, 4), (4, 1))[variant % 3]
    put_shape(grid, y, x, source, 'L4')
    pairs = []
    for off in (0, 1):
        input_grid = clone(grid) if not off else blank()
        if off: put_shape(input_grid, 1 + y % 3, 1 + x % 4, source, 'L4')
        pairs.append({'input': input_grid, 'output': [[target if cell == source else cell for cell in row] for row in input_grid]})
    return {'train': pairs, 'test': [{'input': grid}]}, [[target if cell == source else cell for cell in row] for row in grid]


def historical_exposure_audit(rows, latent):
    candidates = (
        ROOT / 'experiments/capability_repair_baseline_v1/unified_native_model_capability_baseline_v1/SYNTHETIC_BENCHMARK_INPUT_MANIFEST_V1.json',
        ROOT / 'experiments/capability_repair_baseline_v1/e03_v7_lora_gradient_interference_diagnostic_v1/E03_V3_B1_PER_EXAMPLE_GRADIENT_SCREEN_MANIFEST_V1.json',
    )
    current_prompt_hashes = {row['prompt_sha256'] for row in rows}
    current_observations = {row['observation_sha256'] for row in rows}
    current_ids = {row['episode_id'] for row in rows}
    checked, known_prompts, known_observations, known_ids = [], set(), set(), set()
    for path in candidates:
        if not path.is_file():
            checked.append({'path': str(path.relative_to(ROOT)).replace('\\', '/'), 'status': 'MISSING_NOT_USED'})
            continue
        document = json.loads(path.read_text(encoding='utf-8-sig'))
        entries = list(document.get('episodes', []))
        if not entries:
            for family in document.get('families', []): entries.extend(family.get('members', []))
        for entry in entries:
            if isinstance(entry, dict):
                if entry.get('prompt_sha256'): known_prompts.add(entry['prompt_sha256'])
                if entry.get('observation_sha256'): known_observations.add(entry['observation_sha256'])
                if entry.get('episode_id'): known_ids.add(entry['episode_id'])
        checked.append({'path': str(path.relative_to(ROOT)).replace('\\', '/'), 'sha256': sha_file(path), 'entries_checked': len(entries)})
    if current_prompt_hashes & known_prompts or current_observations & known_observations or current_ids & known_ids:
        raise ValueError('HISTORICAL_PUBLIC_CONTENT_EXPOSURE_OVERLAP')
    # Relevant TRAIN task bytes are not present in the checked manifests; do
    # not promote a metadata audit into a transfer claim.
    return {
        'sources': checked,
        'canonical_public_prompt_overlap_count': 0,
        'canonical_public_observation_overlap_count': 0,
        'episode_id_overlap_count': 0,
        'canonical_target_content_audited_locally': True,
        'latent_structural_signatures_checked': sorted({item['canonical_latent_sha256'] for item in latent}),
        'legitimate_synthetic_train_structural_exposure': 'UNVERIFIED_MANIFESTS_DO_NOT_CONTAIN_CANONICAL_TRAIN_TASK_CONTENT',
        'claim_scope': 'WITHIN_SCHEMA_COMPOSITIONAL_EXECUTION_ONLY_NOT_HELD_OUT_TRAIN_TRANSFER',
        'passed_public_content_checks': True,
    }


def adapter_sha(path):
    document = json.loads(path.read_text(encoding='utf-8-sig'))
    return next((entry.get('sha256') for entry in document.get('adapter_files', []) if entry.get('name') == 'adapter_model.safetensors'), None)


def latent_draws(seed):
    factors = list(itertools.product(
        (1, 2, 3, 4), (5, 6, 7, 8, 9), ('L4', 'SQUARE4'),
        ('MARKER_ABOVE_ANCHOR', 'MARKER_LEFT_OF_ANCHOR'), (0, 1, 2),
        ('SHAPE_CONTRAST', 'EXTRA_MARKER_COLOUR'), (1, 2, 3),
    ))
    rng = random.Random(seed); rng.shuffle(factors)
    selected, seen = [], set()
    # Canonical latent uniqueness is necessary but insufficient: distinct
    # distractor factors can otherwise collapse to the same public task.  Test
    # all three public role prompts before assigning a tuple identifier.
    public_prompts = {role: set() for role in ('ATOMIC_SELECT_WITH_CUE', 'ATOMIC_PARAMETERIZED_RECOLOR', 'COMPOSITION_SELECT_RECOLOR')}
    for source, target, shape, relation, layout_family, distractor, distractor_colour in factors:
        if distractor_colour in (source, target): continue
        canonical_latent = {
            'source_colour': source, 'target_colour': target, 'component_shape': shape,
            'selector_relation': relation, 'layout_family': layout_family,
            'distractor_configuration': distractor, 'distractor_colour': distractor_colour,
        }
        key = sha_bytes(canonical(canonical_latent))
        if key in seen: continue
        candidate_prompts = {role: sha_bytes(prompt(task_for(canonical_latent, role)[0]).encode('utf-8')) for role in public_prompts}
        if any(value in public_prompts[role] for role, value in candidate_prompts.items()): continue
        seen.add(key); selected.append((canonical_latent, key))
        for role, value in candidate_prompts.items(): public_prompts[role].add(value)
        if len(selected) == TUPLES: return selected
    raise ValueError('INSUFFICIENT_UNIQUE_LATENT_TUPLES')


def prepare(out, sealed, seed=SEED, tuples=TUPLES, retention=RETENTION):
    if tuples != TUPLES or retention != RETENTION: raise ValueError('FROZEN_COHORT_SIZE_DRIFT')
    rows, targets, latent = [], {}, []
    for index, (params, latent_sha) in enumerate(latent_draws(seed)):
        tuple_id = f'T{index:03d}'
        entry = {'tuple_id': tuple_id, **params, 'canonical_latent_sha256': latent_sha}
        latent.append(entry)
        for role in ('ATOMIC_SELECT_WITH_CUE', 'ATOMIC_PARAMETERIZED_RECOLOR', 'COMPOSITION_SELECT_RECOLOR'):
            task, answer = task_for(params, role)
            raw, anchor, marker = raw_scene(params, 2)
            if recolor_projection(select_projection(raw, params, anchor, marker), params) != task_for(params, 'COMPOSITION_SELECT_RECOLOR')[1]:
                raise ValueError('FUNCTIONAL_PRIMITIVE_COMPOSITION_PROOF_FAILED')
            episode_id = f'{PROTOCOL}:TUPLE:{index:03d}:{role}'
            observation = {'episode_id': episode_id, 'tuple_id': tuple_id, 'role': role, 'task': task}
            rows.append({'episode_id': episode_id, 'tuple_id': tuple_id, 'role': role, 'split': 'SYNTHETIC_TRANSFER',
                         'target_access': 'SEALED_SIDECAR_ONLY', 'observation': observation,
                         'observation_sha256': sha_bytes(canonical(observation)), 'prompt_sha256': sha_bytes(prompt(task).encode('utf-8')),
                         'canonical_target_sha256': sha_bytes(canonical(answer))})
            targets[episode_id] = answer
    colours, target_colours, rng = [1, 2, 3, 4], [5, 6, 7, 8, 9], random.Random(seed + 1)
    for index in range(retention):
        source, target = rng.choice(colours), rng.choice(target_colours)
        task, answer = retention_task(source, target, index)
        episode_id = f'{PROTOCOL}:RETENTION:{index:03d}'
        observation = {'episode_id': episode_id, 'role': 'PROTECTED_RECOLOR_RETENTION', 'task': task}
        rows.append({'episode_id': episode_id, 'tuple_id': None, 'role': 'PROTECTED_RECOLOR_RETENTION', 'split': 'SYNTHETIC_RETENTION',
                     'target_access': 'SEALED_SIDECAR_ONLY', 'observation': observation,
                     'observation_sha256': sha_bytes(canonical(observation)), 'prompt_sha256': sha_bytes(prompt(task).encode('utf-8')),
                     'canonical_target_sha256': sha_bytes(canonical(answer))})
        targets[episode_id] = answer
    if len(rows) != 168 or len({row['episode_id'] for row in rows}) != 168: raise ValueError('EPISODE_IDENTITY_INVALID')
    if len({item['canonical_latent_sha256'] for item in latent}) != TUPLES: raise ValueError('CANONICAL_LATENT_DEDUPLICATION_FAILED')
    for role in ('ATOMIC_SELECT_WITH_CUE', 'ATOMIC_PARAMETERIZED_RECOLOR', 'COMPOSITION_SELECT_RECOLOR'):
        prompts = [row['prompt_sha256'] for row in rows if row['role'] == role]
        if len(prompts) != TUPLES or len(set(prompts)) != TUPLES: raise ValueError('CANONICAL_PUBLIC_PROMPT_DEDUPLICATION_FAILED')
    sidecar = sealed / 'MATCHED_ATOMIC_TO_COMPOSITIONAL_TRANSFER_V2_RC1_RC5_TARGET_SIDECAR.json'
    dump(sidecar, {'protocol_id': PROTOCOL, 'targets': targets})
    v7, fb = ROOT / V7, ROOT / FB
    if not v7.is_file() or not fb.is_file(): raise ValueError('CHECKPOINT_MANIFEST_MISSING')
    batch1_tuples = [f'T{index:03d}' for index in range(0, TUPLES, 4)]
    batch1_retention = [f'{PROTOCOL}:RETENTION:{index:03d}' for index in range(0, RETENTION, 4)]
    audit = historical_exposure_audit(rows, latent)
    manifest = {
        'schema_version': 2, 'protocol_id': PROTOCOL, 'status': 'CPU_FROZEN_TARGETS_SEALED',
        'generator': {'id': 'MATCHED_SELECTOR_RECOLOR_GENERATOR_V2_RC1_RC5', 'seed': seed, 'source_sha256': sha_file(Path(__file__))},
        'tuple_count': TUPLES, 'retention_count': RETENTION, 'episodes': rows, 'latent_parameter_tuples': latent,
        'canonical_unit': 'ONE_DEDUPLICATED_LATENT_TUPLE_WITH_ALL_THREE_ROLES',
        'functional_primitive_contract': {'select_output': 'isolated selected source-colour object plus unambiguous target-colour cue', 'recolor_input': 'exact select output', 'composition_proof': 'select_projection followed by recolor_projection equals every composition target'},
        'fixed_batch1_validation': {'tuple_ids': batch1_tuples, 'retention_episode_ids': batch1_retention, 'episode_count': 42, 'selection': 'every fourth canonical tuple plus every fourth retention row'},
        'content_exposure_audit': audit,
        'sealed_target_sidecar': {'local_path': str(sidecar.resolve()), 'sha256': sha_file(sidecar), 'target_count': len(targets), 'not_committed': True},
    }
    dump(out / 'COHORT_MANIFEST.json', manifest)
    protocol = {
        'schema_version': 2, 'experiment_id': PROTOCOL, 'status': 'CPU_FROZEN_NOT_AUTHORIZED_FOR_MODEL_EXECUTION',
        'cohort_manifest_path': 'COHORT_MANIFEST.json', 'cohort_manifest_sha256': sha_file(out / 'COHORT_MANIFEST.json'),
        'checkpoints': {'RECONSTRUCTED_FOUNDATION_V2_V7': {'manifest_path': V7, 'manifest_sha256': sha_file(v7), 'adapter_model_sha256': adapter_sha(v7), 'role': 'PRIMARY'}, 'FAMILY_BALANCED': {'manifest_path': FB, 'manifest_sha256': sha_file(fb), 'adapter_model_sha256': adapter_sha(fb), 'role': 'DESCRIPTIVE_FROZEN_HISTORICAL_COMPARATOR'}},
        'inference': {'format': 'NATIVE_ARC_OBSERVATION', 'decoding': 'GREEDY_PRIMARY_ONLY', 'batch_primary': 16, 'fallback_ladder': [16, 8, 4, 1], 'batch1_validation': 'fixed 42-episode subset; material parsed-output or metric drift is INVALID_NOT_INTERPRETABLE', 'forbidden': ['Beam', 'DFS', 'TTT', 'augmentation', 'external_selector', 'Gold', 'dGold', 'FINAL_AUDIT']},
        'metrics': {'primary': 'V7 conditional composition-failure rate on V7 own both-atoms-correct deduplicated tuples', 'secondary': ['Family-Balanced analogous conditional rate DESCRIPTIVE_ONLY', 'V7-vs-Family-Balanced paired conditional contrast ONLY on common-support tuples', 'joint_atomic_prerequisite_exact_rate', 'unconditional_composition_exact_rate', 'four_state_tuple_counts', 'separate_synthetic_protected_retention'], 'common_support_requirement': 'at least 12 deduplicated tuples where both atomic prerequisites are exact for both checkpoints; otherwise NOT_ESTIMABLE'},
        'uncertainty': {'bootstrap': '10000 fixed seed 20261010 resampling canonical unique tuples with all roles inseparable; strata frozen by selector_relation x component_shape', 'decision_boundary': 'No 0.15 mechanism boundary: report conditional interval descriptively; label MIXED_OR_INCONCLUSIVE unless a future independently justified decision threshold is pre-registered'},
        'decision_rule': {'COMPOSITION_EXECUTION_EVIDENCE': 'descriptive only when V7 both-atoms condition has at least 12 canonical tuples and Batch1 validation is stable', 'MIXED_OR_INCONCLUSIVE': 'minimum n unmet, interval broad, comparator common support unavailable, or Batch1 validation changes conclusion', 'INVALID_NOT_INTERPRETABLE': 'identity, public-content exposure, tuple matching, functional primitive proof, serialization, target, checkpoint, batching, parser, completeness, or sealed-data failure'},
        'claim_scope': audit['claim_scope'], 'runtime_cap_seconds': 1800, 'optimizer_steps': 0, 'parameter_updates': 0, 'model_execution_authorized': False, 'training_authorized': False,
    }
    dump(out / 'PROTOCOL.json', protocol)
    return {'manifest': out / 'COHORT_MANIFEST.json', 'protocol': out / 'PROTOCOL.json', 'sidecar': sidecar}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True); parser.add_argument('--sealed-root', type=Path, required=True); parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    if args.out.exists(): raise SystemExit('OUTPUT_ROOT_MUST_BE_FRESH')
    result = prepare(args.out, args.sealed_root, args.seed)
    print(json.dumps({name: {'path': str(path), 'sha256': sha_file(path)} for name, path in result.items()}, sort_keys=True))


if __name__ == '__main__': main()
