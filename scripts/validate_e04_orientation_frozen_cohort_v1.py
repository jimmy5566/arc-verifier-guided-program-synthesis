from __future__ import annotations

import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COHORT_DIR = ROOT / 'experiments/capability_repair_baseline_v1/e04_orientation_frozen_cohort_v1'
COHORT = COHORT_DIR / 'COHORT.jsonl'
EXPECTED_SHA256 = '76c6fd57173ec62571f27a312e3f80601c70cdec04cda79432822937e7a37967'
CONDITIONS = {
    'ROTATION_TARGET', 'FIXED_TURN_ROTATION_CONTROL',
    'MARKER_BINDING_CONTROL', 'NO_TRANSFORM_RETENTION_CONTROL',
}
LAYOUTS = {
    'CENTER': (3, 3), 'NW': (1, 1), 'NE': (1, 5), 'SW': (5, 1), 'SE': (5, 5),
    'MID_WEST': (3, 1), 'MID_EAST': (3, 5), 'MID_NORTH': (1, 3), 'MID_SOUTH': (5, 3),
    'OFFSET_A': (2, 2), 'OFFSET_B': (2, 4), 'OFFSET_C': (4, 2),
}
RELATION_MARKERS = {'NORTH': (0, 4), 'EAST': (4, 8), 'SOUTH': (8, 4), 'WEST': (4, 0)}
BASE_FIELDS = ('shape_orientation', 'control_marker_turn', 'reference_relation', 'topology', 'layout_template', 'color_role_tuple', 'instance_index')

class CohortValidationError(RuntimeError):
    pass

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def canonical_content(grid_input, grid_target) -> str:
    data = json.dumps({'input': grid_input, 'target': grid_target}, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return sha256_bytes(data)

def base_signature(base: dict) -> str:
    data = {key: base[key] for key in BASE_FIELDS}
    return sha256_bytes(json.dumps(data, sort_keys=True, separators=(',', ':')).encode('utf-8'))

def rotate(points, turns):
    result = list(points)
    for _ in range(turns % 4):
        result = [(x, 2 - y) for y, x in result]
    return result

def independent_oracle(base: dict, condition: str) -> tuple[list[list[int]], list[list[int]]]:
    # This implementation deliberately does not import the frozen generator.
    bg, shape, marker, output = base['color_role_tuple']
    grid = [[bg for _ in range(9)] for _ in range(9)]
    py, px = LAYOUTS[base['layout_template']]
    for y, x in rotate([(0, 0), (1, 0), (2, 0), (2, 1)], base['shape_orientation']):
        grid[py + y][px + x] = shape
    my, mx = RELATION_MARKERS[base['reference_relation']]
    grid[my][mx] = marker
    grid[0][base['control_marker_turn']] = marker
    if condition == 'MARKER_BINDING_CONTROL':
        target = [[output if x == base['control_marker_turn'] else bg for x in range(4)]]
    else:
        turns = base['control_marker_turn'] if condition == 'ROTATION_TARGET' else 1 if condition == 'FIXED_TURN_ROTATION_CONTROL' else 0
        target = [[bg for _ in range(3)] for _ in range(3)]
        for y, x in rotate([(0, 0), (1, 0), (2, 0), (2, 1)], (base['shape_orientation'] + turns) % 4):
            target[y][x] = output
    return grid, target

def exact_reconstruction(rows: list[dict], newline: bytes) -> bytes:
    return b''.join(json.dumps(row, sort_keys=True).encode('utf-8') + newline for row in rows)

def detect_newline(raw: bytes) -> tuple[str, bytes]:
    if b'\r\n' in raw and raw.replace(b'\r\n', b'').find(b'\n') == -1 and raw.replace(b'\r\n', b'').find(b'\r') == -1:
        return 'CRLF', b'\r\n'
    if b'\r' not in raw and b'\n' in raw:
        return 'LF', b'\n'
    raise CohortValidationError('SERIALIZATION_NEWLINE_INVALID')

def allowed_collision(group: list[dict]) -> bool:
    if len(group) != 2:
        return False
    base = group[0]['base_tuple']
    if any(row['base_tuple'] != base for row in group):
        return False
    names = {row['condition'] for row in group}
    turn = base['control_marker_turn']
    return ((turn == 0 and names == {'ROTATION_TARGET', 'NO_TRANSFORM_RETENTION_CONTROL'}) or
            (turn == 1 and names == {'ROTATION_TARGET', 'FIXED_TURN_ROTATION_CONTROL'}))

def validate_collision_topology(rows: list[dict]) -> tuple[dict[str, list[dict]], list[list[dict]]]:
    """Classify content reuse independently of the rendering-oracle checks."""
    contents = defaultdict(list)
    for row in rows:
        contents[row['content_sha256']].append(row)
    duplicate_groups = []
    train_content, validation_content = set(), set()
    for content, group in contents.items():
        signatures = {r['latent_signature_sha256'] for r in group}
        if len(signatures) != 1:
            raise CohortValidationError('CROSS_BASE_CONTENT_COLLISION')
        group_splits = {r['base_tuple']['split'] for r in group}
        if len(group_splits) != 1:
            raise CohortValidationError('CROSS_SPLIT_CONTENT_COLLISION')
        (train_content if next(iter(group_splits)) == 'TRAIN' else validation_content).add(content)
        if len(group) > 1:
            if not allowed_collision(group):
                raise CohortValidationError('UNDECLARED_WITHIN_BASE_COLLISION')
            duplicate_groups.append(group)
    if train_content & validation_content:
        raise CohortValidationError('CROSS_SPLIT_CONTENT_COLLISION')
    return contents, duplicate_groups

def validate_turn_contingencies(rows: list[dict]) -> dict:
    """Validate the two observable shortcut exclusions without relying on model code."""
    contingency = {split: {'by_layout': defaultdict(set), 'by_color_tuple': defaultdict(set)} for split in ('TRAIN', 'VALIDATION')}
    for row in rows:
        base = row['base_tuple']
        contingency[base['split']]['by_layout'][base['layout_template']].add(base['control_marker_turn'])
        contingency[base['split']]['by_color_tuple'][tuple(base['color_role_tuple'])].add(base['control_marker_turn'])
    for _, tables in contingency.items():
        if any(turns != {0, 1, 2, 3} for turns in tables['by_layout'].values()):
            raise CohortValidationError('LAYOUT_TO_TURN_CONFOUNDING')
        if any(turns != {0, 1, 2, 3} for turns in tables['by_color_tuple'].values()):
            raise CohortValidationError('COLOR_TO_TURN_CONFOUNDING')
    return contingency

def validate_rows(rows: list[dict]) -> dict:
    if len(rows) != 1344:
        raise CohortValidationError('ROW_COUNT_MISMATCH')
    bases = defaultdict(list)
    splits = defaultdict(set)
    oracle_checked = 0
    for row in rows:
        if set(row) != {'base_tuple', 'condition', 'input', 'target', 'content_sha256', 'latent_signature_sha256'}:
            raise CohortValidationError('ROW_SCHEMA_MISMATCH')
        base = row['base_tuple']
        if row['condition'] not in CONDITIONS:
            raise CohortValidationError('UNKNOWN_CONDITION')
        signature = base_signature(base)
        if row['latent_signature_sha256'] != signature:
            raise CohortValidationError('BASE_SIGNATURE_MISMATCH')
        expected_input, expected_target = independent_oracle(base, row['condition'])
        if row['input'] != expected_input or row['target'] != expected_target:
            raise CohortValidationError('INDEPENDENT_ORACLE_MISMATCH')
        if row['content_sha256'] != canonical_content(row['input'], row['target']):
            raise CohortValidationError('CONTENT_HASH_MISMATCH')
        bases[signature].append(row)
        splits[signature].add(base['split'])
        oracle_checked += 1
    if len(bases) != 336 or any(len(group) != 4 or {r['condition'] for r in group} != CONDITIONS for group in bases.values()):
        raise CohortValidationError('BASE_CONDITION_MAP_MISMATCH')
    if any(len(value) != 1 for value in splits.values()):
        raise CohortValidationError('CROSS_SPLIT_BASE_COLLISION')
    base_splits = defaultdict(set)
    for signature, group in bases.items():
        base_splits[group[0]['base_tuple']['split']].add(signature)
    if len(base_splits['TRAIN']) != 288 or len(base_splits['VALIDATION']) != 48 or base_splits['TRAIN'] & base_splits['VALIDATION']:
        raise CohortValidationError('BASE_SPLIT_CARDINALITY_MISMATCH')
    contents, duplicate_groups = validate_collision_topology(rows)
    contingency = validate_turn_contingencies(rows)
    allowed_classes = defaultdict(int)
    for group in duplicate_groups:
        base = group[0]['base_tuple']
        names = tuple(sorted(r['condition'] for r in group))
        allowed_classes[(base['control_marker_turn'], names)] += 1
    if len(duplicate_groups) != 168 or dict(allowed_classes) != {
        (0, ('NO_TRANSFORM_RETENTION_CONTROL', 'ROTATION_TARGET')): 84,
        (1, ('FIXED_TURN_ROTATION_CONTROL', 'ROTATION_TARGET')): 84,
    }:
        raise CohortValidationError('PERMITTED_COLLISION_COUNT_MISMATCH')
    return {
        'rows': len(rows), 'base_tuple_count': len(bases), 'train_base_tuple_count': len(base_splits['TRAIN']),
        'validation_base_tuple_count': len(base_splits['VALIDATION']), 'unique_content_hashes': len(contents),
        'permitted_same_base_collision_classes': len(duplicate_groups), 'cross_base_content_unique': True,
        'cross_split_content_disjoint': True, 'independent_oracle_rows_checked': oracle_checked,
        'turn_contingency': {split: {axis: {str(key): sorted(value) for key, value in table.items()} for axis, table in axes.items()} for split, axes in contingency.items()},
        'collision_class_counts': {f'turn={turn};conditions={"|".join(names)}': count for (turn, names), count in allowed_classes.items()},
        'condition_to_base': {signature: sorted(row['condition'] for row in group) for signature, group in bases.items()},
    }

def load_and_validate(cohort_path: Path = COHORT) -> tuple[list[dict], bytes, str, dict]:
    raw = cohort_path.read_bytes()
    if sha256_bytes(raw) != EXPECTED_SHA256:
        raise CohortValidationError('COHORT_SHA256_MISMATCH')
    newline_name, newline = detect_newline(raw)
    rows = [json.loads(line) for line in raw.splitlines()]
    result = validate_rows(rows)
    if exact_reconstruction(rows, newline) != raw:
        raise CohortValidationError('SERIALIZATION_RECONSTRUCTION_MISMATCH')
    result.update({'cohort_sha256': EXPECTED_SHA256, 'cohort_bytes': len(raw), 'newline': newline_name, 'serialization': f'UTF-8 {newline_name} JSONL, one sorted-key JSON object plus {newline_name} per row'})
    with tempfile.TemporaryDirectory(prefix='e04-cohort-reconstruction-') as temporary_directory:
        recreated = Path(temporary_directory) / 'COHORT.jsonl'
        recreated.write_bytes(exact_reconstruction(rows, newline))
        recreated_sha = sha256_bytes(recreated.read_bytes())
        if recreated_sha != EXPECTED_SHA256:
            raise CohortValidationError('TEMPORARY_REGENERATION_SHA256_MISMATCH')
        result['temporary_directory_regeneration'] = {'status': 'PASS', 'reconstructed_sha256': recreated_sha, 'temporary_output_bytes': recreated.stat().st_size}
    return rows, raw, newline_name, result

def write_reports() -> dict:
    rows, _, _, report = load_and_validate()
    meta = {
        'cohort_sha256': report['cohort_sha256'], 'content_unique': False,
        'cross_base_content_unique': True, 'cross_split_content_disjoint': True,
        'unique_content_hashes': report['unique_content_hashes'], 'permitted_same_base_collision_classes': report['permitted_same_base_collision_classes'],
        'allowed_equivalence_classes': [
            'ROTATION_TARGET=NO_TRANSFORM_RETENTION_CONTROL at turn=0',
            'ROTATION_TARGET=FIXED_TURN_ROTATION_CONTROL at turn=1',
        ], 'serialization': report['serialization'], 'cohort_bytes': report['cohort_bytes'],
        'status': 'PASS_CPU_ONLY_SUCCESSOR_METADATA_CLOSURE_V2',
    }
    outputs = {
        'SUCCESSOR_METADATA_V2.json': meta,
        'TEMPORARY_DIRECTORY_DETERMINISTIC_REGENERATION_REPORT_V1.json': report['temporary_directory_regeneration'] | {'cohort_sha256': report['cohort_sha256'], 'serialization': report['serialization']},
        'INDEPENDENT_TARGET_ORACLE_REPORT_V2.json': {'status': 'PASS', 'oracle_implementation': 'scripts/validate_e04_orientation_frozen_cohort_v1.py:independent_oracle', 'rows_checked': report['independent_oracle_rows_checked'], 'cohort_sha256': report['cohort_sha256']},
        'COLLISION_REPORT_V2.json': {'status': 'PASS', 'content_unique': False, 'unique_content_hashes': report['unique_content_hashes'], 'permitted_same_base_collision_classes': report['permitted_same_base_collision_classes'], 'collision_class_counts': report['collision_class_counts'], 'cross_base_content_unique': True, 'cross_split_content_disjoint': True},
        'TURN_CONTINGENCY_V2.json': {'status': 'PASS', 'turns_required_per_factor': [0, 1, 2, 3], **report['turn_contingency']},
        'CONDITION_TO_BASE_MAP_V2.json': {'status': 'PASS', 'base_tuple_count': report['base_tuple_count'], 'conditions_per_base': 4, 'map': report['condition_to_base']},
    }
    for filename, payload in outputs.items():
        path = COHORT_DIR / filename
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    return report

def main() -> int:
    report = write_reports()
    print(json.dumps({key: value for key, value in report.items() if key not in {'turn_contingency', 'condition_to_base'}}, sort_keys=True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
