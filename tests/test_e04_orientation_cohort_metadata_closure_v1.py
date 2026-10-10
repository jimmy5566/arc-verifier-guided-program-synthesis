from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_e04_orientation_frozen_cohort_v1 import (
    CohortValidationError, load_and_validate, validate_collision_topology,
    validate_turn_contingencies,
)

def rejects(operation, code):
    try:
        operation()
    except CohortValidationError as exc:
        assert str(exc) == code
    else:
        raise AssertionError(code)

def test_positive_and_temporary_reconstruction():
    _, _, newline, report = load_and_validate()
    assert newline == 'CRLF'
    assert report['temporary_directory_regeneration']['status'] == 'PASS'
    assert report['unique_content_hashes'] == 1176

def test_negative_collision_and_confounding_cases():
    rows, _, _, _ = load_and_validate()
    cross_base = deepcopy(rows)
    cross_base[4]['content_sha256'] = cross_base[0]['content_sha256']
    rejects(lambda: validate_collision_topology(cross_base), 'CROSS_BASE_CONTENT_COLLISION')
    cross_split = deepcopy(rows)
    source = next(row for row in cross_split if row['base_tuple']['split'] == 'TRAIN')
    target = next(row for row in cross_split if row['base_tuple']['split'] == 'VALIDATION')
    target['content_sha256'] = source['content_sha256']
    target['latent_signature_sha256'] = source['latent_signature_sha256']
    rejects(lambda: validate_collision_topology(cross_split), 'CROSS_SPLIT_CONTENT_COLLISION')
    undeclared = deepcopy(rows)
    target = next(row for row in undeclared if row['condition'] == 'MARKER_BINDING_CONTROL')
    source = next(row for row in undeclared if row['condition'] == 'ROTATION_TARGET' and row['base_tuple']['control_marker_turn'] == 0)
    target['content_sha256'] = source['content_sha256']
    target['base_tuple'] = deepcopy(source['base_tuple'])
    target['latent_signature_sha256'] = source['latent_signature_sha256']
    rejects(lambda: validate_collision_topology(undeclared), 'UNDECLARED_WITHIN_BASE_COLLISION')
    layout_confounded = deepcopy(rows)
    for row in layout_confounded:
        if row['base_tuple']['split'] == 'TRAIN' and row['base_tuple']['layout_template'] == 'CENTER':
            row['base_tuple']['control_marker_turn'] = 0
    rejects(lambda: validate_turn_contingencies(layout_confounded), 'LAYOUT_TO_TURN_CONFOUNDING')
    color_confounded = deepcopy(rows)
    color = deepcopy(color_confounded[0]['base_tuple']['color_role_tuple'])
    for row in color_confounded:
        if row['base_tuple']['color_role_tuple'] == color:
            row['base_tuple']['control_marker_turn'] = 0
    rejects(lambda: validate_turn_contingencies(color_confounded), 'COLOR_TO_TURN_CONFOUNDING')

if __name__ == '__main__':
    test_positive_and_temporary_reconstruction()
    test_negative_collision_and_confounding_cases()
    print('E04_COHORT_METADATA_CLOSURE_NEGATIVE_TESTS_PASS')
