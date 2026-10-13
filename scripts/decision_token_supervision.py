"""CPU-only count-code-pair locator and mass-preserving supervision weights."""
from __future__ import annotations


def _grid_length(grid):
    return len("\n".join("".join(map(str, row)) for row in grid))


def count_code_pair_label_indices(task):
    """Locate row-1 columns 1/2 in every supervised count output.

    ``task_to_sample`` serializes each input/output pair as a user turn followed
    by an assistant turn.  The two selected cells are count codes: for a run of
    length 1 they are both zero; for length 2 only column 1 is foreground; for
    length 3 both are foreground.  They therefore identify count independently
    of the foreground color.
    """
    pairs = [*task['train'], *task['test']]
    if len(pairs) != 4:
        raise ValueError('COUNT_PAIR_COUNT')
    indices, counts = [], []
    offset = 0
    for pair in pairs:
        grid = pair['output']
        if len(grid) != 3 or any(len(row) != 3 for row in grid):
            raise ValueError('COUNT_OUTPUT_SHAPE')
        run = [column for column, value in enumerate(grid[1]) if value != 0]
        if not run or run != list(range(len(run))) or any(value != 0 for value in grid[0] + grid[2]):
            raise ValueError('COUNT_OUTPUT_SEMANTICS')
        # user turn then assistant header <im_start>, assistant, newline.
        assistant_content = offset + 3 + _grid_length(pair['input']) + 1 + 3
        indices.extend((assistant_content + 4, assistant_content + 5))
        counts.append(len(run))
        offset += 3 + _grid_length(pair['input']) + 1 + 3 + _grid_length(grid) + 1
    return tuple(indices), tuple(counts)


def matched_weights(labels, decision_indices, multiplier=2.0):
    valid = [index for index, value in enumerate(labels) if value != -100]
    focused = tuple(decision_indices)
    if len(focused) != 8 or len(set(focused)) != 8 or any(index not in valid for index in focused):
        raise ValueError('COUNT_CODE_PAIR_INDEX_INVALID')
    if not multiplier > 0 or len(valid) <= len(focused):
        raise ValueError('LOSS_MASS_PARAMETERS')
    compensation = (len(valid) - multiplier * len(focused)) / (len(valid) - len(focused))
    if compensation <= 0:
        raise ValueError('LOSS_MASS_MISMATCH')
    control = {index: 1.0 for index in valid}
    treatment = {index: (multiplier if index in focused else compensation) for index in valid}
    if abs(sum(control.values()) - sum(treatment.values())) > 1e-12:
        raise ValueError('LOSS_MASS_MISMATCH')
    return control, treatment
