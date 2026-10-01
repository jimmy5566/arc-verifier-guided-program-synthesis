#!/usr/bin/env python3
"""Freeze the public-NVARC augmentation audit and a target-blind research pool.

This is deliberately CPU-only.  It reads only ARC challenge JSON (train pairs and
test inputs), never solutions, models, tokenizers, TTT state, or candidate pools.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PUBLIC_COMMIT = "846d0198efa752534594e321fc3289fc0a06c657"
PUBLIC_NOTEBOOK = "ARC-AGI1/002_ivan_arc1.ipynb"
PUBLIC_NOTEBOOK_SHA256 = "ec0ac985d7de396c7fb7bede70cc0de40a2161d3b6ec00a123d41024fb8e3104"
SOURCE_URL = (
    "https://raw.githubusercontent.com/1ytic/NVARC/"
    f"{PUBLIC_COMMIT}/{PUBLIC_NOTEBOOK}"
)
SCREEN_SALT = "ARC2_AUGMENTATION_CANDIDATE_SCREENING_V1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def grid_to_text(grid: list[list[int]]) -> str:
    return "\n".join("".join(str(int(value)) for value in row) for row in grid)


def rot90(grid: list[list[int]]) -> list[list[int]]:
    # np.rot90(grid, 1): counter-clockwise, which is the public NVARC operation.
    return [list(row) for row in zip(*grid)][::-1]


def transpose(grid: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in zip(*grid)]


def transform_geometry(grid: list[list[int]], geometry: str) -> list[list[int]]:
    if geometry == "identity":
        return [row[:] for row in grid]
    if geometry == "rot90":
        return rot90(grid)
    if geometry == "rot180":
        return rot90(rot90(grid))
    if geometry == "rot270":
        return rot90(rot90(rot90(grid)))
    if geometry == "flip_lr":
        return [list(reversed(row)) for row in grid]
    if geometry == "flip_ud":
        return [row[:] for row in reversed(grid)]
    if geometry == "transpose":
        return transpose(grid)
    if geometry == "anti_transpose":
        return rot90(rot90(transpose(grid)))
    raise ValueError(f"unsupported geometry: {geometry}")


INVERSE_GEOMETRY = {
    "identity": "identity",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "flip_lr": "flip_lr",
    "flip_ud": "flip_ud",
    "transpose": "transpose",
    "anti_transpose": "anti_transpose",
}


def inverse_mapping(mapping: str) -> str:
    inverse = [0] * 10
    for source, target in enumerate(mapping):
        inverse[int(target)] = source
    return "".join(str(item) for item in inverse)


def recolor(grid: list[list[int]], mapping: str) -> list[list[int]]:
    return [[int(mapping[int(value)]) for value in row] for row in grid]


def order_train(train: list[dict[str, Any]], order: str) -> list[dict[str, Any]]:
    if order == "canonical":
        return list(train)
    if order == "reverse":
        return list(reversed(train))
    if order == "rotate_left_1":
        return list(train[1:]) + list(train[:1])
    raise ValueError(f"unsupported order: {order}")


def transform_example(example: dict[str, Any], geometry: str, color: str) -> dict[str, Any]:
    transformed = {"input": recolor(transform_geometry(example["input"], geometry), color)}
    if "output" in example:
        transformed["output"] = recolor(transform_geometry(example["output"], geometry), color)
    return transformed


def public_formatter(task: dict[str, Any], candidate: dict[str, Any]) -> str:
    """Exact public QwenFormatter text layout, without tokenizing or loading Qwen."""
    geometry = candidate["geometry"]
    color = candidate["color_mapping"]
    train = [transform_example(example, geometry, color) for example in task["train"]]
    train = order_train(train, candidate["demo_order"])
    test = transform_example(task["test"][0], geometry, color)
    chunks: list[str] = []
    for example in train:
        chunks.append("<|im_start|>user\n" + grid_to_text(example["input"]))
        chunks.append("<|im_end|><|im_start|>assistant\n" + grid_to_text(example["output"]))
        chunks.append("<|im_end|>")
    chunks.append("<|im_start|>user\n" + grid_to_text(test["input"]))
    chunks.append("<|im_end|><|im_start|>assistant\n")
    return "".join(chunks)


# These are the first public eval-draw descriptors from np.random.seed(2) with
# np.random.permutation(10), recorded as old-colour -> new-colour strings.
PUBLIC_SEED2_PERMUTATIONS = [
    "4150723698", "1602397845", "4950812673", "5980213764",
    "5386701942", "5436201789", "1654278039", "7690351428",
    "9302675184", "9180254367", "4281673590", "4907281356",
    "8579032416", "0986123745", "4965803712", "6142358709",
]


def make_candidate(candidate_id: str, geometry: str, color_name: str, color_mapping: str, demo_order: str, source_family: str, rationale: str) -> dict[str, Any]:
    if sorted(int(value) for value in color_mapping) != list(range(10)):
        raise ValueError(f"non-bijective colour mapping for {candidate_id}")
    return {
        "candidate_id": candidate_id,
        "geometry": geometry,
        "inverse_geometry": INVERSE_GEOMETRY[geometry],
        "color_name": color_name,
        "color_mapping": color_mapping,
        "inverse_color_mapping": inverse_mapping(color_mapping),
        "demo_order": demo_order,
        "rng_source": "none; explicit frozen mapping/order",
        "source_family": source_family,
        "provenance_class": "PROJECT_DEFINED",
        "label_solution_preserving": True,
        "test_target_independent": True,
        "gold_independent": True,
        "rationale": rationale,
    }


def build_candidates() -> tuple[list[dict[str, Any]], list[str], list[str], list[str], list[str]]:
    identity = "0123456789"
    d4_order = ["identity", "flip_ud", "transpose", "anti_transpose", "rot90", "rot180", "rot270", "flip_lr"]
    candidates: list[dict[str, Any]] = []
    for geometry in d4_order:
        candidates.append(make_candidate(
            f"geom={geometry}__color=id__order=canonical", geometry, "id", identity, "canonical",
            "PROJECT_D4_BASELINE", "Mandatory D4 structural baseline; uses no colour or order perturbation."
        ))
    aug4 = [item["candidate_id"] for item in candidates[:4]]
    aug8 = [item["candidate_id"] for item in candidates]

    additions16 = [
        ("identity", "p00_4150723698", PUBLIC_SEED2_PERMUTATIONS[0], "canonical", "Public seed=2 colour draw on identity isolates colour representation."),
        ("rot90", "p02_4950812673", PUBLIC_SEED2_PERMUTATIONS[2], "canonical", "Geometry-times-colour condition with an independent public seed=2 draw."),
        ("flip_ud", "p03_5980213764", PUBLIC_SEED2_PERMUTATIONS[3], "canonical", "Geometry-times-colour condition spanning a reflection."),
        ("anti_transpose", "p05_5436201789", PUBLIC_SEED2_PERMUTATIONS[5], "canonical", "Geometry-times-colour condition spanning the anti-diagonal symmetry."),
        ("identity", "id", identity, "reverse", "Pure deterministic demonstration-order intervention."),
        ("rot180", "id", identity, "rotate_left_1", "Geometry-times-demo-order intervention with no recolouring."),
        ("transpose", "p01_1602397845", PUBLIC_SEED2_PERMUTATIONS[1], "reverse", "Combined geometry, public-colour and order condition."),
        ("flip_lr", "p07_7690351428", PUBLIC_SEED2_PERMUTATIONS[7], "rotate_left_1", "Combined reflection, public-colour and order condition."),
    ]
    for geometry, color_name, mapping, order, rationale in additions16:
        candidates.append(make_candidate(
            f"geom={geometry}__color={color_name}__order={order}", geometry, color_name, mapping, order,
            "PROJECT_COMBINED_PUBLIC_COLOR_SEED2", rationale
        ))
    aug16 = [item["candidate_id"] for item in candidates]

    additions24 = [
        ("identity", "p06_1654278039", PUBLIC_SEED2_PERMUTATIONS[6], "canonical", "Additional public-colour draw on identity."),
        ("rot270", "p14_4965803712", PUBLIC_SEED2_PERMUTATIONS[14], "canonical", "Additional geometry-times-colour condition."),
        ("flip_lr", "p07_7690351428", PUBLIC_SEED2_PERMUTATIONS[7], "canonical", "Separates the flip-left-right colour condition from its AUG16 order perturbation."),
        ("transpose", "p09_9180254367", PUBLIC_SEED2_PERMUTATIONS[9], "canonical", "Additional transposed public-colour condition."),
        ("identity", "id", identity, "rotate_left_1", "Pure rotate-left demonstration-order condition."),
        ("rot90", "id", identity, "rotate_left_1", "Rotational geometry-times-order condition."),
        ("flip_ud", "id", identity, "reverse", "Reflection-times-order condition."),
        ("anti_transpose", "id", identity, "reverse", "Anti-diagonal geometry-times-order condition."),
    ]
    for geometry, color_name, mapping, order, rationale in additions24:
        candidates.append(make_candidate(
            f"geom={geometry}__color={color_name}__order={order}", geometry, color_name, mapping, order,
            "PROJECT_AUG24_EXTENSION", rationale
        ))
    aug24 = [item["candidate_id"] for item in candidates]
    if len(candidates) != 24 or len(set(aug24)) != 24:
        raise AssertionError("candidate pool must contain 24 distinct stable IDs")
    return candidates, aug4, aug8, aug16, aug24


def public_inference_recipe() -> dict[str, Any]:
    # In augment(n=2), mod(..., stack=False) makes draw i cover all eight
    # geometries.  eval_ds.keys is sorted before the slot/batch construction.
    # eval_ds.keys is sorted at source line 765.  This is the resulting
    # lexicographic geometry order, not ArcDataset.augment's construction
    # order (identity, transpose, rot90, flip_ud, ...).
    source_geometry = [
        ("identity", 0), ("rot90", 2), ("rot180", 4), ("rot270", 6),
        ("transpose", 1), ("flip_ud", 3), ("anti_transpose", 5), ("flip_lr", 7),
    ]
    rows: list[dict[str, Any]] = []
    for geometry, source_index in source_geometry:
        descriptors = [PUBLIC_SEED2_PERMUTATIONS[source_index], PUBLIC_SEED2_PERMUTATIONS[8 + source_index]]
        for descriptor in sorted(descriptors):
            rows.append({
                "sorted_subkey_slot": None,
                "geometry": geometry,
                "color_mapping_old_to_new": descriptor,
                "inverse_color_mapping_new_to_old": inverse_mapping(descriptor),
                "demo_order": "np.random.permutation(n_train); task-dependent draw after all 16 colour draws",
                "rng": "np.random.seed(2) inside ArcDataset.augment; NumPy version not pinned in source",
                "output_inverse": "ArcDataset.invert_mod(array, subkey, inv_perm=True)",
            })
    for index, row in enumerate(rows):
        row["sorted_subkey_slot"] = index
    physical_order = [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]
    slot_to_batch = {slot: batch for batch, slots in enumerate(physical_order) for slot in slots}
    for row in rows:
        row["physical_dfs_batch"] = slot_to_batch[row["sorted_subkey_slot"]]
    return {
        "classification": "PUBLIC_DFS_AUG16_CONFIRMED",
        "configured_distinct_dfs_cells_per_task_output": 16,
        "physical_batch_count": 4,
        "physical_batch_width": 4,
        "source_slot_order_after_sorted_eval_keys": rows,
        "actual_batch_selection_slot_order": physical_order,
        "descriptor_observation": {
            "purpose": "Mechanical expansion of the source algorithm only; not a claim of a portable public canonical recipe.",
            "emulation": "np.random.seed(2); np.random.permutation(10), observed with NumPy 2.5.3 legacy global RNG",
            "source_limit": "The public notebook does not pin a NumPy version.",
        },
        "execution_caveat": "The source has a 1200-second per-puzzle timeout check before each batch; configured 16 cells can be truncated by timeout.",
        "fixed_canonical_aug16": False,
        "why_not_fixed_canonical": "Colour draws use global NumPy RNG with no pinned NumPy version, and shuffle_ex uses task-dependent random train-pair permutations.",
    }


def select_representative_tasks(tasks: dict[str, Any]) -> list[str]:
    ranked = sorted(tasks, key=lambda task_id: hashlib.sha256(f"{SCREEN_SALT}:{task_id}".encode("utf-8")).hexdigest())
    return ranked[:8]


def build_public_source_provenance() -> dict[str, Any]:
    return {
        "audit_scope": "public-source recovery only; no model, GPU, TTT, DFS, Gold, or target output was used",
        "kaggle_notebook": {
            "url": "https://www.kaggle.com/code/sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue",
            "owner": "sorokin (Ivan)",
            "script_version_id_discovered_from_oembed": 276863151,
            "direct_export_status": "UNAVAILABLE_TO_THIS_AUDIT; anonymous public API endpoints returned 401/403/404",
        },
        "authoritative_mirror": {
            "repo": "https://github.com/1ytic/NVARC",
            "commit": PUBLIC_COMMIT,
            "commit_message": "Update README.md",
            "notebook": PUBLIC_NOTEBOOK,
            "notebook_raw_url": SOURCE_URL,
            "notebook_sha256": PUBLIC_NOTEBOOK_SHA256,
            "mirror_relationship": "ARC-AGI1/README.md states the code is the same as the Kaggle winning notebook, with only test-time fine-tuning evaluation-data changes.",
            "mirror_relationship_source": "ARC-AGI1/README.md at the pinned commit",
        },
        "source_confidence": "HIGH_FOR_AUGMENTATION_CONTROL_FLOW; the repository explicitly identifies this notebook as same-code mirror of the Kaggle notebook.",
    }


def notebook_references() -> str:
    return """# Public NVARC augmentation code references

Source: [`ARC-AGI1/002_ivan_arc1.ipynb`](https://raw.githubusercontent.com/1ytic/NVARC/846d0198efa752534594e321fc3289fc0a06c657/ARC-AGI1/002_ivan_arc1.ipynb) at commit `846d0198efa752534594e321fc3289fc0a06c657` (SHA-256 `ec0ac985d7de396c7fb7bede70cc0de40a2161d3b6ec00a123d41024fb8e3104`).

The references below use **logical source-code lines** formed by concatenating code-cell `source` arrays in notebook order; the embedded `%%writefile` targets identify the intended Python file.

| Control | Exact source reference | Audit reading |
|---|---|---|
| Colour permutation | embedded `arc_loader.py`, lines 24–39 (`permute_mod`, `permute_rnd_all_`) | A permutation descriptor is a 10-element bijection. For 2D grids it maps old colour `i` to descriptor digit `i`; inversion uses the inverse permutation. |
| D4 transform / inverse | embedded `arc_loader.py`, lines 91–119 (`ArcDataset.forward_mod`, `ArcDataset.invert_mod`) | `transpose`, then three rotations, yields all eight dihedral representations; output arrays are inverse-transformed before storage. |
| Dataset expansion | embedded `arc_loader.py`, lines 183–191 and 251–258 (`ArcDataset.mod`, `ArcDataset.augment`) | `augment` sets NumPy's global seed, builds transpose × four rotations, then makes `n` independent colour draws and finally calls `shuffle_ex`. |
| Train/demo ordering | embedded `arc_loader.py`, lines 237–249 (`ArcDataset.shuffle_ex`) | One `np.random.permutation(n_train)` draw is used per augmented view. It is deterministic only conditional on seed, NumPy behavior, and task train-count; it is not a universal fixed order. |
| TTT data | embedded `arc_solver.py`, lines 724–741 | `puzzle_ds.augment(n=16, shfl_keys=True, seed=1)` makes 8 D4 × 16 colour samples = 128 TTT records before length cutting. |
| Inference cells | embedded `arc_solver.py`, lines 759–815 | `eval_ds.augment(n=2, seed=2)` makes 8 D4 × 2 colour samples = 16 subkeys. Sorted subkeys are arranged into four physical batches and each batch calls `inference_turbo_dfs`. |
| DFS physical batch | embedded `arc_solver.py`, lines 769–795 and 810–815 | Batches are `[0,1,4,5]`, `[2,3,6,7]`, `[8,9,12,13]`, `[10,11,14,15]` of sorted subkeys: four batches of four. |
| Candidate inverse | embedded `arc_solver.py`, lines 817–831 | Each decoded array is inverted with `puzzle_ds_multi.invert_mod(..., inv_perm=True)`. |
| Candidate rescoring | embedded `arc_solver.py`, lines 833–852 | Each unique candidate grid is transformed with `aug_dataset.augment(seed=hash(bk) % 1024**2)` using default `n=1`: 8 D4 × 1 colour draw = 8 score views, evaluated as two score batches of four. |
| Final selector | embedded `arc_decoder.py`, lines 294–320 and 345–346 | `score_kgmon` groups identical inverse-transformed grids; it uses support count minus mean augmentation score. |

## Mirror provenance

At the same pinned commit, `ARC-AGI1/README.md` says this folder's code is the same as the winning Kaggle notebook and names `002_ivan_arc1.ipynb` as its evaluation code; it limits the stated difference to use of public ARC-AGI 2024 evaluation data for test-time fine-tuning. The direct Kaggle page identifies version `276863151` through its oEmbed metadata, but anonymous source-export endpoints were unavailable during this audit. Therefore the fixed Git mirror is the controlling code source and the direct Kaggle source is recorded as discovered but not independently byte-compared.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    challenge = json.loads(args.challenge.read_text(encoding="utf-8"))
    if not isinstance(challenge, dict) or not challenge:
        raise ValueError("challenge must be a nonempty task mapping")
    for task_id, task in challenge.items():
        if any("output" in test for test in task["test"]):
            raise ValueError(f"challenge unexpectedly includes a test output for {task_id}")
    if not all("train" in task and "test" in task and task["test"] for task in challenge.values()):
        raise ValueError("challenge lacks required train/test structure")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    candidates, aug4, aug8, aug16, aug24 = build_candidates()
    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    representative_tasks = select_representative_tasks(challenge)

    per_task_prompt: dict[str, dict[str, dict[str, Any]]] = {}
    for task_id in representative_tasks:
        task = challenge[task_id]
        per_task_prompt[task_id] = {}
        for candidate in candidates:
            prompt = public_formatter(task, candidate)
            transformed_test = transform_example(task["test"][0], candidate["geometry"], candidate["color_mapping"])["input"]
            per_task_prompt[task_id][candidate["candidate_id"]] = {
                "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                "prompt_bytes": len(prompt.encode("utf-8")),
                "prompt_token_length": None,
                "prompt_token_length_status": "NOT_MEASURED; Qwen tokenizer/model deliberately not loaded in this CPU-only source audit",
                "transformed_test_input_sha256": sha256_json(transformed_test),
                "colour_mapping_sha256": sha256_bytes(candidate["color_mapping"].encode("ascii")),
                "demo_order": candidate["demo_order"],
            }

    redundancy_rows: list[dict[str, Any]] = []
    signatures: dict[str, str] = {}
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        signature_payload = [per_task_prompt[task_id][candidate_id]["prompt_sha256"] for task_id in representative_tasks]
        signature = sha256_json(signature_payload)
        duplicates = []
        for task_id in representative_tasks:
            this_hash = per_task_prompt[task_id][candidate_id]["prompt_sha256"]
            same = [other["candidate_id"] for other in candidates if other["candidate_id"] != candidate_id and per_task_prompt[task_id][other["candidate_id"]]["prompt_sha256"] == this_hash]
            if same:
                duplicates.append(task_id)
        duplicate_of = next((other for other, other_signature in signatures.items() if other_signature == signature), "")
        signatures[candidate_id] = signature
        candidate["representative_prompt_sha256_by_task"] = {task_id: per_task_prompt[task_id][candidate_id]["prompt_sha256"] for task_id in representative_tasks}
        candidate["representative_prompt_identity_status"] = "TASK_DEPENDENT; exact values frozen for deterministic representative task screen"
        redundancy_rows.append({
            "candidate_id": candidate_id,
            "representative_prompt_signature_sha256": signature,
            "duplicate_of_across_all_representatives": duplicate_of,
            "duplicate_representative_task_count": len(duplicates),
            "duplicate_representative_task_ids": ";".join(duplicates),
            "prompt_bytes_min": min(per_task_prompt[task_id][candidate_id]["prompt_bytes"] for task_id in representative_tasks),
            "prompt_bytes_max": max(per_task_prompt[task_id][candidate_id]["prompt_bytes"] for task_id in representative_tasks),
            "prompt_token_length_status": "NOT_MEASURED_NO_TOKENIZER_LOADED",
        })

    counts = [
        {"quantity": "D4 geometry transforms", "value": 8, "stage": "TTT; DFS generation; rescoring", "deterministic": "transform family deterministic", "source": "ArcDataset.augment lines 251-258"},
        {"quantity": "TTT colour samples per geometry", "value": 16, "stage": "TTT", "deterministic": "seed=1, NumPy global RNG; package version unpinned", "source": "arc_solver lines 724-727"},
        {"quantity": "TTT augmented records before cut_to_len", "value": 128, "stage": "TTT", "deterministic": "8 geometry x 16 colour draws; order shuffles are task-dependent", "source": "ArcDataset.augment + arc_solver lines 724-727"},
        {"quantity": "TTT train-pair order draws", "value": 128, "stage": "TTT", "deterministic": "one shuffle_ex draw per record; possible repeats when n_train is small", "source": "ArcDataset.shuffle_ex lines 237-249"},
        {"quantity": "DFS colour samples per geometry", "value": 2, "stage": "DFS generation", "deterministic": "seed=2, NumPy global RNG; package version unpinned", "source": "arc_solver lines 759-762"},
        {"quantity": "DFS generation augmentations", "value": 16, "stage": "DFS generation", "deterministic": "8 geometry x 2 colour draws; train order is task-dependent", "source": "arc_solver lines 759-815"},
        {"quantity": "DFS physical batches", "value": 4, "stage": "DFS generation", "deterministic": "four source-coded batches of width four", "source": "arc_solver lines 769-795"},
        {"quantity": "rescore colour samples per geometry", "value": 1, "stage": "candidate rescoring", "deterministic": "seed uses process-salted hash(bk) unless PYTHONHASHSEED is controlled", "source": "arc_solver lines 833-852"},
        {"quantity": "candidate rescoring augmentations", "value": 8, "stage": "candidate rescoring", "deterministic": "8 geometry x 1 colour draw; task/candidate-specific demo order", "source": "arc_solver lines 833-852"},
    ]

    contract = {
        "experiment_id": "AUGMENTATION_CANDIDATE_SCREENING_V1",
        "source_head": "ce617e27fcea1061cfa946b661dba2f5d41afe16",
        "mode": "CPU_ONLY_SOURCE_AND_REPRESENTATION_AUDIT",
        "forbidden": ["GPU", "Qwen model", "tokenizer/model load", "TTT", "DFS", "Gold", "test outputs", "R128", "R256", "Retention30", "decoder-policy changes"],
        "representative_task_selection": {
            "source": "ARC evaluation challenge file only",
            "salt": SCREEN_SALT,
            "rule": "sort task IDs by sha256(salt + ':' + task_id), take first 8",
            "task_ids": representative_tasks,
            "challenge_sha256": sha256_bytes(args.challenge.read_bytes()),
        },
        "public_classification": "PUBLIC_DFS_AUG16_CONFIRMED",
        "research_classification": "RESEARCH_AUG16_CANDIDATES_FROZEN",
    }
    grammar = {
        "allowed": {
            "geometry": list(INVERSE_GEOMETRY),
            "colour": "Any explicit 10-colour bijection represented as old-colour -> new-colour digits; inverse stored with candidate.",
            "demo_order": ["canonical", "reverse", "rotate_left_1"],
        },
        "invariants": ["target-independent", "Gold-independent", "label/solution preserving", "exactly invertible geometry and colour mapping", "test input remains final query", "deterministic frozen descriptors"],
        "excluded": ["crop", "resize", "translation", "noise", "non-bijective recolouring", "topology changes", "lossy transforms", "synthetic perturbation", "Gold-dependent selection"],
    }
    subsets = {
        "nesting": "set inclusion; order is deliberately frozen but not claimed to be a public canonical ordering",
        "AUG4": aug4,
        "AUG8": aug8,
        "AUG16": aug16,
        "AUG24": aug24,
        "public_vs_project": "All four named research subsets are PROJECT_DEFINED.  Public NVARC has 16 configured inference cells but its RNG-constructed and task-dependent recipe is not this project pool.",
    }
    future = {
        "status": "PREREGISTERED_SCREENING_DESIGN_ONLY; no accuracy or Gold results exist in this artifact",
        "per_augmentation_metrics": ["total_FIX_cells", "unique_FIX_outputs", "overlap_matrix", "Jaccard_overlap", "incremental_union_contribution", "first_Gold_node", "first_Gold_time", "R512/R1024/R2048/R4096_hit", "search_runtime", "physical_batch_occupancy", "marginal_FIX_per_GPU_minute"],
        "selection_rule": "Select best K only after development scoring using a preregistered deterministic objective and evaluate the selected subset on held-out outputs, grouped CV, or a separate validation cohort. Never report same-set selection performance as unbiased generalization.",
    }
    decision = {
        "public": {
            "classification": "PUBLIC_DFS_AUG16_CONFIRMED",
            "distinct_dfs_cells_under_one_task_adapter": True,
            "fixed_canonical_aug16": False,
            "reason": "The source constructs 16 DFS cells but colour/demonstration variants are RNG-generated; NumPy is not pinned and train-pair order depends on task train count.",
        },
        "research": {
            "classification": "RESEARCH_AUG16_CANDIDATES_FROZEN",
            "pool_size": len(candidates),
            "aug16_is_public_nvarc": False,
            "rationale": "AUG16 extends D4 with balanced source-supported colour, demonstration-order, and combined representation changes without target/Gold selection.",
        },
        "next_gpu_experiment": "DYNAMIC_B16_ONLY_AFTER_EXPLICIT_GPU_AUTHORIZATION; B16 is the relevant next width because the public source configures 16 DFS cells and PROJECT_RESEARCH_AUG16 is now frozen. This audit launches no GPU work.",
    }

    write_json(output / "CONTRACT.json", contract)
    write_json(output / "PUBLIC_SOURCE_PROVENANCE.json", build_public_source_provenance())
    with (output / "AUGMENTATION_COUNTS.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(counts[0]))
        writer.writeheader(); writer.writerows(counts)
    write_json(output / "PUBLIC_INFERENCE_RECIPE.json", public_inference_recipe())
    write_json(output / "ALLOWED_AUGMENTATION_GRAMMAR.json", grammar)
    write_json(output / "CANDIDATE_POOL.json", {"pool_size": len(candidates), "candidates": candidates, "representative_task_prompt_evidence": per_task_prompt})
    with (output / "CANDIDATE_REDUNDANCY.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(redundancy_rows[0]))
        writer.writeheader(); writer.writerows(redundancy_rows)
    write_json(output / "AUG4_IDS.json", {"subset": "PROJECT_RESEARCH_AUG4", "candidate_ids": aug4})
    write_json(output / "AUG8_IDS.json", {"subset": "PROJECT_RESEARCH_AUG8", "candidate_ids": aug8})
    write_json(output / "AUG16_IDS.json", {"subset": "PROJECT_RESEARCH_AUG16", "candidate_ids": aug16})
    write_json(output / "AUG24_OR_32_IDS.json", {"subset": "PROJECT_RESEARCH_AUG24", "candidate_ids": aug24})
    write_json(output / "NESTED_SUBSET_CONTRACT.json", subsets)
    write_json(output / "FUTURE_SCREENING_PROTOCOL.json", future)
    write_json(output / "DECISION.json", decision)
    (output / "NOTEBOOK_CODE_REFERENCES.md").write_text(notebook_references(), encoding="utf-8", newline="\n")
    report = """# Augmentation candidate screening V1

## Result

- **Public classification:** `PUBLIC_DFS_AUG16_CONFIRMED`.
- **Research classification:** `RESEARCH_AUG16_CANDIDATES_FROZEN`.
- The public source configures 16 inference/DFS subkeys per output: eight D4 geometries times two seeded full-colour permutations. They are decoded in four physical batches of four. This is distinct from the public TTT expansion (eight geometries times sixteen seeded colour samples = 128 records) and from rescoring (eight geometries times one colour sample = eight views).
- Public `16` is not a portable canonical set of 16 static recipes: source code does not pin NumPy, and every augmented view's train-pair ordering is drawn according to the task's number of demonstrations. Candidate rescoring also derives its seed from `hash(bk)`, whose process salt is not pinned in source.

## Research pool

The frozen 24-candidate pool deliberately keeps `PROJECT_RESEARCH_AUG4 ⊂ AUG8 ⊂ AUG16 ⊂ AUG24`. Its AUG16 adds eight target-blind, exactly invertible representation conditions to D4; it is **not** the public NVARC AUG16.

The redundancy screen used eight challenge-only representative tasks chosen by a SHA-256 rule. It verifies serialized prompt and transformed-test-input identities. Exact Qwen token lengths are marked unavailable because no tokenizer/model was loaded; byte lengths are recorded as the CPU-only representation measure. Per-task duplicate variants caused by a one-example train set are reported but retained when the same candidate differs on other tasks.

## Next step

No GPU work occurred. If a new GPU experiment is explicitly authorized, the appropriate next width is `DYNAMIC_B16`, using the frozen **project** AUG16 definition and an independently held-out screening protocol.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")

    hashes: dict[str, str] = {}
    for path in sorted(output.iterdir()):
        if path.name != "HASHES.json" and path.is_file():
            hashes[path.name] = sha256_bytes(path.read_bytes())
    write_json(output / "HASHES.json", {"algorithm": "sha256", "files": hashes})


if __name__ == "__main__":
    main()
