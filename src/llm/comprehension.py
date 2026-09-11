"""Registry-derived capability comprehension benchmark."""
from __future__ import annotations

from collections import Counter
from typing import Any

from .catalog import build_capability_catalog


DISTINCTIONS = (
    ("OBJ_MOVE_V1", "GRID_TRANSLATE_FOREGROUND_V0", "object translation vs whole-foreground translation"),
    ("REL_CONTAINS_BBOX_V1", "REL_OVERLAP_V1", "bbox containment vs mask overlap"),
    ("REL_TOUCHING_4_V1", "REL_OVERLAP_V1", "4-neighbour touching vs diagonal/non-overlap"),
    ("GRID_FLIP_HORIZONTAL_V0", "PAT_COMPLETE_MIRROR_HORIZONTAL_V1", "whole-grid flip vs symmetry completion"),
    ("PAT_FIND_ROW_PERIOD_V1", "PAT_REPAIR_PERIOD_VIOLATION_V1", "period finding vs repair"),
    ("OBJ_CROP_V1", "GEN_RECTANGLE_V1", "crop/extraction vs canvas generation"),
    ("REG_FIND_ENCLOSED_REGIONS_V1", "REG_INTERIOR_MASK_V1", "enclosed region vs interior mask"),
    ("COUNT_OBJECTS_V1", "COUNT_COLOR_CELLS_V1", "object count vs color-cell count"),
    ("GRAPH_TRACE_PATH_V1", "GRAPH_SHORTEST_PATH_V1", "ambiguous endpoint traversal vs explicit shortest path"),
    ("GEN_N_CELLS_ROW_V1", "GEN_RECTANGLE_V1", "bounded row generation vs rectangle generation"),
)


def build_comprehension_benchmark() -> dict[str, Any]:
    catalog = build_capability_catalog()
    by_id = {entry["primitive_id"]: entry for entry in catalog["capabilities"]}
    questions = []
    for answer, contrast, topic in DISTINCTIONS:
        if answer not in by_id or contrast not in by_id:
            continue
        for variant in range(4):
            entry = by_id[answer]
            questions.append(
                {
                    "question_id": f"{answer.lower()}_{variant + 1}",
                    "topic": topic,
                    "question": f"Which canonical primitive matches this registry semantics: {entry['semantics']}",
                    "options": [answer, contrast],
                    "answer": answer,
                    "ground_truth_source": "frozen capability registry",
                }
            )
    return {"benchmark_version": "capability_comprehension_v1", "question_count": len(questions), "questions": questions}


def score_comprehension(benchmark: dict[str, Any], answers: dict[str, str]) -> dict[str, Any]:
    questions = benchmark["questions"]
    correct = [question for question in questions if answers.get(question["question_id"]) == question["answer"]]
    errors = Counter(question["topic"] for question in questions if answers.get(question["question_id"]) not in (None, question["answer"]))
    unanswered = sum(question["question_id"] not in answers for question in questions)
    return {
        "accuracy": len(correct) / len(questions) if questions else None,
        "correct": len(correct),
        "total": len(questions),
        "unanswered": unanswered,
        "error_categories": dict(errors),
    }
