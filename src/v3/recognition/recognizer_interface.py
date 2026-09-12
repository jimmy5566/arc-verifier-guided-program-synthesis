"""The sole V3 LLM-facing interface; it cannot return concrete parameters."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from arc.task import ARCTask
from v3.evidence.cross_pair import CrossPairEvidence
from v3.evidence.extractor import EvidenceBundle
from v3.schema.rule_skeleton import OperationId, RuleSkeleton


class RuleRecognizer(Protocol):
    def recognize(self, task: ARCTask, evidence: EvidenceBundle, cross_pair: CrossPairEvidence, *, top_k: int = 1) -> Sequence[RuleSkeleton]:
        """Return zero or more parameter-free skeleton hypotheses only."""


class PredefinedRecognizer:
    """Deterministic test seam; production adapters implement RuleRecognizer."""
    def __init__(self, skeletons: Sequence[RuleSkeleton]) -> None:
        self._skeletons = tuple(skeletons)

    def recognize(self, task: ARCTask, evidence: EvidenceBundle, cross_pair: CrossPairEvidence, *, top_k: int = 1) -> Sequence[RuleSkeleton]:
        del task, evidence, cross_pair
        return self._skeletons[:top_k]


def _object_payload(item: Any) -> dict[str, Any]:
    return {"color": item.color, "bbox": list(item.bbox), "area": item.area}


def _bounded(values: Sequence[Any], *, limit: int) -> dict[str, Any]:
    ordered = sorted(values, key=repr)
    return {"items": list(ordered[:limit]), "total": len(ordered), "truncated": len(ordered) > limit}


def _relation_summary(graph: Any) -> dict[str, Any]:
    """Keep relation *facts* while avoiding a redundant full graph dump.

    Exact pixel grids are already sent.  Object positions are retained below;
    this summary adds deterministic relation labels and correspondence counts
    without serialising every graph node ID and every pairwise edge.
    """
    if not isinstance(graph, dict):
        return {"available": False}
    labels: set[str] = set()
    edge_count = correspondence_count = 0
    for key in ("input_graph", "output_graph"):
        value = graph.get(key, {})
        edges = value.get("edges", []) if isinstance(value, dict) else []
        edge_count += len(edges)
        for edge in edges:
            if len(edge) > 2 and isinstance(edge[2], list):
                labels.update(str(label) for label in edge[2])
    correspondences = graph.get("candidate_correspondences", [])
    correspondence_count = len(correspondences) if isinstance(correspondences, list) else 0
    if isinstance(correspondences, list):
        for item in correspondences:
            if len(item) > 2 and isinstance(item[2], list):
                labels.update(str(label) for label in item[2])
    return {
        "a": True,
        "e": edge_count,
        "c": correspondence_count,
        "l": sorted(labels)[:5],
    }


def _object_summary(objects: Sequence[Any]) -> dict[str, Any]:
    """Compact object evidence; full object pixels remain in the raw grid."""
    grouped: dict[int, list[Any]] = {}
    for item in objects:
        grouped.setdefault(int(item.color), []).append(item)
    by_color = []
    for color, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: (item.area, item.bbox))
        # tuple format: colour, count, min-area, max-area, representative bbox
        by_color.append([color, len(ordered), ordered[0].area, ordered[-1].area, list(ordered[0].bbox)])
    return {"n": len(objects), "by_color": by_color[:3], "truncated": len(by_color) > 3}


def _symmetry_summary(grid: Any) -> dict[str, bool]:
    import numpy as np
    values = np.asarray(grid)
    return {
        "h": bool(np.array_equal(values, np.fliplr(values))),
        "v": bool(np.array_equal(values, np.flipud(values))),
        "d": bool(values.ndim == 2 and values.shape[0] == values.shape[1] and np.array_equal(values, values.T)),
    }


def _changed_summary(cells: Sequence[tuple[int, int]]) -> dict[str, Any]:
    sample = [list(cell) for cell in cells[:2]]
    if not cells:
        return [0, [], None, False]
    rows, columns = zip(*cells)
    return [len(cells), sample, [min(rows), min(columns), max(rows), max(columns)], len(cells) > len(sample)]


def _facts(evidence: EvidenceBundle, cross_pair: CrossPairEvidence) -> dict[str, Any]:
    return {
        "pair_facts": [
            {
                "i": _object_summary(pair.input_objects),
                "o": _object_summary(pair.output_objects),
                "x": _changed_summary(pair.changed_cells),
                "r": [
                    sorted(list(pair.parameter_candidates[key]), key=repr)[:2]
                    for key in ("DIRECTION", "DISTANCE", "STEP", "COUNT", "TERMINATION")
                ],
                "g": _relation_summary(pair.a3_relation_graph),
                "s": [_symmetry_summary(pair.input_grid), _symmetry_summary(pair.output_grid)],
            }
            for pair in evidence.pairs
        ],
        "cross_pair": {
            "pair_count": cross_pair.pair_count,
            "common_candidate_counts": {
                key: len(cross_pair.candidates.get(key, ()))
                for key in ("DIRECTION", "DISTANCE", "STEP", "COUNT", "TERMINATION")
            },
            "all_intersections_nonempty": bool(cross_pair.invariants["all_pair_candidate_intersection_nonempty"]),
        },
    }


def _json_default(value: Any) -> Any:
    """Convert NumPy scalar facts without changing their semantic value.

    Evidence is deliberately built directly from grids, where ``argwhere`` and
    relation-graph helpers may retain NumPy integer scalars.  The recognizer
    prompt is an interchange boundary, so it must normalize those scalars
    rather than fail a worker after its model has loaded.
    """
    item = getattr(value, "item", None)
    if callable(item):
        normalized = item()
        if normalized is not value:
            return normalized
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def recognition_prompt(task: ARCTask, evidence: EvidenceBundle, cross_pair: CrossPairEvidence, *, top_k: int) -> str:
    raw = {"train_pairs": [{"input": item.input.values.tolist(), "output": item.output.values.tolist()} for item in task.train]}
    return json.dumps({
        "train_grids": raw,
        "deterministic_evidence": _facts(evidence, cross_pair),
        "slot_contract": "SELECT:$SELECTOR;COPY/MOVE:$DIRECTION,$DISTANCE;REPEAT:$DIRECTION,$STEP,$COUNT,$TERMINATION;RECOLOR/FILL:$TARGET_COLOR;RELATIONAL_COPY:$REFERENCE_COLOR,$DIRECTION,$DISTANCE;ROTATE/REFLECT/CROP:[]",
        "instruction": (
            f"Infer at most {top_k} distinct general rule skeletons from TRAIN only. "
            "Return exactly one JSON object with one key named hypotheses. Each hypothesis has exactly family, operations, and required_slots. "
            "family is uppercase. operations is a non-empty ordered list from slot_contract. required_slots is its sorted slot union. "
            "Do not copy any prompt text or contract. Do not emit literal parameter values, grids, code, rationale, or markdown."
        ),
    }, separators=(",", ":"), default=_json_default)


def parse_hypotheses(raw: str, *, limit: int) -> tuple[tuple[RuleSkeleton, ...], str]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return (), "SCHEMA_FAILURE:no JSON"
    try:
        value = json.loads(raw[start:end + 1])
        hypotheses = value["hypotheses"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return (), "SCHEMA_FAILURE:invalid JSON envelope"
    if not isinstance(hypotheses, list) or not 1 <= len(hypotheses) <= limit:
        return (), "SCHEMA_FAILURE:hypothesis count"
    parsed: list[RuleSkeleton] = []
    for item in hypotheses:
        if not isinstance(item, dict) or set(item) != {"family", "operations", "required_slots"} or not isinstance(item["family"], str) or not item["family"]:
            return (), "SCHEMA_FAILURE:hypothesis fields"
        if not isinstance(item["operations"], list) or not all(isinstance(name, str) for name in item["operations"]):
            return (), "SCHEMA_FAILURE:operations"
        try:
            skeleton = RuleSkeleton.from_operations(item["family"], tuple(OperationId(name) for name in item["operations"]))
        except (TypeError, ValueError):
            return (), "SCHEMA_FAILURE:unsupported operation"
        if not isinstance(item["required_slots"], list) or not all(isinstance(slot, str) for slot in item["required_slots"]):
            return (), "SCHEMA_FAILURE:required slots"
        if sorted(slot.value for slot in skeleton.required_slots) != sorted(item["required_slots"]):
            return (), "SCHEMA_FAILURE:wrong typed slots"
        if skeleton in parsed:
            return (), "SCHEMA_FAILURE:duplicate skeleton"
        parsed.append(skeleton)
    return tuple(parsed), "SUCCESS"


@dataclass
class QwenRuleRecognizer:
    """Production, offline adapter. It is deliberately upstream-only."""
    provider: Any
    generation_config: Any

    def recognize(self, task: ARCTask, evidence: EvidenceBundle, cross_pair: CrossPairEvidence, *, top_k: int = 1) -> Sequence[RuleSkeleton]:
        generated = self.provider.generate_text(recognition_prompt(task, evidence, cross_pair, top_k=top_k), self.generation_config)
        skeletons, status = parse_hypotheses(generated.text, limit=top_k)
        if status != "SUCCESS":
            return ()
        return skeletons
