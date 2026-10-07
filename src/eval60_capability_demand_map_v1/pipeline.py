"""Build the frozen, target-blind Eval60 capability-demand map.

The classification table in this module was authored from training pairs and
test inputs only.  Historical outcomes are opened only after the taxonomy
artifacts have been written, hashed, and independently verified.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VERSION = "EVAL60_CAPABILITY_DEMAND_MAP_V1"
OFFICIAL_COMMIT = "6726759ffd41b9087095796e1e11acd865a56066"
OFFICIAL_REPOSITORY = "https://github.com/arcprize/ARC-AGI-2.git"
PRIMARY_FAMILIES = {
    "GEOMETRIC_TRANSFORM", "OBJECT_SELECTION", "SPATIAL_RELATION", "MOTION",
    "COLOR_MAPPING", "MASK_SET", "CONSTRUCTION", "COUNTING_NUMERIC",
    "PATTERN_PROGRESSION", "CONDITIONAL_CONTROL", "MULTI_STAGE_COMPOSITION",
    "OTHER_OR_UNCERTAIN",
}
GEOMETRIES = {"SAME_SIZE", "CROP", "EXPAND", "RESHAPE", "CONSTRUCT_NEW", "VARIABLE_OR_UNKNOWN"}
COVERAGES = {"FULLY_COVERED", "PARTIALLY_COVERED", "OUTSIDE_CURRENT_ONTOLOGY", "UNCERTAIN"}
DEPTHS = {"1", "2", "3_PLUS", "UNKNOWN"}
CONFIDENCES = {"HIGH", "MEDIUM", "LOW"}
AXES = {"color", "position", "grid_size", "object_count", "object_size", "displacement", "orientation", "distractors", "other"}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(value) + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ann(
    primary: str,
    caps: tuple[str, ...],
    depth: str,
    geometry: str,
    coverage: str,
    confidence: str,
    evidence: str,
    alternatives: str = "",
    axes: tuple[str, ...] = ("color", "position", "grid_size"),
    secondary: tuple[str, ...] = (),
    parameter_inference: bool = True,
    gap: str = "",
) -> dict[str, Any]:
    return {
        "primary_family": primary,
        "secondary_families": list(secondary),
        "required_capabilities": list(caps),
        "estimated_composition_depth": depth,
        "output_geometry": geometry,
        "ontology_coverage": coverage,
        "classification_confidence": confidence,
        "classification_evidence": evidence,
        "ambiguous_alternative_interpretations": alternatives,
        "parameter_generalization_axes": list(axes),
        "requires_parameter_inference": parameter_inference,
        "ontology_gap": gap,
    }


# Frozen before any historical outcome was opened by this pipeline.  Every
# concept below is an exact member of Foundation Capability Bank V3's 83-item
# ontology (V3 intentionally aliases the V2 ontology vocabulary).
ANNOTATIONS: dict[str, dict[str, Any]] = {
    "5dbc8537": ann("OTHER_OR_UNCERTAIN", ("separators/subgrids", "crop", "extract", "position"), "3_PLUS", "CONSTRUCT_NEW", "OUTSIDE_CURRENT_ONTOLOGY", "LOW", "Demonstrations extract a narrow structured panel from a multicolour field using repeated separator-like layout cues.", "The selected panel may encode a row/column projection rather than a literal crop.", ("color", "position", "grid_size", "object_count", "other"), ("OBJECT_SELECTION", "CONSTRUCTION"), gap="hierarchical panel decoding / latent row-column projection"),
    "97d7923e": ann("OBJECT_SELECTION", ("lines", "height", "nth/order", "compare quantities", "recolor"), "2", "SAME_SIZE", "PARTIALLY_COVERED", "LOW", "Vertical bars are compared by height/order and one bar's interior colour assignment changes while positions remain fixed.", "The trigger could be endpoint-colour matching rather than rank.", ("color", "position", "object_count", "object_size", "other"), ("COUNTING_NUMERIC",)),
    "3a25b0d8": ann("OBJECT_SELECTION", ("connected components", "unique", "object vs color grouping", "extract", "crop"), "2", "CROP", "FULLY_COVERED", "HIGH", "Among separated objects, the multicolour/structurally unique object is selected and tightly extracted.", axes=("color", "position", "grid_size", "object_size", "distractors"), secondary=("CONSTRUCTION",)),
    "cb2d8a2c": ann("PATTERN_PROGRESSION", ("lines", "connected", "aligned", "propagation", "until alignment"), "3_PLUS", "SAME_SIZE", "PARTIALLY_COVERED", "MEDIUM", "Green paths extend orthogonally from starts through blue waypoints while red segments act as constraints.", "Could be ordered waypoint connection rather than state propagation.", ("color", "position", "grid_size", "object_count", "displacement"), ("SPATIAL_RELATION", "MOTION")),
    "b99e7126": ann("PATTERN_PROGRESSION", ("periodic pattern", "repeated motifs", "complete missing structure"), "2", "SAME_SIZE", "FULLY_COVERED", "HIGH", "A locally corrupted patch in a regular two-dimensional tiling is repaired from surrounding periodic context.", axes=("color", "position", "grid_size", "distractors"), secondary=("CONSTRUCTION",)),
    "de809cff": ann("MULTI_STAGE_COMPOSITION", ("mask", "color mapping", "copy reference color", "inside/contains", "fill"), "3_PLUS", "SAME_SIZE", "PARTIALLY_COVERED", "MEDIUM", "Sparse coloured reference cells determine colours inserted into holes within large coloured regions.", "Reference-to-hole correspondence may be based on relative coordinates or component identity.", ("color", "position", "grid_size", "object_count", "distractors"), ("COLOR_MAPPING", "MASK_SET", "CONSTRUCTION")),
    "7b3084d4": ann("MULTI_STAGE_COMPOSITION", ("connected components", "orientation", "position", "crop", "copy", "overlay"), "3_PLUS", "CONSTRUCT_NEW", "PARTIALLY_COVERED", "MEDIUM", "Corner-distributed marked objects are cropped and assembled into a compact mosaic using their marker orientations.", "The marker may specify quadrant placement rather than rotation.", ("color", "position", "grid_size", "object_count", "object_size", "orientation"), ("GEOMETRIC_TRANSFORM", "CONSTRUCTION"), gap="marker-directed canonical packing"),
    "dfadab01": ann("CONSTRUCTION", ("frames", "copy", "complete missing structure", "position", "color mapping"), "3_PLUS", "SAME_SIZE", "PARTIALLY_COVERED", "MEDIUM", "Sparse anchors on an implicit lattice are expanded into repeated coloured frame motifs.", "Anchor colours may jointly encode both motif colour and placement.", ("color", "position", "grid_size", "object_count"), ("PATTERN_PROGRESSION", "COLOR_MAPPING")),
    "58490d8a": ann("COUNTING_NUMERIC", ("connected components", "count components", "color", "encode quantity geometrically", "crop"), "3_PLUS", "CONSTRUCT_NEW", "FULLY_COVERED", "HIGH", "A black legend orders colours; the output encodes the count and positions of matching coloured objects as marks by legend row.", axes=("color", "position", "grid_size", "object_count", "distractors"), secondary=("CONSTRUCTION", "COLOR_MAPPING")),
    "142ca369": ann("CONSTRUCTION", ("lines", "extend line/ray", "inferred displacement", "same color", "aligned"), "2", "SAME_SIZE", "FULLY_COVERED", "HIGH", "Short coloured segments and boundary seeds are extended diagonally along their inferred direction until aligned endpoints/bounds.", axes=("color", "position", "grid_size", "displacement", "orientation"), secondary=("MOTION", "SPATIAL_RELATION")),
    "446ef5d2": ann("OBJECT_SELECTION", ("connected components", "largest", "area", "extract", "crop"), "2", "CROP", "FULLY_COVERED", "HIGH", "The largest connected composite object is selected from distractors and tightly cropped.", axes=("color", "position", "grid_size", "object_count", "object_size", "distractors"), secondary=("CONSTRUCTION",)),
    "b5ca7ac4": ann("MULTI_STAGE_COMPOSITION", ("connected components", "uniqueness", "color", "translate", "aligned"), "3_PLUS", "SAME_SIZE", "PARTIALLY_COVERED", "MEDIUM", "Repeated framed tiles are deduplicated by inner colour/attribute and repacked into an aligned arrangement.", "Selection may prefer a particular outer frame rather than simply remove duplicate inner colours.", ("color", "position", "grid_size", "object_count", "distractors"), ("OBJECT_SELECTION", "MOTION"), gap="attribute-keyed deduplication and packing"),
    "80a900e0": ann("PATTERN_PROGRESSION", ("frames", "color", "extend line/ray", "propagation", "orientation"), "3_PLUS", "SAME_SIZE", "PARTIALLY_COVERED", "MEDIUM", "Colours on a compact ring seed diagonal rays that propagate outward in the corresponding orientations.", axes=("color", "position", "grid_size", "object_count", "orientation"), secondary=("CONSTRUCTION",)),
    "16de56c4": ann("PATTERN_PROGRESSION", ("same row", "same column", "propagation", "repeat", "color"), "2", "SAME_SIZE", "FULLY_COVERED", "MEDIUM", "Repeated coloured markers determine whether rows or columns are completed by regularly spaced colour marks.", axes=("color", "position", "grid_size", "object_count", "orientation"), secondary=("SPATIAL_RELATION",)),
    "71e489b6": ann("CONSTRUCTION", ("holes/enclosures", "border", "complete missing structure", "position"), "2", "SAME_SIZE", "FULLY_COVERED", "HIGH", "Black defects inside or adjacent to blue regions are enclosed with orange rectangular borders.", axes=("color", "position", "grid_size", "object_count", "object_size"), secondary=("PERCEPTION_SEGMENTATION",)),
    "2d0172a1": ann("OBJECT_SELECTION", ("lines", "frames", "inside/contains", "smallest", "extract", "crop"), "3_PLUS", "CROP", "FULLY_COVERED", "MEDIUM", "The smallest coherent enclosed spiral/loop motif is located inside a larger line field and tightly extracted.", axes=("color", "position", "grid_size", "object_size", "distractors"), secondary=("CONSTRUCTION",)),
    "981571dc": ann("CONSTRUCTION", ("repeated motifs", "symmetry completion", "complete missing structure", "fill"), "2", "SAME_SIZE", "FULLY_COVERED", "HIGH", "Black rectangular omissions in a dense symmetric/repeated texture are filled from the surrounding pattern.", axes=("color", "position", "grid_size", "object_count", "distractors"), secondary=("PATTERN_PROGRESSION",)),
    "13e47133": ann("PATTERN_PROGRESSION", ("separators/subgrids", "inside/contains", "fill", "periodic pattern", "color mapping"), "3_PLUS", "SAME_SIZE", "FULLY_COVERED", "MEDIUM", "Divider lines partition regions; seed colours expand into nested periodic rectangular fills within each region.", axes=("color", "position", "grid_size", "object_count", "object_size"), secondary=("CONSTRUCTION", "COLOR_MAPPING")),
    "e3721c99": ann("COLOR_MAPPING", ("connected components", "orientation", "same color", "color mapping", "recolor"), "3_PLUS", "SAME_SIZE", "FULLY_COVERED", "HIGH", "Grey objects are shape-matched, allowing rotation/orientation variation, to coloured legend examples and then recoloured.", axes=("color", "position", "grid_size", "object_count", "object_size", "orientation"), secondary=("OBJECT_SELECTION", "GEOMETRIC_TRANSFORM")),
    "cbebaa4b": ann("MULTI_STAGE_COMPOSITION", ("connected components", "adjacent/touching", "translate", "until touching", "overlay"), "3_PLUS", "SAME_SIZE", "FULLY_COVERED", "HIGH", "Open marked components are translated so their red connection points attach around the central yellow hub.", axes=("color", "position", "grid_size", "object_count", "displacement", "orientation"), secondary=("MOTION", "CONSTRUCTION")),
    "45a5af55": ann("CONSTRUCTION", ("lines", "nth/order", "periodic pattern", "complete missing structure"), "3_PLUS", "EXPAND", "PARTIALLY_COVERED", "HIGH", "The ordered sequence of horizontal stripe colours is expanded into a square inward spiral/ring construction.", axes=("color", "grid_size", "object_count", "object_size", "other"), secondary=("PATTERN_PROGRESSION",), gap="sequence-to-spiral rasterization"),
    "c4d067a0": ann("MULTI_STAGE_COMPOSITION", ("copy", "same row", "same column", "position", "color mapping"), "3_PLUS", "SAME_SIZE", "PARTIALLY_COVERED", "MEDIUM", "A small reference pattern is copied at locations specified by larger coloured blocks and their relative arrangement.", "The top-left marks may be a coordinate legend rather than a pattern template.", ("color", "position", "grid_size", "object_count", "displacement"), ("CONSTRUCTION", "SPATIAL_RELATION")),
    "67e490f4": ann("MULTI_STAGE_COMPOSITION", ("frames", "position", "color mapping", "copy", "crop"), "3_PLUS", "CONSTRUCT_NEW", "PARTIALLY_COVERED", "LOW", "A framed lattice/template is reconstructed as a compact output, with scattered coloured objects projected into corresponding cells.", "The outside objects may encode row/column coordinates through shape rather than position.", ("color", "position", "grid_size", "object_count", "object_size", "other"), ("SPATIAL_RELATION", "CONSTRUCTION"), gap="template projection from external spatial codes"),
    "20270e3b": ann("MASK_SET", ("separators/subgrids", "mask", "difference", "crop"), "2", "RESHAPE", "FULLY_COVERED", "MEDIUM", "Orange-marked rows/columns or panels are removed and the remaining blue/yellow structure is compacted.", axes=("color", "position", "grid_size", "object_count", "orientation"), secondary=("CONSTRUCTION",)),
    "2c181942": ann("MULTI_STAGE_COMPOSITION", ("connected components", "position", "translate", "aligned", "overlay"), "3_PLUS", "SAME_SIZE", "FULLY_COVERED", "HIGH", "Detached monochrome objects are translated into the colour-coded slots around a central multicolour scaffold.", axes=("color", "position", "grid_size", "object_count", "displacement", "orientation"), secondary=("MOTION", "CONSTRUCTION")),
    "6ffbe589": ann("COLOR_MAPPING", ("frames", "crop", "color mapping", "copy reference color", "mask"), "3_PLUS", "CROP", "FULLY_COVERED", "MEDIUM", "A large framed motif is cropped and its layers are recoloured using nearby reference swatches.", axes=("color", "position", "grid_size", "object_size", "distractors"), secondary=("CONSTRUCTION", "MASK_SET")),
    "4a21e3da": ann("CONDITIONAL_CONTROL", ("orientation", "position", "translate", "inferred displacement", "if/else on attribute"), "3_PLUS", "SAME_SIZE", "PARTIALLY_COVERED", "MEDIUM", "An edge marker selects an orientation; parts of the orange figure are translated toward the corresponding boundaries around an extended marker axis.", "The transformation may be a directional unfolding rather than conditional translation.", ("color", "position", "grid_size", "object_size", "displacement", "orientation"), ("MOTION", "GEOMETRIC_TRANSFORM")),
    "a32d8b75": ann("MULTI_STAGE_COMPOSITION", ("separators/subgrids", "copy", "overlay", "color mapping", "mask"), "3_PLUS", "SAME_SIZE", "FULLY_COVERED", "MEDIUM", "Shape templates in a side legend are copied and overlaid into matching coloured bands/regions of the main canvas.", axes=("color", "position", "grid_size", "object_count", "object_size", "orientation"), secondary=("CONSTRUCTION", "COLOR_MAPPING")),
    "aa4ec2a5": ann("CONSTRUCTION", ("connected components", "border", "holes/enclosures", "fill", "recolor"), "3_PLUS", "SAME_SIZE", "FULLY_COVERED", "MEDIUM", "Each blue component receives a red border and enclosed/internal regions are filled with a nested colour code.", axes=("color", "position", "grid_size", "object_count", "object_size"), secondary=("COLOR_MAPPING",)),
    "0934a4d8": ann("OBJECT_SELECTION", ("repeated motifs", "unique", "symmetry completion", "extract", "crop"), "3_PLUS", "CROP", "PARTIALLY_COVERED", "MEDIUM", "A small regular/symmetric motif embedded in a dense noisy field is identified as the unique coherent structure and extracted.", "The selected patch may be a local symmetry centre rather than a unique object.", ("color", "position", "grid_size", "object_size", "distractors"), ("CONSTRUCTION",)),
    "db0c5428": ann("PATTERN_PROGRESSION", ("symmetry completion", "repeat", "propagation", "complete missing structure"), "2", "SAME_SIZE", "FULLY_COVERED", "HIGH", "A compact central motif is propagated by rotational/reflection symmetry to complete a larger repeated pattern.", axes=("color", "position", "grid_size", "object_size", "orientation"), secondary=("CONSTRUCTION",)),
    "9bbf930d": ann("CONSTRUCTION", ("lines", "same row", "border", "complete missing structure"), "2", "SAME_SIZE", "PARTIALLY_COVERED", "MEDIUM", "A magenta left boundary cue is mirrored/completed at the right ends of selected horizontal stripe runs.", "The right-end marks may encode parity of run length.", ("color", "position", "grid_size", "object_count", "object_size"), ("SPATIAL_RELATION",)),
    "b6f77b65": ann("CONDITIONAL_CONTROL", ("color", "position", "nth/order", "translate", "if/else on attribute"), "3_PLUS", "SAME_SIZE", "PARTIALLY_COVERED", "MEDIUM", "The top-left colour cue selects an ordering/repositioning of the multicolour line assembly.", "The cue may choose a traversal root rather than a simple branch.", ("color", "position", "object_count", "displacement", "orientation", "other"), ("MULTI_STAGE_COMPOSITION",), gap="cue-selected graph traversal / canonicalization"),
    "31f7f899": ann("COUNTING_NUMERIC", ("lines", "height", "compare quantities", "nth/order", "aligned"), "3_PLUS", "SAME_SIZE", "PARTIALLY_COVERED", "MEDIUM", "Coloured vertical bars crossing a common axis are reordered/aligned according to their lengths.", "The ordering may use distance above versus below the axis rather than total height.", ("color", "position", "grid_size", "object_count", "object_size"), ("SPATIAL_RELATION", "MOTION")),
    "8e5c0c38": ann("CONSTRUCTION", ("connected components", "uniqueness", "symmetry completion", "difference"), "2", "SAME_SIZE", "FULLY_COVERED", "HIGH", "The single protruding/asymmetric cell on each object is removed to restore the object's regular symmetry.", axes=("color", "position", "grid_size", "object_count", "orientation"), secondary=("OBJECT_SELECTION", "MASK_SET")),
    "898e7135": ann("MULTI_STAGE_COMPOSITION", ("connected components", "crop", "fill", "copy reference color", "mask"), "3_PLUS", "CONSTRUCT_NEW", "PARTIALLY_COVERED", "MEDIUM", "The meaningful objects in a black work area are tightly reframed while scattered reference marks determine the replacement background/retained layout.", "Scattered points may specify a crop rectangle rather than only a colour.", ("color", "position", "grid_size", "object_count", "distractors"), ("CONSTRUCTION", "COLOR_MAPPING")),
    "d59b0160": ann("OBJECT_SELECTION", ("separators/subgrids", "same color", "relation-conditioned selector", "mask", "difference"), "3_PLUS", "SAME_SIZE", "PARTIALLY_COVERED", "MEDIUM", "A compact colour key selects matching black subregions; nonmatching regions are erased to background.", "Matching may depend on ordered colours or on a geometric subpattern.", ("color", "position", "grid_size", "object_count", "distractors"), ("MASK_SET",)),
    "4c3d4a41": ann("COUNTING_NUMERIC", ("lines", "height", "nth/order", "compare quantities", "translate"), "3_PLUS", "SAME_SIZE", "PARTIALLY_COVERED", "MEDIUM", "A grey histogram-like key specifies the order in which coloured vertical bars are rearranged inside the frame.", axes=("color", "position", "object_count", "object_size", "displacement"), secondary=("OBJECT_SELECTION", "MOTION")),
    "f931b4a8": ann("OTHER_OR_UNCERTAIN", ("separators/subgrids", "repeat", "overlay", "color mapping"), "UNKNOWN", "CONSTRUCT_NEW", "OUTSIDE_CURRENT_ONTOLOGY", "LOW", "Quadrant/panel motifs determine a new repeated output texture, but several demonstration-consistent panel-composition grammars remain.", "Could be Cartesian product, substitution tiling, or panel-conditioned repetition.", ("color", "grid_size", "object_count", "other"), ("PATTERN_PROGRESSION", "CONSTRUCTION"), gap="panel-to-texture grammar induction"),
    "36a08778": ann("PATTERN_PROGRESSION", ("lines", "connected", "extend line/ray", "propagation", "complete missing structure"), "2", "SAME_SIZE", "FULLY_COVERED", "HIGH", "Red path fragments and magenta endpoints are extended/connected to complete orthogonal maze-like paths.", axes=("color", "position", "grid_size", "object_count", "displacement"), secondary=("CONSTRUCTION",)),
    "53fb4810": ann("MOTION", ("lines", "same row", "same column", "until touching", "extend line/ray"), "2", "SAME_SIZE", "FULLY_COVERED", "HIGH", "A coloured ray is extended from one marked object until it meets the row/column of another marked object.", axes=("color", "position", "grid_size", "object_count", "displacement", "orientation"), secondary=("SPATIAL_RELATION", "CONSTRUCTION")),
    "332f06d7": ann("MOTION", ("connected", "position", "inferred displacement", "translate", "until boundary"), "3_PLUS", "SAME_SIZE", "PARTIALLY_COVERED", "LOW", "A marked black region and boundary marker are relocated through the connected blue/green corridor structure.", "The operation may swap endpoint regions rather than move a single object along a path.", ("color", "position", "grid_size", "object_size", "displacement", "other"), ("SPATIAL_RELATION",), gap="maze/corridor-constrained relocation"),
    "7b0280bc": ann("OTHER_OR_UNCERTAIN", ("connected", "same color", "nearest/farthest", "relation-conditioned selector", "complete missing structure"), "3_PLUS", "SAME_SIZE", "OUTSIDE_CURRENT_ONTOLOGY", "LOW", "Among black route graphs with coloured terminals, one route segment is repaired/replaced to satisfy endpoint connectivity.", "May require shortest-path selection, cycle breaking, or terminal pairing.", ("color", "position", "grid_size", "object_count", "other"), ("SPATIAL_RELATION", "CONSTRUCTION"), gap="graph routing / shortest valid terminal path"),
    "247ef758": ann("MULTI_STAGE_COMPOSITION", ("separators/subgrids", "same color", "same row", "same column", "copy"), "3_PLUS", "SAME_SIZE", "FULLY_COVERED", "HIGH", "A small side pattern is projected into a framed canvas at row/column intersections determined by matching border colours.", axes=("color", "position", "grid_size", "object_count"), secondary=("SPATIAL_RELATION", "CONSTRUCTION")),
    "16b78196": ann("MULTI_STAGE_COMPOSITION", ("separators/subgrids", "inside/contains", "translate", "until alignment", "overlay"), "3_PLUS", "SAME_SIZE", "FULLY_COVERED", "MEDIUM", "Objects on opposite sides of a perforated separator are aligned through compatible gaps and overlaid into paired composites.", axes=("color", "position", "grid_size", "object_count", "object_size", "displacement", "orientation"), secondary=("MOTION", "CONSTRUCTION")),
    "edb79dae": ann("COLOR_MAPPING", ("separators/subgrids", "position", "color mapping", "recolor", "crop"), "3_PLUS", "CROP", "FULLY_COVERED", "HIGH", "A border legend maps ordered colours to cells in a framed interior matrix, which is then cropped as the output.", axes=("color", "position", "grid_size", "object_count"), secondary=("CONSTRUCTION",)),
    "4e34c42c": ann("MULTI_STAGE_COMPOSITION", ("connected components", "same color", "nth/order", "translate", "overlay"), "3_PLUS", "CONSTRUCT_NEW", "FULLY_COVERED", "HIGH", "Separate coloured chain fragments are ordered by matching endpoint colours and concatenated into one strip.", axes=("color", "position", "grid_size", "object_count", "displacement", "orientation"), secondary=("SPATIAL_RELATION", "CONSTRUCTION")),
    "62593bfd": ann("MOTION", ("connected components", "orientation", "translate", "until boundary", "inferred displacement"), "2", "SAME_SIZE", "FULLY_COVERED", "HIGH", "Each asymmetric object is moved to the boundary in the direction indicated by its protruding stem/orientation.", axes=("color", "position", "grid_size", "object_count", "displacement", "orientation"), secondary=("GEOMETRIC_TRANSFORM",)),
    "64efde09": ann("CONSTRUCTION", ("lines", "color", "extend line/ray", "until boundary", "repeat"), "3_PLUS", "SAME_SIZE", "FULLY_COVERED", "MEDIUM", "Multi-colour seed segments are extended toward boundaries, preserving/repeating their ordered colour profile.", axes=("color", "position", "grid_size", "object_count", "displacement", "orientation"), secondary=("PATTERN_PROGRESSION", "MOTION")),
    "1818057f": ann("OTHER_OR_UNCERTAIN", ("lines", "inside/contains", "mask", "recolor", "periodic pattern"), "UNKNOWN", "SAME_SIZE", "OUTSIDE_CURRENT_ONTOLOGY", "LOW", "Selected cells inside an irregular red/yellow partition pattern are recoloured light blue according to a repeated enclosure pattern.", "The marked cells may be centres of rectangles, topological holes, or local parity violations.", ("color", "position", "grid_size", "object_count", "other"), ("PATTERN_PROGRESSION", "MASK_SET"), gap="topological region selection in irregular line partitions"),
    "dd6b8c4b": ann("CONDITIONAL_CONTROL", ("frames", "inside/contains", "color mapping", "if/else on attribute", "recolor"), "3_PLUS", "SAME_SIZE", "PARTIALLY_COVERED", "LOW", "Outlying maroon cells around nested frames control which inner bands are recoloured/retained.", "Several mappings from outlier location/count to affected ring remain plausible.", ("color", "position", "grid_size", "object_count", "other"), ("COLOR_MAPPING",), gap="outlier-coded nested-ring state update"),
    "7ed72f31": ann("GEOMETRIC_TRANSFORM", ("connected components", "orientation", "rotate 90", "reflect LR", "position"), "2", "SAME_SIZE", "FULLY_COVERED", "MEDIUM", "Local multicolour objects are reoriented into a canonical arrangement determined by their protrusion/axis.", "Some examples can also be described as moving coloured arms without rotating the whole object.", ("color", "position", "grid_size", "object_count", "orientation"), ("MOTION",)),
    "c7f57c3e": ann("COLOR_MAPPING", ("connected components", "orientation", "scale", "color mapping", "copy reference color"), "3_PLUS", "SAME_SIZE", "PARTIALLY_COVERED", "MEDIUM", "A large multicolour exemplar supplies relative colour roles that are transferred to smaller shape variants/orientations.", axes=("color", "position", "grid_size", "object_count", "object_size", "orientation"), secondary=("GEOMETRIC_TRANSFORM",)),
    "e376de54": ann("COUNTING_NUMERIC", ("lines", "height", "width", "nth/order", "aligned"), "2", "SAME_SIZE", "PARTIALLY_COVERED", "MEDIUM", "Same-colour line segments are shifted/trimmed into an ordered aligned sequence according to their lengths.", axes=("color", "position", "grid_size", "object_count", "object_size", "orientation"), secondary=("MOTION", "SPATIAL_RELATION")),
    "88bcf3b4": ann("MOTION", ("lines", "same color", "translate", "until touching", "aligned"), "2", "SAME_SIZE", "FULLY_COVERED", "HIGH", "Separated orthogonal coloured path segments are translated until their endpoints form one continuous aligned path.", axes=("color", "position", "grid_size", "object_count", "displacement", "orientation"), secondary=("SPATIAL_RELATION",)),
    "8b7bacbf": ann("MULTI_STAGE_COMPOSITION", ("connected", "inside/contains", "propagation", "color mapping", "fill"), "3_PLUS", "SAME_SIZE", "FULLY_COVERED", "MEDIUM", "Colours propagate from track endpoints along blue routes to fill the interiors of the red loops they reach.", axes=("color", "position", "grid_size", "object_count", "displacement"), secondary=("PATTERN_PROGRESSION", "COLOR_MAPPING")),
    "9385bd28": ann("PATTERN_PROGRESSION", ("connected components", "color", "propagation", "fill", "until boundary"), "3_PLUS", "SAME_SIZE", "PARTIALLY_COVERED", "MEDIUM", "Boundary colour keys and scattered paired cells seed rectangular filled regions that grow to inferred extents.", "Extent may be defined by pair distance rather than boundary contact.", ("color", "position", "grid_size", "object_count", "object_size", "displacement"), ("CONSTRUCTION",)),
    "2b83f449": ann("PATTERN_PROGRESSION", ("lines", "same row", "position", "periodic pattern", "propagation"), "3_PLUS", "SAME_SIZE", "PARTIALLY_COVERED", "LOW", "Orange intervals across horizontal lanes induce magenta vertical traces at selected columns, respecting gaps and lane endpoints.", "Could encode interval overlap/intersection across adjacent lanes.", ("color", "position", "grid_size", "object_count", "displacement", "other"), ("SPATIAL_RELATION",)),
    "7491f3cf": ann("MASK_SET", ("separators/subgrids", "overlay", "union", "intersection", "difference"), "3_PLUS", "SAME_SIZE", "FULLY_COVERED", "HIGH", "The fourth panel is completed by a set operation/composition over the first three aligned panels.", axes=("color", "grid_size", "object_count", "other"), secondary=("MULTI_STAGE_COMPOSITION",)),
    "78332cb0": ann("GEOMETRIC_TRANSFORM", ("separators/subgrids", "nth/order", "rotate 90", "copy", "overlay"), "3_PLUS", "RESHAPE", "PARTIALLY_COVERED", "HIGH", "Tiles separated by magenta dividers are reordered and concatenated along the transposed long axis.", axes=("color", "grid_size", "object_count", "orientation"), secondary=("CONSTRUCTION",)),
}


def load_ontology(repo: Path) -> tuple[dict[str, list[str]], dict[str, Any]]:
    v2 = repo / "src/foundation_capability_bank_v2/pipeline.py"
    v3 = repo / "src/foundation_capability_bank_v3/pipeline.py"
    tree = ast.parse(v2.read_text(encoding="utf-8"))
    ontology = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ONTOLOGY" for t in node.targets):
            ontology = ast.literal_eval(node.value)
            break
    if ontology is None:
        raise RuntimeError("ONTOLOGY_NOT_FOUND")
    concepts = [item for values in ontology.values() for item in values]
    if len(concepts) != 83 or len(set(concepts)) != 83:
        raise RuntimeError("ONTOLOGY_NOT_EXACTLY_83_UNIQUE_CONCEPTS")
    return ontology, {
        "status": "PASS",
        "identity": "FOUNDATION_CAPABILITY_BANK_V3_FROZEN_ONTOLOGY_83",
        "concept_count": len(concepts),
        "canonical_sha256": digest(ontology),
        "v2_source_sha256": sha256_file(v2),
        "v3_source_sha256": sha256_file(v3),
    }


def git_json(repo: Path, commit: str, task_id: str) -> dict[str, Any]:
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:data/evaluation/{task_id}.json"])
    parsed = json.loads(raw)
    # This is the irreversible classification boundary: test outputs are not
    # returned, hashed, logged, or retained by the classifier.
    return {"train": parsed["train"], "test": [{"input": item["input"]} for item in parsed["test"]]}


def component_count(grid: list[list[int]]) -> int:
    h, w = len(grid), len(grid[0])
    background = Counter(cell for row in grid for cell in row).most_common(1)[0][0]
    seen: set[tuple[int, int]] = set()
    count = 0
    for r in range(h):
        for c in range(w):
            if (r, c) in seen or grid[r][c] == background:
                continue
            count += 1
            color = grid[r][c]
            stack = [(r, c)]
            seen.add((r, c))
            while stack:
                rr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (rr + dr, cc + dc)
                    if 0 <= q[0] < h and 0 <= q[1] < w and q not in seen and grid[q[0]][q[1]] == color:
                        seen.add(q)
                        stack.append(q)
    return count


def grid_metadata(grid: list[list[int]]) -> dict[str, Any]:
    values = [cell for row in grid for cell in row]
    counts = Counter(values)
    background = counts.most_common(1)[0][0]
    active = [(r, c) for r, row in enumerate(grid) for c, cell in enumerate(row) if cell != background]
    bbox = None if not active else {
        "r0": min(r for r, _ in active), "r1": max(r for r, _ in active),
        "c0": min(c for _, c in active), "c1": max(c for _, c in active),
    }
    return {
        "height": len(grid),
        "width": len(grid[0]),
        "colors": sorted(counts),
        "color_count": len(counts),
        "background_color": background,
        "nonbackground_cells": len(active),
        "same_color_connected_components_4": component_count(grid),
        "nonbackground_bbox": bbox,
    }


def flags(caps: list[str], ontology: dict[str, list[str]]) -> dict[str, bool]:
    groups = {name: set(values) for name, values in ontology.items()}
    capset = set(caps)
    return {
        "requires_selector": bool(capset & groups["SELECTORS"]),
        "requires_relation": bool(capset & groups["RELATIONS"]),
        "requires_numeric_reasoning": bool(capset & groups["NUMERIC"]),
        "requires_state_or_progression": bool(capset & groups["PATTERN_STATE"]),
        "requires_conditional_control": bool(capset & groups["CONTROL"]),
    }


def build(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    out = args.output.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    ontology, ontology_identity = load_ontology(repo)
    concepts = {item for values in ontology.values() for item in values}
    if set(manifest["task_ids"]) != set(ANNOTATIONS) or len(ANNOTATIONS) != 60:
        raise RuntimeError("ANNOTATION_COHORT_MISMATCH")

    task_manifest = manifest["tasks"]
    redacted: dict[str, dict[str, Any]] = {}
    identity_rows = []
    for task_id in manifest["task_ids"]:
        task = git_json(args.official_repo, args.official_commit, task_id)
        expected = task_manifest[task_id]
        task_sha = digest(task)
        test_input_shas = [digest(item["input"]) for item in task["test"]]
        expected_test_shas = [item["input_sha256"] for item in expected["test_index_structure"]]
        status = "PASS" if task_sha == expected["task_sha256"] and test_input_shas == expected_test_shas else "FAIL"
        identity_rows.append({"task_id": task_id, "task_sha256": task_sha, "test_input_sha256": test_input_shas, "status": status})
        redacted[task_id] = task
    output_count = sum(len(task["test"]) for task in redacted.values())
    identity_status = "PASS" if len(redacted) == 60 and output_count == 89 and all(row["status"] == "PASS" for row in identity_rows) else "FAIL"
    if identity_status != "PASS":
        raise RuntimeError("EVAL60_IDENTITY_FAIL")
    identity = {
        "status": identity_status,
        "cohort_id": "eval60",
        "task_count": len(redacted),
        "test_output_count": output_count,
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_task_ids_sha256": manifest["task_ids_sha256"],
        "manifest_source_challenge_byte_sha256": manifest["source_challenge_sha256"],
        "source_dataset_version": manifest["source_dataset_version"],
        "official_repository": OFFICIAL_REPOSITORY,
        "official_compatible_commit": args.official_commit,
        "selected_task_hash_parity": "60_OF_60_PASS",
        "test_input_hash_parity": "89_OF_89_PASS",
        "source_monolithic_challenge_not_reconstructed": True,
        "test_gold_emitted_to_classifier": False,
        "test_gold_hashed_by_classifier": False,
        "records": identity_rows,
    }
    write_json(out / "EVAL60_IDENTITY_AUDIT.json", identity)

    task_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    gaps = []
    for task_id in manifest["task_ids"]:
        base = dict(ANNOTATIONS[task_id])
        unknown = sorted(set(base["required_capabilities"]) - concepts)
        if unknown:
            raise RuntimeError(f"NON_ONTOLOGY_CAPABILITY:{task_id}:{unknown}")
        if base["primary_family"] not in PRIMARY_FAMILIES or base["output_geometry"] not in GEOMETRIES:
            raise RuntimeError(f"ENUM_FAIL:{task_id}")
        if base["ontology_coverage"] not in COVERAGES or base["estimated_composition_depth"] not in DEPTHS or base["classification_confidence"] not in CONFIDENCES:
            raise RuntimeError(f"ENUM_FAIL:{task_id}")
        if not set(base["parameter_generalization_axes"]) <= AXES:
            raise RuntimeError(f"AXIS_FAIL:{task_id}")
        task = redacted[task_id]
        row = {
            "task_id": task_id,
            "test_output_count": len(task["test"]),
            **{key: value for key, value in base.items() if key != "ontology_gap"},
            **flags(base["required_capabilities"], ontology),
        }
        task_rows.append(row)
        if base["ontology_gap"] or base["ontology_coverage"] in {"PARTIALLY_COVERED", "OUTSIDE_CURRENT_ONTOLOGY", "UNCERTAIN"}:
            gaps.append({
                "task_id": task_id,
                "coverage": base["ontology_coverage"],
                "missing_abstraction": base["ontology_gap"] or "No missing primitive asserted; uncertainty is in composition/interpretation.",
                "evidence": base["classification_evidence"],
                "alternatives": base["ambiguous_alternative_interpretations"],
            })
        for idx, item in enumerate(task["test"]):
            output_rows.append({
                "task_id": task_id,
                "output_id": f"{task_id}:o{idx}",
                "primary_family": row["primary_family"],
                "secondary_families": row["secondary_families"],
                "required_capabilities": row["required_capabilities"],
                "estimated_composition_depth": row["estimated_composition_depth"],
                "ontology_coverage": row["ontology_coverage"],
                "test_input_dimensions": [len(item["input"]), len(item["input"][0])],
                "test_input_structural_metadata": grid_metadata(item["input"]),
            })

    map_json = {
        "schema_version": VERSION,
        "status": "FROZEN_TARGET_BLIND_CLASSIFICATION",
        "task_count": len(task_rows),
        "test_output_count": len(output_rows),
        "ontology_identity": ontology_identity,
        "classification_contract": {
            "allowed": ["task_id", "train_inputs", "train_outputs", "test_inputs", "grid_structural_metadata"],
            "TEST_GOLD_ACCESSED_DURING_CLASSIFICATION": False,
            "HISTORICAL_OUTCOME_ACCESSED_DURING_CLASSIFICATION": False,
            "CAPABILITY_PROFILE_ACCESSED_DURING_CLASSIFICATION": False,
            "test_output_fields_discarded_before_classification": True,
        },
        "tasks": task_rows,
    }
    write_json(out / "EVAL60_CAPABILITY_DEMAND_MAP_V1.json", map_json)
    csv_rows = []
    for row in task_rows:
        flat = dict(row)
        for key in ("secondary_families", "required_capabilities", "parameter_generalization_axes"):
            flat[key] = "|".join(flat[key])
        csv_rows.append(flat)
    task_fields = [
        "task_id", "test_output_count", "primary_family", "secondary_families", "required_capabilities",
        "ontology_coverage", "estimated_composition_depth", "requires_selector", "requires_relation",
        "requires_numeric_reasoning", "requires_state_or_progression", "requires_conditional_control",
        "requires_parameter_inference", "output_geometry", "parameter_generalization_axes",
        "classification_confidence", "classification_evidence", "ambiguous_alternative_interpretations",
    ]
    write_csv(out / "EVAL60_CAPABILITY_DEMAND_MAP_V1.csv", csv_rows, task_fields)
    out_csv_rows = []
    for row in output_rows:
        out_csv_rows.append({
            **row,
            "secondary_families": "|".join(row["secondary_families"]),
            "required_capabilities": "|".join(row["required_capabilities"]),
            "test_input_dimensions": "x".join(map(str, row["test_input_dimensions"])),
            "test_input_structural_metadata": canonical(row["test_input_structural_metadata"]),
        })
    write_csv(out / "EVAL60_OUTPUT_CAPABILITY_INDEX_V1.csv", out_csv_rows, [
        "task_id", "output_id", "primary_family", "secondary_families", "required_capabilities",
        "estimated_composition_depth", "ontology_coverage", "test_input_dimensions", "test_input_structural_metadata",
    ])
    gaps_json = {
        "schema_version": VERSION,
        "status": "FROZEN",
        "gap_or_uncertainty_task_count": len(gaps),
        "outside_current_ontology_task_count": sum(g["coverage"] == "OUTSIDE_CURRENT_ONTOLOGY" for g in gaps),
        "records": gaps,
    }
    write_json(out / "EVAL60_ONTOLOGY_GAPS.json", gaps_json)

    taxonomy_names = [
        "EVAL60_CAPABILITY_DEMAND_MAP_V1.json",
        "EVAL60_CAPABILITY_DEMAND_MAP_V1.csv",
        "EVAL60_OUTPUT_CAPABILITY_INDEX_V1.csv",
        "EVAL60_ONTOLOGY_GAPS.json",
    ]
    taxonomy_hashes = {name: sha256_file(out / name) for name in taxonomy_names}
    freeze = {
        "schema_version": VERSION,
        "status": "FROZEN",
        "task_count": 60,
        "test_output_count": 89,
        "taxonomy_artifact_sha256": taxonomy_hashes,
        "ontology_canonical_sha256": ontology_identity["canonical_sha256"],
        "identity_audit_sha256": sha256_file(out / "EVAL60_IDENTITY_AUDIT.json"),
        "taxonomy_frozen_before_historical_outcome_join": True,
        "TEST_GOLD_ACCESSED_DURING_CLASSIFICATION": False,
        "HISTORICAL_OUTCOME_ACCESSED_DURING_CLASSIFICATION": False,
        "CAPABILITY_PROFILE_ACCESSED_DURING_CLASSIFICATION": False,
    }
    write_json(out / "EVAL60_FAMILY_MAP_FREEZE.json", freeze)
    for name, expected in taxonomy_hashes.items():
        if sha256_file(out / name) != expected:
            raise RuntimeError(f"TAXONOMY_FREEZE_HASH_FAIL:{name}")

    # Outcome boundary: nothing above this line opens the historical CSV.
    historical = list(csv.DictReader(args.historical_results.open(encoding="utf-8", newline="")))
    if len(historical) != 89 or sum(row["ORC_UNION"] == "True" for row in historical) != 35:
        raise RuntimeError("HISTORICAL_ORACLE_IDENTITY_FAIL")
    by_output = {row["output_id"]: row for row in output_rows}
    if set(by_output) != {row["output_id"] for row in historical}:
        raise RuntimeError("HISTORICAL_OUTPUT_SET_MISMATCH")
    summary_buckets: dict[tuple[str, str], list[bool]] = defaultdict(list)
    task_by_id = {row["task_id"]: row for row in task_rows}
    for result in historical:
        task = task_by_id[result["task_id"]]
        solved = result["ORC_UNION"] == "True"
        summary_buckets[("primary_family", task["primary_family"])].append(solved)
        summary_buckets[("composition_depth", task["estimated_composition_depth"])].append(solved)
        summary_buckets[("ontology_coverage", task["ontology_coverage"])].append(solved)
        for capability in task["required_capabilities"]:
            summary_buckets[("required_capability", capability)].append(solved)
        for flag in ("requires_relation", "requires_state_or_progression", "requires_conditional_control", "requires_numeric_reasoning", "requires_selector"):
            summary_buckets[(flag, str(task[flag]).lower())].append(solved)
    family_rows = []
    for (dimension, group), values in sorted(summary_buckets.items()):
        solved = sum(values)
        family_rows.append({
            "dimension": dimension,
            "group": group,
            "output_count": len(values),
            "orc_union_solved_count": solved,
            "orc_union_solve_rate": round(solved / len(values), 6),
        })
    oracle_json = {
        "schema_version": VERSION,
        "status": "POST_FREEZE_HISTORICAL_JOIN_COMPLETE",
        "taxonomy_freeze_sha256": sha256_file(out / "EVAL60_FAMILY_MAP_FREEZE.json"),
        "historical_results_sha256": sha256_file(args.historical_results),
        "total_outputs": 89,
        "ORC_UNION_solved": 35,
        "measure": "R1024_D24_D48_CANDIDATE_POOL_ORACLE_UNION",
        "claim_limit": "35/89 is a candidate-pool oracle measure, not direct Base free-generation accuracy.",
        "groups": family_rows,
    }
    write_json(out / "EVAL60_HISTORICAL_ORACLE_BY_FAMILY.json", oracle_json)
    write_csv(out / "EVAL60_HISTORICAL_ORACLE_BY_FAMILY.csv", family_rows, [
        "dimension", "group", "output_count", "orc_union_solved_count", "orc_union_solve_rate",
    ])

    alignment = {
        "schema_version": VERSION,
        "status": "NOT_RUN_AWAITING_COMPLETE_FOUNDATION_V2_CAPABILITY_DIAGNOSTIC",
        "taxonomy_input": "EVAL60_FAMILY_MAP_FREEZE.json",
        "taxonomy_labels_are_immutable": True,
        "future_inputs": ["Base capability profile", "Foundation-V2 capability profile"],
        "future_categories": {
            "REPRESENTATION_GAP_EXPECTED": "At least one important independently measured required capability is WEAK.",
            "COMPOSITION_GAP_EXPECTED": "Required primitives are STRONG/SATURATED but matching composition-development evidence is WEAK/PARTIAL.",
            "PIPELINE_OR_INDUCTION_GAP_SUSPECT": "Required primitives and matching composition evidence are STRONG, but the historical candidate-pool oracle misses.",
            "ONTOLOGY_GAP": "The task requires an abstraction outside the frozen 83-concept ontology.",
            "UNCERTAIN": "Diagnostic correspondence is insufficient.",
        },
        "claim_limits": [
            "Capability demand alignment is explanatory evidence, not direct task accuracy.",
            "Primitive strength does not establish latent rule induction, composition, TTT, search, traversal, or selector/ranker success.",
            "A WEAK synthetic capability does not prove an Eval60 miss has the same cause.",
        ],
        "Base_profile_loaded": False,
        "Foundation_V2_profile_loaded": False,
        "alignment_categories_calculated": False,
    }
    write_json(out / "EVAL60_CAPABILITY_ALIGNMENT_PROTOCOL.json", alignment)

    primary_counts = Counter(row["primary_family"] for row in task_rows)
    depth_counts = Counter(row["estimated_composition_depth"] for row in task_rows)
    coverage_counts = Counter(row["ontology_coverage"] for row in task_rows)
    capability_counts = Counter(cap for row in task_rows for cap in row["required_capabilities"])
    report = {
        "schema_version": VERSION,
        "status": "COMPLETE_STOP_BEFORE_MODEL_PROFILE_ALIGNMENT",
        "identity": "PASS_60_TASKS_89_OUTPUTS",
        "primary_family_counts": dict(sorted(primary_counts.items())),
        "composition_depth_distribution": dict(sorted(depth_counts.items())),
        "ontology_coverage_counts": dict(sorted(coverage_counts.items())),
        "most_frequent_required_capabilities": [{"capability": k, "task_count": v} for k, v in capability_counts.most_common(15)],
        "relation_heavy_task_count": sum(row["requires_relation"] for row in task_rows),
        "state_progression_heavy_task_count": sum(row["requires_state_or_progression"] for row in task_rows),
        "conditional_control_task_count": sum(row["requires_conditional_control"] for row in task_rows),
        "multi_stage_composition_task_count": sum(row["estimated_composition_depth"] == "3_PLUS" for row in task_rows),
        "historical_oracle": "35/89 R1024 candidate-pool ORC_UNION",
        "taxonomy_frozen_before_historical_outcome_join": "PASS",
        "test_gold_used_during_classification": False,
        "current_Base_or_Foundation_V2_results_used_during_classification": False,
        "model_profile_alignment": "NOT_RUN",
    }
    write_json(out / "SUMMARY.json", report)

    ledger_files = [
        "EVAL60_IDENTITY_AUDIT.json", *taxonomy_names, "EVAL60_FAMILY_MAP_FREEZE.json",
        "EVAL60_HISTORICAL_ORACLE_BY_FAMILY.json", "EVAL60_HISTORICAL_ORACLE_BY_FAMILY.csv",
        "EVAL60_CAPABILITY_ALIGNMENT_PROTOCOL.json", "SUMMARY.json",
    ]
    lines = [f"{sha256_file(out / name)}  {name}" for name in ledger_files]
    (out / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--official-repo", type=Path, required=True)
    p.add_argument("--official-commit", default=OFFICIAL_COMMIT)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--historical-results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p


def main() -> None:
    build(parser().parse_args())


if __name__ == "__main__":
    main()
