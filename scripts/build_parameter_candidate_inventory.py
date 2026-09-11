"""Write Phase-A finite parameter candidate inventory before model inference."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm.parameter_candidate_inventory import build_candidate_inventory
from llm.parameter_semantic_ontology import ParameterSemanticOntologyV1


OUTPUT = ROOT / "experiments" / "results" / "PARAMETER_CANDIDATE_INVENTORY_V1.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite candidate inventory: {OUTPUT}")
    frozen = json.loads((ROOT / "configs" / "PARAMETER_GROUNDING_REPAIR_V1_FROZEN_CONFIG.json").read_text(encoding="utf-8"))
    ontology = ParameterSemanticOntologyV1()
    result = build_candidate_inventory(s2_baseline_programs=frozen["s2_baseline_programs"], s2_outcomes=frozen["p0_s2_baseline"]["case_outcomes"], ontology=ontology)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(OUTPUT), "case_count": result["case_count"], "sha256": result["sha256"]}))


if __name__ == "__main__":
    main()
