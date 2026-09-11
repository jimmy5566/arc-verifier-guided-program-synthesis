"""Task-level, characteristic-aware split construction with no task-specific rules."""
from __future__ import annotations
from pathlib import Path
import hashlib
import pandas as pd

def make_task_splits(metadata_csv: str | Path, output_csv: str | Path, development_fraction: float=.60, held_out_fraction: float=.25) -> pd.DataFrame:
    """Assign deterministic strata-aware task splits from recorded characteristics.

    The stable hash is used only to make assignments reproducible; task identifiers never encode a rule.
    """
    if not 0 < development_fraction < 1 or not 0 < held_out_fraction < 1 or development_fraction + held_out_fraction >= 1: raise ValueError("fractions must be positive and leave challenge-like holdout")
    frame=pd.read_csv(metadata_csv).copy(); strata=frame[["shape_change","multi_object","symmetry"]].astype(str).agg("|".join,axis=1)
    split=[]
    for task_id,stratum in zip(frame.task_id,strata):
        u=int(hashlib.sha256(f"ARC2-split:{stratum}:{task_id}".encode()).hexdigest()[:8],16)/2**32
        split.append("development" if u < development_fraction else "held_out" if u < development_fraction+held_out_fraction else "challenge_like")
    frame["split"]=split; frame["split_note"]="characteristic-stratified deterministic hash; no task-specific heuristic"
    Path(output_csv).parent.mkdir(parents=True,exist_ok=True); frame.to_csv(output_csv,index=False); return frame
