from __future__ import annotations
import numpy as np
def exact_accuracy(exact: list[bool]) -> float: return float(np.mean(exact)) if exact else 0.0
