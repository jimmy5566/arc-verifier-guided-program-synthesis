from __future__ import annotations
from dataclasses import dataclass, asdict
import os, platform
@dataclass(frozen=True)
class Environment:
    environment: str; cuda_available: bool; gpu_name: str | None; gpu_memory_bytes: int | None; cpu_cores: int
def detect_environment() -> Environment:
    location="kaggle" if os.getenv("KAGGLE_KERNEL_RUN_TYPE") else "local"
    try:
        import torch
        return Environment(location,torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,os.cpu_count() or 1)
    except ImportError: return Environment(location,False,None,None,os.cpu_count() or 1)
def environment_dict(): return asdict(detect_environment())
