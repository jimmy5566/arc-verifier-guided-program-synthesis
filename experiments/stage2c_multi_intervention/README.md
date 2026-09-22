# Stage-2C multi-intervention routing

CPU-only, isolated Stage-2C experiment. It imports the unchanged Stage-2A
synthetic generator and writes all new results locally in this directory.

```powershell
.\.venv\Scripts\python.exe .\experiments\stage2c_multi_intervention\src\run_experiment.py
.\.venv\Scripts\python.exe .\experiments\stage2c_multi_intervention\src\plotting.py
.\.venv\Scripts\python.exe -m pytest .\experiments\stage2c_multi_intervention\tests -q
```

No GPU, LLM, Kaggle, ARC data, or production component is accessed.
