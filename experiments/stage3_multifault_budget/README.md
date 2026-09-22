# Stage-3 multi-fault budgeted intervention

CPU-only, preregistered test of marginal component responsibility when an
episode has no fault, one fault, or two simultaneous faults and actions have a
limited budget.

```powershell
.\.venv\Scripts\python.exe .\experiments\stage3_multifault_budget\src\run_experiment.py
.\.venv\Scripts\python.exe .\experiments\stage3_multifault_budget\src\plotting.py
.\.venv\Scripts\python.exe -m pytest .\experiments\stage3_multifault_budget\tests -q
```

No GPU, LLM, Kaggle, ARC dataset, or production component is accessed.
