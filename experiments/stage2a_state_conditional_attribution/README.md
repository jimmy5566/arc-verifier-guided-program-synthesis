# Stage-2A state-conditional attribution

CPU-only, isolated extension of Stage-1. It preserves Stage-0 and Stage-1
artifacts and uses a new synthetic single-primary-fault sandbox only.

Run:

```powershell
.\.venv\Scripts\python.exe .\experiments\stage2a_state_conditional_attribution\src\simulate.py
.\.venv\Scripts\python.exe .\experiments\stage2a_state_conditional_attribution\src\plotting.py
.\.venv\Scripts\python.exe -m pytest .\experiments\stage2a_state_conditional_attribution\tests -q
```

The pre-registered setup is in `PREREGISTRATION.md`; results and the decision
interpretation are in `STAGE2A_REPORT.md`. No GPU, LLM, ARC target, Kaggle, or
production artifact is accessed.
