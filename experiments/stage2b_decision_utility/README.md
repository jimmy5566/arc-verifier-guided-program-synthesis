# Stage-2B decision-utility sensitivity

Isolated CPU-only Stage-2B sensitivity study. It reads the unchanged Stage-2A
simulator code but writes only to this directory. See `PREREGISTRATION.md` and
`STAGE2B_REPORT.md`.

```powershell
.\.venv\Scripts\python.exe .\experiments\stage2b_decision_utility\src\run_sensitivity.py
.\.venv\Scripts\python.exe .\experiments\stage2b_decision_utility\src\plotting.py
.\.venv\Scripts\python.exe -m pytest .\experiments\stage2b_decision_utility\tests -q
```
