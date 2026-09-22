# Stage-1 interactive attribution sandbox

This is a preregistered, CPU-only synthetic sequential environment. It is
isolated from production ARC2 code and Stage-0. Run with:

```powershell
.\.venv\Scripts\python.exe .\experiments\stage1_interactive_attribution\src\simulate.py
.\.venv\Scripts\python.exe -m pytest -q .\experiments\stage1_interactive_attribution\tests
```

The simulator holds latent labels internally for metric evaluation only. The
attribution model sees only noisy telemetry columns.
