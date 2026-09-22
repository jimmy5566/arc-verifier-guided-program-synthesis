# Stage-0 attribution pilot (CPU-only)

This isolated pilot analyses an already-frozen 30-task ARC candidate-ranking
artifact.  It does not load a model, use CUDA, regenerate candidates, modify
production code, or interact with Kaggle.

## Reproduction

From the repository root:

```powershell
.\.venv\Scripts\python.exe .\experiments\stage0_attribution_pilot\src\extract_arc30.py
.\.venv\Scripts\python.exe .\experiments\stage0_attribution_pilot\src\run_stage0.py
.\.venv\Scripts\python.exe -m pytest -q .\experiments\stage0_attribution_pilot\tests
```

`extract_arc30.py` reads cached candidate/ranking records first and reads the
already-local development solution file only to score those frozen predictions.
`run_stage0.py` is a deterministic NumPy/SciPy/Matplotlib analysis of the
derived CSV.  No script writes outside this directory.

The selection layer in this pilot is a likelihood ranker/selector.  It is not
renamed as a general verifier.
