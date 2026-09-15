# Paper draft

`main.tex` is a research preprint draft, not a submitted, peer-reviewed, or accepted paper. It uses only repository evidence and labels local/private artifact sources where detailed generated outputs cannot be redistributed.

## Build

```powershell
cd docs/paper
powershell -ExecutionPolicy Bypass -File figures/generate_figures.ps1
latexmk -pdf main.tex
```

If `latexmk` is unavailable, use `pdflatex main.tex`, `bibtex main`, then run `pdflatex` twice. The PowerShell figure script is CPU-only, uses only the Windows graphics runtime, and contains audited aggregate values with source paths in comments. It does not run inference.

The expected manuscript length is approximately 6–10 pages with the four figures, depending on the installed TeX distribution.
