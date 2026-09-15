# Paper draft

`main.tex` is a research preprint draft, not a submitted, peer-reviewed, or accepted paper. It uses only repository evidence and labels local/private artifact sources where detailed generated outputs cannot be redistributed.

## Build

```powershell
cd docs/paper
powershell -ExecutionPolicy Bypass -File figures/generate_figures.ps1
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

`latexmk -pdf main.tex` is an equivalent shortcut when available. The PowerShell figure script is CPU-only, uses only the Windows graphics runtime, and contains audited aggregate values with source paths in comments. It does not run inference.

The expected manuscript length is approximately 6–10 pages with the four figures, depending on the installed TeX distribution.
