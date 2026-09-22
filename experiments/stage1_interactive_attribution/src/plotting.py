"""Static plots for preregistered Stage-1 outputs."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def confusion(matrix, output: Path) -> None:
    names=["success","generator","selector","executor","environment"]
    figure, axis=plt.subplots(figsize=(6.2,5.3)); image=axis.imshow(matrix,cmap="Blues")
    axis.set(xticks=range(5),yticks=range(5),xticklabels=names,yticklabels=names,xlabel="Predicted",ylabel="Latent (evaluation only)",title="RandomForest attribution confusion: balanced test")
    axis.tick_params(axis="x",rotation=30)
    for i in range(5):
        for j in range(5): axis.text(j,i,str(matrix[i][j]),ha="center",va="center",fontsize=8)
    figure.colorbar(image,ax=axis); figure.tight_layout(); figure.savefig(output,dpi=180); plt.close(figure)


def lines(rows, xkey, ykey, groupkey, title, xlabel, ylabel, output: Path) -> None:
    figure,axis=plt.subplots(figsize=(7.4,4.5))
    groups={}
    for row in rows: groups.setdefault(row[groupkey],[]).append(row)
    raw_x=sorted({row[xkey] for row in rows}, key=lambda value: str(value))
    numeric=all(isinstance(value,(int,float)) for value in raw_x)
    positions={value:index for index,value in enumerate(raw_x)}
    for name,values in sorted(groups.items()):
        values.sort(key=lambda r: positions[r[xkey]] if not numeric else float(r[xkey]))
        x=[float(r[xkey]) for r in values] if numeric else [positions[r[xkey]] for r in values]
        axis.plot(x,[float(r[ykey]) for r in values],marker="o",label=name)
    if not numeric:
        axis.set_xticks(range(len(raw_x)), [str(value) for value in raw_x], rotation=20)
    axis.set(title=title,xlabel=xlabel,ylabel=ylabel);axis.grid(alpha=.25);axis.legend();figure.tight_layout();figure.savefig(output,dpi=180);plt.close(figure)


def scatter(rows, output: Path) -> None:
    figure,axis=plt.subplots(figsize=(6.4,4.6))
    for row in rows:
        if row["recovery_ratio"] is None: continue
        axis.scatter(row["macro_f1"],row["recovery_ratio"],s=55);axis.annotate(row.get("label",row.get("regime","")),(row["macro_f1"],row["recovery_ratio"]),xytext=(4,4),textcoords="offset points",fontsize=8)
    axis.axhline(0,color="gray",lw=1);axis.set(xlabel="Attribution macro-F1",ylabel="Oracle-benefit recovery ratio",title="Attribution quality versus adaptation utility");axis.grid(alpha=.25);figure.tight_layout();figure.savefig(output,dpi=180);plt.close(figure)
