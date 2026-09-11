"""Notebook-independent plotting helpers for ARC grids and failures."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from .task import ARCTask

ARC_COLORS = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]
CMAP = ListedColormap(ARC_COLORS)

def show_grid(grid: np.ndarray, ax=None, title: str = ""):
    ax = ax or plt.subplots(figsize=(3, 3))[1]
    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=9, interpolation="nearest")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    return ax

def plot_task(task: ARCTask, output_path: str | Path | None = None):
    pairs = list(task.train) + list(task.test)
    fig, axes = plt.subplots(len(pairs), 2, figsize=(6, max(2, 2.5 * len(pairs))), squeeze=False)
    for row, example in enumerate(pairs):
        kind = "train" if row < len(task.train) else "test"
        show_grid(example.input.values, axes[row, 0], f"{kind} input {row}")
        if example.output is not None: show_grid(example.output.values, axes[row, 1], f"{kind} output {row}")
        else: axes[row, 1].axis("off")
    fig.suptitle(task.task_id); fig.tight_layout()
    if output_path: Path(output_path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig

def plot_prediction(expected: np.ndarray, predicted: np.ndarray, output_path: str | Path | None = None, title: str = "prediction"):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3)); show_grid(expected, axes[0], "expected"); show_grid(predicted, axes[1], title); fig.tight_layout()
    if output_path: Path(output_path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig
