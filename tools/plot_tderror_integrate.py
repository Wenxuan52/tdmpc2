#!/usr/bin/env python3
"""Plot per-task Cross TD-error curves for TD-MPC2 and MBDPO."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_ROOT = Path("/media/datasets/cheliu21/cxy_worldmodel/tderror_metric")
OUT_DIR = Path("figures")
OUT_FILE_PREFIX = "tderror_separate"
OUT_FILE = f"{OUT_FILE_PREFIX}.png"

TASK_GRID = [
    ["acrobot-swingup", "mw-button-press-wall"],
    ["cheetah-run", "mw-handle-pull-side"],
    ["dog-trot", "mw-pick-place"],
    ["humanoid-walk", "mw-window-open"],
]
ALL_TASKS = [t for row in TASK_GRID for t in row]

METHODS = {
    "TD-MPC2": {
        "seeds": [4, 5, 6],
        "col": "mppi/mppi_cross_td_error",
        "color": "#2b6cb0",
    },
    "MBDPO": {
        "seeds": [1, 2, 3],
        "col": "diffusion/diffusion_cross_td_error",
        "color": "#d64545",
    },
}

X_MAX = 1_000_000
X_TICKS = np.linspace(0, X_MAX, 6)
X_TICK_LABELS = ["0k", "200k", "400k", "600k", "800k", "1M"]

X_TICK_FONT_SIZE = 17
Y_TICK_FONT_SIZE = 17
X_LABEL_FONT_SIZE = 18
Y_LABEL_FONT_SIZE = 20
SUBPLOT_TITLE_FONT_SIZE = 22
LEGEND_FONT_SIZE = 20
LEGEND_Y = -0.01
LINE_WIDTH = 2.0
LEGEND_LINE_WIDTH = 5.0

MEAN_ALPHA = 0.95
CI_ALPHA = 0.14
CI_Z = 1.96

MEAN_EMA_SMOOTH = 0.80
CI_EMA_SMOOTH = 0.90


USE_MANUAL_Y_BOUNDS = False
Y_BOUNDS_MANUAL = {task: None for task in ALL_TASKS}
Y_BOUNDS_OVERRIDE = {}


def _load_task_seed(task: str, seed: int) -> pd.DataFrame:
    path = DATA_ROOT / f"TD_error_metric_{task}_seed{seed}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df if "step" in df.columns else pd.DataFrame()


def _interp(df: pd.DataFrame, col: str, x_grid: np.ndarray) -> np.ndarray:
    if df.empty or col not in df.columns:
        return np.full_like(x_grid, np.nan, dtype=float)
    sdf = df[["step", col]].copy()
    sdf["step"] = pd.to_numeric(sdf["step"], errors="coerce")
    sdf[col] = pd.to_numeric(sdf[col], errors="coerce")
    sdf = sdf.dropna(subset=["step", col]).sort_values("step").drop_duplicates(subset=["step"], keep="last")
    if sdf.empty:
        return np.full_like(x_grid, np.nan, dtype=float)
    aligned = pd.Series(sdf[col].to_numpy(dtype=float), index=sdf["step"].to_numpy(dtype=float)).reindex(x_grid.astype(float))
    return aligned.interpolate(method="index", limit_area="inside").to_numpy(dtype=float)


def _ema(values: np.ndarray, smooth: float) -> np.ndarray:
    smooth = float(np.clip(smooth, 0.0, 0.999999))
    out = values.astype(float, copy=True)
    finite_idx = np.flatnonzero(np.isfinite(out))
    if finite_idx.size == 0:
        return out
    prev = out[finite_idx[0]]
    for idx in finite_idx[1:]:
        prev = smooth * prev + (1.0 - smooth) * out[idx]
        out[idx] = prev
    return out


def _mean_ci95(curves: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.vstack(curves).astype(float)
    finite = np.isfinite(arr)
    n = np.sum(finite, axis=0)

    mean = np.full(arr.shape[1], np.nan, dtype=float)
    valid_mean = n > 0
    if np.any(valid_mean):
        sums = np.where(finite, arr, 0.0).sum(axis=0)
        mean[valid_mean] = sums[valid_mean] / n[valid_mean]

    ci95 = np.full_like(mean, np.nan)
    valid_ci = n >= 2
    if np.any(valid_ci):
        centered = np.where(finite, arr - mean[None, :], 0.0)
        var = np.where(valid_ci, (centered**2).sum(axis=0) / (n - 1), np.nan)
        se = np.sqrt(var) / np.sqrt(n)
        ci95[valid_ci] = CI_Z * se[valid_ci]

    return mean, ci95


def _smooth_mean_ci(mean: np.ndarray, ci95: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _ema(mean, MEAN_EMA_SMOOTH), _ema(ci95, CI_EMA_SMOOTH)


def _prettify_task(task: str) -> str:
    return " ".join(tok.capitalize() for tok in task.split("-"))


def _collect_curves(task: str, seeds: List[int], col: str, x_grid: np.ndarray) -> List[np.ndarray]:
    out = []
    for seed in seeds:
        df = _load_task_seed(task, seed)
        if df.empty:
            continue
        curve = _interp(df, col, x_grid)
        if np.isfinite(curve).any():
            out.append(curve)
    return out


def _compute_unified_ymax(x_grid: np.ndarray) -> dict[str, float]:
    ymax = {}
    for task in ALL_TASKS:
        vals = []
        for meta in METHODS.values():
            curves = _collect_curves(task, meta["seeds"], meta["col"], x_grid)
            if not curves:
                continue
            mean, ci95 = _mean_ci95(curves)
            mean, ci95 = _smooth_mean_ci(mean, ci95)
            upper = mean + ci95
            finite = upper[np.isfinite(upper)]
            if finite.size:
                vals.append(float(np.max(finite)) + 0.01)
        ymax[task] = max(vals) if vals else 0.05
    return ymax


def _get_y_bounds(task: str, unified_ymax: dict[str, float]) -> tuple[float, float]:
    if USE_MANUAL_Y_BOUNDS:
        manual = Y_BOUNDS_MANUAL.get(task)
        if manual is not None:
            return manual
    if task in Y_BOUNDS_OVERRIDE:
        return Y_BOUNDS_OVERRIDE[task]
    return 0.0, unified_ymax.get(task, 0.05)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x_grid = np.linspace(0, X_MAX, 1001, dtype=float)
    unified_ymax = _compute_unified_ymax(x_grid)

    fig, axes = plt.subplots(4, 2, figsize=(18, 10), sharex=True)
    for r in range(4):
        for c in range(2):
            task = TASK_GRID[r][c]
            ax = axes[r, c]

            for label, meta in METHODS.items():
                curves = _collect_curves(task, meta["seeds"], meta["col"], x_grid)
                if not curves:
                    continue
                mean, ci95 = _mean_ci95(curves)
                mean, ci95 = _smooth_mean_ci(mean, ci95)
                ax.plot(x_grid, mean, lw=LINE_WIDTH, alpha=MEAN_ALPHA, color=meta["color"], label=label)
                ax.fill_between(x_grid, mean - ci95, mean + ci95, color=meta["color"], alpha=CI_ALPHA, linewidth=0)

            ymin, ymax = _get_y_bounds(task, unified_ymax)
            ax.set_ylim(ymin, ymax)
            ax.set_yticks([ymin, ymax])
            ax.set_yticklabels([f"{ymin:.1f}", f"{ymax:.1f}"], fontsize=Y_TICK_FONT_SIZE)

            title_task = task[3:] if c == 1 and task.startswith("mw-") else task
            ax.set_title(_prettify_task(title_task), fontsize=SUBPLOT_TITLE_FONT_SIZE)
            ax.set_facecolor("white")
            ax.grid(color="#d9d9d9", linewidth=0.8, alpha=0.55)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(True)
            ax.spines["bottom"].set_visible(True)
            ax.spines["left"].set_color("black")
            ax.spines["bottom"].set_color("black")
            ax.spines["left"].set_linewidth(1.2)
            ax.spines["bottom"].set_linewidth(1.2)

            if r == 3:
                ax.set_xlabel("Time steps (1M)", fontsize=X_LABEL_FONT_SIZE)
                ax.set_xticks(X_TICKS)
                ax.set_xticklabels(X_TICK_LABELS, fontsize=X_TICK_FONT_SIZE)
            else:
                ax.tick_params(axis="x", labelsize=X_TICK_FONT_SIZE)

    fig.supylabel("Cross TD-error", fontsize=Y_LABEL_FONT_SIZE)

    legend_items = {}
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l not in legend_items:
                legend_items[l] = h

    if legend_items:
        leg = fig.legend(
            list(legend_items.values()),
            list(legend_items.keys()),
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.5, LEGEND_Y),
            fontsize=LEGEND_FONT_SIZE,
            frameon=False,
        )
        for h in leg.get_lines():
            h.set_linewidth(LEGEND_LINE_WIDTH)

    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.99])
    out_path = OUT_DIR / OUT_FILE
    fig.savefig(out_path, dpi=300)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
