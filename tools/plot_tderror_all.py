#!/usr/bin/env python3
"""Aggregate Cross TD-error plots for DMControl and MetaWorld."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_ROOT = Path("/media/datasets/cheliu21/cxy_worldmodel/tderror_metric")
OUT_PATH = Path("figures/TD_error.png")

DM_TASKS = ["acrobot-swingup", "cheetah-run", "dog-trot", "humanoid-walk"]
MW_TASKS = ["mw-button-press-wall", "mw-handle-pull-side", "mw-pick-place", "mw-window-open"]

X_MAX = 1_000_000
X_GRID = np.arange(0, X_MAX + 1, 1_000, dtype=float)
X_TICKS = [0, 200_000, 400_000, 600_000, 800_000, 1_000_000]
X_TICK_LABELS = ["0k", "200k", "400k", "600k", "800k", "1M"]

COLORS = {
    "TD-MPC2": "#2b6cb0",  # blue
    "MBDPO": "#d64545",  # red
}
MEAN_ALPHA = 0.95
CI_ALPHA = 0.18


def _file(task: str, seed: int) -> Path:
    return DATA_ROOT / f"TD_error_metric_{task}_seed{seed}.csv"


def _interp_series(path: Path, column: str) -> np.ndarray | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "step" not in df.columns or column not in df.columns:
        return None

    sdf = df[["step", column]].copy()
    sdf["step"] = pd.to_numeric(sdf["step"], errors="coerce")
    sdf[column] = pd.to_numeric(sdf[column], errors="coerce")
    sdf = sdf.dropna(subset=["step", column]).sort_values("step").drop_duplicates(subset=["step"], keep="last")
    if sdf.empty:
        return None

    ser = pd.Series(sdf[column].to_numpy(dtype=float), index=sdf["step"].to_numpy(dtype=float))
    out = ser.reindex(X_GRID).interpolate(method="index", limit_area="inside")
    return out.to_numpy(dtype=float)


def _collect_domain_samples(tasks: Iterable[str], seeds: Iterable[int], column: str) -> np.ndarray:
    samples: list[np.ndarray] = []
    for task in tasks:
        for seed in seeds:
            arr = _interp_series(_file(task, seed), column)
            if arr is not None and np.isfinite(arr).any():
                samples.append(arr)
    if not samples:
        return np.full((1, len(X_GRID)), np.nan)
    return np.vstack(samples)


def _mean_ci95(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.sum(np.isfinite(samples), axis=0)
    mean = np.nanmean(samples, axis=0)
    std = np.nanstd(samples, axis=0, ddof=1)
    se = np.where(n >= 2, std / np.sqrt(n), np.nan)
    ci = 1.96 * se
    return mean, mean - ci, mean + ci


def _style_axis(ax: plt.Axes, title: str, show_ylabel: bool) -> None:
    ax.set_title(title, fontsize=24)
    ax.set_xlabel("Env Steps", fontsize=22)
    ax.set_ylabel("Cross TD-error" if show_ylabel else "", fontsize=22)
    ax.set_xlim(0, X_MAX)
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels(X_TICK_LABELS, fontsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.grid(color="#d9d9d9", linewidth=0.8, alpha=0.55, axis="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)


def _plot(ax: plt.Axes, tasks: list[str], title: str, show_ylabel: bool) -> None:
    # MBDPO: seeds 1-3, diffusion/diffusion_cross_td_error
    mbdpo = _collect_domain_samples(tasks, [1, 2, 3], "diffusion/diffusion_cross_td_error")
    mbdpo_mean, mbdpo_lo, mbdpo_hi = _mean_ci95(mbdpo)
    ax.plot(X_GRID, mbdpo_mean, color=COLORS["MBDPO"], linewidth=2.8, alpha=MEAN_ALPHA, label="MBDPO")
    ax.fill_between(X_GRID, mbdpo_lo, mbdpo_hi, color=COLORS["MBDPO"], alpha=CI_ALPHA, linewidth=0)

    # TD-MPC2: seeds 4-6, mppi/mppi_cross_td_error
    tdmpc2 = _collect_domain_samples(tasks, [4, 5, 6], "mppi/mppi_cross_td_error")
    td_mean, td_lo, td_hi = _mean_ci95(tdmpc2)
    ax.plot(X_GRID, td_mean, color=COLORS["TD-MPC2"], linewidth=2.8, alpha=MEAN_ALPHA, label="TD-MPC2")
    ax.fill_between(X_GRID, td_lo, td_hi, color=COLORS["TD-MPC2"], alpha=CI_ALPHA, linewidth=0)

    _style_axis(ax, title, show_ylabel)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    _plot(axes[0], DM_TASKS, "DMControl", show_ylabel=True)
    _plot(axes[1], MW_TASKS, "MetaWorld", show_ylabel=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=18)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(OUT_PATH, dpi=300)
    print(f"Saved figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
