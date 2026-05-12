#!/usr/bin/env python3
"""Plot online RL diffusion-step ablation curves (2x4 tasks)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_config import load_plot_config

TASKS: List[str] = [
    "dog-walk",
    "hopper-hop",
    "humanoid-run",
    "humanoid-stand",
    "humanoid-walk",
    "mw-assembly",
]

SEED_TO_DIFFUSION: Dict[int, int] = {
    1: 20,
    2: 20,
    3: 20,
    4: 15,
    5: 15,
    6: 15,
    7: 10,
    8: 10,
    9: 10,
    10: 5,
    11: 5,
    12: 5,
}
DIFFUSION_LEVELS = [20, 15, 10, 5]
PLOT_ORDER = [5, 10, 15, 20]

COLORS = {
    20: "#d64a4b",
    15: "#7fd54c",
    10: "#5da7df",
    5: "#8a6bc7",
}

LINEWIDTH = 3.0
MEAN_ALPHA = 0.75
SHADE_ALPHA = 0.12

X_MAX = 2_000_000
GRID_STEP = 100_000
Y_MIN, Y_MAX = -3.0, 103.0
PLOT_CFG = load_plot_config()


DMC_TASK_PREFIXES = ("dog-", "hopper-", "humanoid-")
METAWORLD_TASK_PREFIX = "mw-"


def _align_curve(task: str, df: pd.DataFrame, step_grid: np.ndarray) -> np.ndarray:
    series = pd.Series(df["reward"].to_numpy(dtype=float), index=df["step"].to_numpy(dtype=float))

    if task.startswith(DMC_TASK_PREFIXES):
        half = GRID_STEP / 2
        x = df["step"].to_numpy(dtype=float)
        y = df["reward"].to_numpy(dtype=float)
        out = np.full_like(step_grid, np.nan, dtype=float)
        for i, g in enumerate(step_grid.astype(float)):
            mask = (x >= g - half) & (x <= g + half)
            if np.any(mask):
                out[i] = float(np.nanmax(y[mask]))
        # keep initial point at zero when available
        if len(out) > 0 and np.isnan(out[0]):
            out[0] = 0.0
        return out

    return series.reindex(step_grid.astype(float)).interpolate(method="index", limit_area="inside").to_numpy(dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-root",
        type=Path,
        default=Path("/media/datasets/cheliu21/cxy_worldmodel/online_ablation_csv"),
        help="Directory containing {task}_{seed}.csv files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/online_ablation_diffusion_steps.pdf"),
        help="Output figure path.",
    )
    return parser.parse_args()


def prettify_task_name(task: str) -> str:
    return " ".join(piece.capitalize() for piece in task.split("-"))


def _extract_seed(path: Path) -> int | None:
    m = re.search(r"_(\d+)\.csv$", path.name)
    return int(m.group(1)) if m else None


def _load_single_csv(path: Path, task: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if {"step", "episode_reward"}.issubset(raw.columns):
        df = raw[["step", "episode_reward"]].rename(columns={"episode_reward": "reward"}).copy()
    elif {"step", "episode_success"}.issubset(raw.columns):
        df = raw[["step", "episode_success"]].rename(columns={"episode_success": "reward"}).copy()
    elif {"step", "success"}.issubset(raw.columns):
        df = raw[["step", "success"]].rename(columns={"success": "reward"}).copy()
    elif {"step", "reward"}.issubset(raw.columns):
        df = raw[["step", "reward"]].copy()
    else:
        raise ValueError(f"{path} missing columns: expected step + episode_reward/reward/success/episode_success")

    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
    df = df.dropna(subset=["step", "reward"])
    df = df[(df["step"] >= 0) & (df["step"] <= X_MAX)].copy()
    df["step"] = df["step"].astype(int)
    df = df.sort_values("step").drop_duplicates("step", keep="last")

    if df.empty or int(df.iloc[0]["step"]) != 0:
        df = pd.concat([pd.DataFrame([{"step": 0, "reward": 0.0}]), df], ignore_index=True)
        df = df.sort_values("step").drop_duplicates("step", keep="last")

    return _apply_task_scaling(task, df)


def summarize_ci(curves: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.vstack(curves)
    n = np.sum(np.isfinite(arr), axis=0)
    mean = np.nanmean(arr, axis=0)
    std = np.full_like(mean, np.nan, dtype=float)
    valid_std = n >= 2
    if np.any(valid_std):
        std[valid_std] = np.nanstd(arr[:, valid_std], axis=0, ddof=1)
    ci95 = np.full_like(mean, np.nan, dtype=float)
    valid = n > 0
    ci95[valid] = 1.96 * np.nan_to_num(std[valid], nan=0.0) / np.sqrt(n[valid])
    return mean, ci95




def _apply_task_scaling(task: str, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if task.startswith(DMC_TASK_PREFIXES):
        out["reward"] = out["reward"] / 10.0
    elif task.startswith(METAWORLD_TASK_PREFIX):
        out["reward"] = out["reward"] * 100.0
    return out


def load_task_group_curves(task: str, csv_root: Path, step_grid: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
    files = sorted(csv_root.glob(f"{task}_*.csv"), key=lambda p: (_extract_seed(p) or 10**9, p.name))
    if not files:
        raise FileNotFoundError(f"No CSV files found for task: {task}")

    grouped: Dict[int, List[np.ndarray]] = {k: [] for k in DIFFUSION_LEVELS}
    for path in files:
        seed = _extract_seed(path)
        if seed is None or seed not in SEED_TO_DIFFUSION:
            continue
        df = _load_single_csv(path, task)
        aligned = _align_curve(task, df, step_grid)
        grouped[SEED_TO_DIFFUSION[seed]].append(aligned)


    output: Dict[int, Dict[str, np.ndarray]] = {}
    for diff in DIFFUSION_LEVELS:
        if not grouped[diff]:
            nan = np.full_like(step_grid, np.nan, dtype=float)
            output[diff] = {"mean": nan, "ci": nan}
            continue
        mean, ci = summarize_ci(grouped[diff])
        output[diff] = {"mean": mean, "ci": ci}
    return output


def plot_all(args: argparse.Namespace) -> None:
    max_step_seen = 0
    for task in TASKS:
        for fp in args.csv_root.glob(f"{task}_*.csv"):
            try:
                raw = pd.read_csv(fp, usecols=["step"])
            except Exception:
                continue
            if not raw.empty:
                max_step_seen = max(max_step_seen, int(pd.to_numeric(raw["step"], errors="coerce").max()))
    plot_x_max = int(np.ceil(max(100_000, min(X_MAX, max_step_seen)) / GRID_STEP) * GRID_STEP)
    step_grid = np.arange(0, plot_x_max + GRID_STEP, GRID_STEP, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    legend_handle_map = {}

    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        stats_by_diff = load_task_group_curves(task, args.csv_root, step_grid)

        for diff in PLOT_ORDER:
            stats = stats_by_diff[diff]
            mean, ci = stats["mean"], stats["ci"]
            color = COLORS[diff]
            ax.plot(step_grid, mean, color=color, linewidth=LINEWIDTH, alpha=MEAN_ALPHA)
            ax.fill_between(step_grid, mean - ci, mean + ci, color=color, alpha=SHADE_ALPHA, linewidth=0)

            if idx == 0:
                legend_handle_map[diff] = plt.Line2D([], [], color=color, linewidth=LINEWIDTH, alpha=MEAN_ALPHA)

        ax.set_title(prettify_task_name(task), fontsize=24)
        ax.set_xlim(-1000, plot_x_max)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.grid(True, linestyle="-", linewidth=0.8, alpha=0.25)

        row, col = divmod(idx, 3)
        if plot_x_max <= 500_000:
            ax.set_xticks([0, plot_x_max // 2, plot_x_max])
            ax.set_xticklabels(["0", "250K", "500K"] if row == 1 else [])
        else:
            ax.set_xticks([0, 1_000_000, min(2_000_000, plot_x_max)])
        ax.set_yticks([0, 50, 100])
        if row == 1:
            if plot_x_max > 500_000:
                ax.set_xticklabels(["0", "1M", "2M" if plot_x_max >= 2_000_000 else f"{plot_x_max/1_000_000:.1f}M"])
            ax.tick_params(axis="x", labelsize=20, labelbottom=True)
        else:
            ax.tick_params(axis="x", labelbottom=False)

        if col == 0:
            ax.tick_params(axis="y", labelsize=20, labelleft=True)
        else:
            ax.tick_params(axis="y", labelleft=False)

    legend_order = [20, 15, 10, 5]
    fig.legend(
        [legend_handle_map[d] for d in legend_order],
        [str(d) for d in legend_order],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
        fontsize=22,
    )
    fig.subplots_adjust(left=0.06, right=0.995, top=0.93, bottom=0.18, wspace=0.15, hspace=0.35)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {args.out}")


if __name__ == "__main__":
    plot_all(parse_args())
