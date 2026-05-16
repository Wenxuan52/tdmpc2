#!/usr/bin/env python3
"""Plot online RL diffusion-step ablation curves with runtime bar+line chart."""

from __future__ import annotations

import argparse
import re
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

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

RUNTIME_TEXT = {
    "dog-walk": {1: "1 day, 0:07:12", 2: "23:45:10", 3: "22:19:53", 4: "19:59:22", 5: "18:28:13", 6: "17:43:53", 7: "14:22:38", 8: "13:05:55", 9: "12:58:08", 10: "8:48:34", 11: "8:36:26", 12: "7:41:43"},
    "hopper-hop": {1: "1 day, 0:43:10", 2: "1 day, 0:36:35", 3: "23:18:30", 4: "19:23:14", 5: "19:17:26", 6: "17:55:02", 7: "13:43:36", 8: "13:45:32", 9: "12:30:44", 10: "8:08:05", 11: "8:17:48", 12: "8:04:38"},
    "humanoid-run": {1: "1 day, 0:28:14", 2: "1 day, 0:44:56", 3: "1 day, 0:34:14", 4: "18:19:41", 5: "19:16:18", 6: "19:36:09", 7: "13:07:28", 8: "13:56:42", 9: "13:51:45", 10: "7:51:03", 11: "8:19:40", 12: "8:14:28"},
    "humanoid-stand": {4: "19:20:46", 5: "19:32:24", 6: "19:22:05", 7: "13:50:09", 8: "13:51:10", 9: "13:56:46", 10: "8:23:18", 11: "8:21:55", 12: "8:17:27"},
    "humanoid-walk": {1: "1 day, 0:54:17", 2: "1 day, 0:32:26", 3: "1 day, 0:49:19", 4: "19:31:10", 5: "19:15:45", 6: "19:49:40", 7: "13:53:52", 8: "13:49:51", 9: "13:39:44", 10: "8:18:17", 11: "8:18:20", 12: "8:20:01"},
    "mw-assembly": {1: "23:37:49", 2: "23:26:40", 3: "1 day, 0:50:30", 4: "17:39:30", 5: "18:42:24", 6: "19:32:10", 7: "13:24:10", 8: "13:17:19", 9: "14:01:45", 10: "7:24:08", 11: "8:12:11", 12: "8:28:53"},
}


def _parse_duration_to_hours(text: str) -> float:
    if "day" in text:
        day_part, hhmmss = text.split(",")
        days = int(day_part.split()[0])
        h, m, s = [int(x) for x in hhmmss.strip().split(":")]
    else:
        days = 0
        h, m, s = [int(x) for x in text.strip().split(":")]
    # convert to per-100K-step runtime (total runtime / 5) in hours
    return timedelta(days=days, hours=h, minutes=m, seconds=s).total_seconds() / 3600.0 / 5.0


def _runtime_mean_std_hours() -> tuple[np.ndarray, np.ndarray]:
    vals_by_diff: Dict[int, List[float]] = {d: [] for d in DIFFUSION_LEVELS}
    for task in TASKS:
        for seed, txt in RUNTIME_TEXT[task].items():
            vals_by_diff[SEED_TO_DIFFUSION[seed]].append(_parse_duration_to_hours(txt))
    means = np.array([np.mean(vals_by_diff[d]) for d in PLOT_ORDER], dtype=float)
    stds = np.array([np.std(vals_by_diff[d], ddof=1) for d in PLOT_ORDER], dtype=float)
    return means, stds


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
        if len(out) > 0 and np.isnan(out[0]):
            out[0] = 0.0
        return out
    return series.reindex(step_grid.astype(float)).interpolate(method="index", limit_area="inside").to_numpy(dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, default=Path("/media/datasets/cheliu21/cxy_worldmodel/online_ablation_csv"))
    parser.add_argument("--out", type=Path, default=Path("figures/Diffusion_ablation.pdf"))
    return parser.parse_args()


def prettify_task_name(task: str) -> str:
    if task == "mw-assembly":
        return "Assembly"
    return " ".join(piece.capitalize() for piece in task.split("-"))


def _extract_seed(path: Path) -> int | None:
    m = re.search(r"_(\d+)\.csv$", path.name)
    return int(m.group(1)) if m else None


def _apply_task_scaling(task: str, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if task.startswith(DMC_TASK_PREFIXES):
        out["reward"] = out["reward"] / 10.0
    elif task.startswith(METAWORLD_TASK_PREFIX):
        out["reward"] = out["reward"] * 100.0
    return out


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
        raise ValueError(f"{path} missing required columns")
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


def load_task_group_curves(task: str, csv_root: Path, step_grid: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
    files = sorted(csv_root.glob(f"{task}_*.csv"), key=lambda p: (_extract_seed(p) or 10**9, p.name))
    grouped: Dict[int, List[np.ndarray]] = {k: [] for k in DIFFUSION_LEVELS}
    for path in files:
        seed = _extract_seed(path)
        if seed is None or seed not in SEED_TO_DIFFUSION:
            continue
        grouped[SEED_TO_DIFFUSION[seed]].append(_align_curve(task, _load_single_csv(path, task), step_grid))
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for diff in DIFFUSION_LEVELS:
        if not grouped[diff]:
            nan = np.full_like(step_grid, np.nan, dtype=float)
            out[diff] = {"mean": nan, "ci": nan}
        else:
            mean, ci = summarize_ci(grouped[diff])
            out[diff] = {"mean": mean, "ci": ci}
    return out


def plot_all(args: argparse.Namespace) -> None:
    max_step_seen = 0
    for task in TASKS:
        for fp in args.csv_root.glob(f"{task}_*.csv"):
            try:
                raw = pd.read_csv(fp, usecols=["step"])
                if not raw.empty:
                    max_step_seen = max(max_step_seen, int(pd.to_numeric(raw["step"], errors="coerce").max()))
            except Exception:
                pass

    plot_x_max = int(np.ceil(max(100_000, min(X_MAX, max_step_seen)) / GRID_STEP) * GRID_STEP)
    step_grid = np.arange(0, plot_x_max + GRID_STEP, GRID_STEP, dtype=int)

    fig = plt.figure(figsize=(16, 7), dpi=300)
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1.05, 1, 1, 1])

    # ---------------------------------------------------------------------
    # Runtime subplot: bar plot + line plot
    # Bar: mean only
    # Line: mean with std error bars
    # ---------------------------------------------------------------------
    bar_ax = fig.add_subplot(gs[:, 0])
    means, stds = _runtime_mean_std_hours()

    # Smaller spacing between bars. Increase this value if you want bars farther apart.
    x = np.arange(len(PLOT_ORDER), dtype=float) * 0.65

    bar_colors = [COLORS[d] for d in PLOT_ORDER]

    bar_ax.set_axisbelow(True)
    bar_ax.grid(
        True,
        axis="y",
        linestyle="-",
        linewidth=0.8,
        alpha=0.25,
        zorder=0,
    )

    # Bar plot: only mean values.
    bar_ax.bar(
        x,
        means,
        width=0.24,
        color=bar_colors,
        alpha=0.55,
        edgecolor="none",
        zorder=1,
    )

    # # Line plot: connect mean values.
    # bar_ax.plot(
    #     x,
    #     means,
    #     color="#5f5f5f",
    #     linewidth=2.2,
    #     alpha=0.75,
    #     zorder=3,
    # )

    # Colored points and std error bars.
    for xi, mean, std, color in zip(x, means, stds, bar_colors):
        bar_ax.errorbar(
            xi,
            mean,
            yerr=std,
            fmt="o",
            markersize=7,
            markerfacecolor=mcolors.to_rgba(color, 0.90),
            markeredgecolor=mcolors.to_rgba("#222222", 0.65),
            markeredgewidth=1.2,
            ecolor=mcolors.to_rgba("black", 0.65),
            elinewidth=1.8,
            capsize=4.5,
            capthick=1.8,
            zorder=4,
        )

    # Optional value labels above std error bars.
    label_offset = 0.08
    for xi, mean, std in zip(x, means, stds):
        bar_ax.text(
            xi,
            mean + std + label_offset,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            color="#333333",
            zorder=5,
            clip_on=False,
        )

    bar_ax.set_title("Runtime", fontsize=22)
    bar_ax.set_ylabel("Hours / 100K", fontsize=20)

    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels([str(d) for d in PLOT_ORDER], fontsize=18)

    bar_ax.tick_params(axis="y", labelsize=20)
    bar_ax.tick_params(axis="x", labelsize=18, width=1.2, length=5)

    bar_ax.spines["top"].set_visible(False)
    bar_ax.spines["right"].set_visible(False)
    bar_ax.spines["left"].set_linewidth(1.3)
    bar_ax.spines["bottom"].set_linewidth(1.3)

    bar_ax.set_xlim(x[0] - 0.35, x[-1] + 0.35)
    bar_ax.set_ylim(0, float(np.max(means + stds) * 1.16))

    # ---------------------------------------------------------------------
    # Six task curve subplots
    # ---------------------------------------------------------------------
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(1, 4)]
    legend_handle_map = {}

    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        stats_by_diff = load_task_group_curves(task, args.csv_root, step_grid)

        for diff in PLOT_ORDER:
            mean, ci = stats_by_diff[diff]["mean"], stats_by_diff[diff]["ci"]
            color = COLORS[diff]

            ax.plot(
                step_grid,
                mean,
                color=color,
                linewidth=LINEWIDTH,
                alpha=MEAN_ALPHA,
            )
            ax.fill_between(
                step_grid,
                mean - ci,
                mean + ci,
                color=color,
                alpha=SHADE_ALPHA,
                linewidth=0,
            )

            if idx == 0:
                legend_handle_map[diff] = plt.Line2D(
                    [],
                    [],
                    color=color,
                    linewidth=LINEWIDTH,
                    alpha=MEAN_ALPHA,
                )

        ax.set_title(prettify_task_name(task), fontsize=22)
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
                ax.set_xticklabels([
                    "0",
                    "1M",
                    "2M" if plot_x_max >= 2_000_000 else f"{plot_x_max / 1_000_000:.1f}M",
                ])
            ax.tick_params(axis="x", labelsize=20, labelbottom=True)
        else:
            ax.tick_params(axis="x", labelbottom=False)

        if col == 0:
            ax.tick_params(axis="y", labelsize=20, labelleft=True)
        else:
            ax.tick_params(axis="y", labelleft=False)

    dummy_handle = plt.Line2D([], [], linestyle="none", linewidth=0)

    fig.legend(
        [dummy_handle] + [legend_handle_map[d] for d in [20, 15, 10, 5]],
        ["Denoise Steps"] + [str(d) for d in [20, 15, 10, 5]],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=False,
        fontsize=22,
        handlelength=1.5,
        columnspacing=1.4,
    )

    fig.subplots_adjust(
        left=0.05,
        right=0.995,
        top=0.93,
        bottom=0.20,
        wspace=0.30,
        hspace=0.35,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {args.out}")


if __name__ == "__main__":
    plot_all(parse_args())