#!/usr/bin/env python3
"""Plot DMControl single-task curves with mean and 95% CI over 3 seeds."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_config import load_plot_config

from handle_multi_csv import load_task_seed_csvs

DMCONTROL_TASKS: List[str] = [
    "acrobot-swingup",
    "cartpole-balance",
    "cartpole-balance-sparse",
    "cartpole-swingup",
    "cartpole-swingup-sparse",
    "cheetah-jump",
    "cheetah-run",
    "cheetah-run-back",
    "cheetah-run-backwards",
    "cheetah-run-front",
    "cup-catch",
    "cup-spin",
    "dog-run",
    "dog-trot",
    "dog-stand",
    "dog-walk",
    "finger-spin",
    "finger-turn-easy",
    "finger-turn-hard",
    "fish-swim",
    "hopper-hop",
    "hopper-hop-backwards",
    "hopper-stand",
    "humanoid-run",
    "humanoid-stand",
    "humanoid-walk",
    "pendulum-spin",
    "pendulum-swingup",
    "quadruped-run",
    "quadruped-walk",
    "reacher-easy",
    "reacher-hard",
    "reacher-three-easy",
    "reacher-three-hard",
    "walker-run",
    "walker-run-backwards",
    "walker-stand",
    "walker-walk",
    "walker-walk-backwards",
]

METHODS = ["tdmpc2", "tdmpc", "dreamerv3", "sac", "ours"]
METHOD_DIR = {
    "tdmpc2": "tdmpc2",
    "tdmpc": "tdmpc",
    "dreamerv3": "dreamerv3",
    "sac": "sac",
}

COLORS = {
    "ours": "#d64a4b",
    "tdmpc2": "#7fd54c",
    "tdmpc": "#5da7df",
    "dreamerv3": "#8a6bc7",
    "sac": "#5ad7c3",
}

DEFAULT_LABELS = {
    "ours": "MBDPO",
    "tdmpc2": "TD-MPC2",
    "tdmpc": "TD-MPC",
    "dreamerv3": "DreamerV3",
    "sac": "SAC",
}

X_MAX = 4_000_000
Y_MIN, Y_MAX = -10, 1010
GRID_STEP = 100_000
EXPECTED_SEEDS = [1, 2, 3]
PLOT_CFG = load_plot_config()
FINAL_CSV_DIR = Path("/media/datasets/cheliu21/cxy_worldmodel/final_csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("/root/workspace/tdmpc2/results"),
        help="Root containing baseline folders (sac/dreamerv3/tdmpc/tdmpc2).",
    )
    parser.add_argument(
        "--ours-root",
        type=Path,
        default=Path("/media/datasets/cheliu21/cxy_worldmodel/online_csv"),
        help="Root containing ours CSVs named {task}_{seed}.csv.",
    )
    parser.add_argument(
        "--ours-legend",
        type=str,
        default="MBDPO",
        help="Legend label for the improved TD-MPC2 baseline.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/DMControl.pdf"),
        help="Output pdf path.",
    )
    parser.add_argument(
        "--seed-config",
        type=Path,
        default=Path("tools/dmcontrol/seed.yaml"),
        help="Task-to-seeds config for Ours method.",
    )
    return parser.parse_args()


def prettify_task_name(task: str) -> str:
    return " ".join(word.capitalize() for word in task.replace("-", " ").split())


def _read_baseline_task_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["step", "reward", "seed"])
    df = pd.read_csv(path)
    required = {"step", "reward", "seed"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing columns {required}")
    return df[["step", "reward", "seed"]].copy()


def _clean_curve_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ["step", "reward", "seed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["step", "reward", "seed"])
    df = df[(df["step"] >= 0) & (df["step"] <= X_MAX)].copy()
    df["step"] = df["step"].astype(int)
    df["seed"] = df["seed"].astype(int)
    df = df.sort_values(["seed", "step"]).drop_duplicates(["seed", "step"], keep="last")

    # Enforce every existing seed curve has (step=0, reward=0.0).
    # - If step=0 exists, force reward to 0.0.
    # - If step=0 is missing, append (0, 0.0) for that seed.
    df.loc[df["step"] == 0, "reward"] = 0.0
    missing_seed0 = set(df["seed"].unique()) - set(df.loc[df["step"] == 0, "seed"].unique())
    if missing_seed0:
        pad_rows = pd.DataFrame(
            [{"step": 0, "reward": 0.0, "seed": seed} for seed in sorted(missing_seed0)]
        )
        df = pd.concat([df, pad_rows], ignore_index=True)
    df = df.sort_values(["seed", "step"]).drop_duplicates(["seed", "step"], keep="last")

    return df


def _extract_seed_list(raw: str) -> List[int]:
    return [int(x) for x in re.findall(r"\d+", raw)]


def load_ours_seed_config(path: Path, tasks: List[str]) -> Dict[str, List[int]]:
    task_to_seeds = {task: list(EXPECTED_SEEDS) for task in tasks}
    if not path.exists():
        return task_to_seeds

    current_task = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if raw_line and not raw_line.startswith((" ", "\t")) and line.endswith(":"):
            current_task = line[:-1].strip()
            continue
        if current_task is None:
            continue
        if "seed" in line:
            parsed = _extract_seed_list(line)
            if parsed:
                task_to_seeds[current_task] = parsed
            continue
        if line.startswith("-"):
            parsed = _extract_seed_list(line)
            if parsed:
                task_to_seeds[current_task] = parsed
    return task_to_seeds


def load_method_task_data(
    method: str,
    task: str,
    baseline_root: Path,
    ours_root: Path,
    ours_seeds: List[int],
) -> pd.DataFrame:
    if method == "ours":
        df = load_task_seed_csvs(
            root=ours_root,
            task=task,
            seeds=ours_seeds,
            step_bucket=GRID_STEP,
        )
    else:
        path = baseline_root / METHOD_DIR[method] / f"{task}.csv"
        df = _read_baseline_task_csv(path)
    return _clean_curve_df(df)


def summarize_mean_ci(df: pd.DataFrame, step_grid: np.ndarray, seeds: List[int]) -> Dict[str, np.ndarray]:
    if not seeds:
        nan_arr = np.full_like(step_grid, np.nan, dtype=float)
        return {"mean": nan_arr, "ci95": nan_arr}
    seed_curves = []
    for seed in seeds:
        sdf = df[df["seed"] == seed].sort_values("step")
        if sdf.empty:
            seed_curves.append(np.full_like(step_grid, np.nan, dtype=float))
            continue
        x = sdf["step"].to_numpy(dtype=float)
        y = sdf["reward"].to_numpy(dtype=float)
        series = pd.Series(y, index=x)
        aligned = series.reindex(step_grid.astype(float)).interpolate(method="index", limit_area="inside")
        seed_curves.append(aligned.to_numpy(dtype=float))

    curves = np.vstack(seed_curves)
    n = np.sum(np.isfinite(curves), axis=0)
    mean = np.nanmean(curves, axis=0)

    std = np.full_like(mean, np.nan, dtype=float)
    valid_for_std = n >= 2
    if np.any(valid_for_std):
        std_vals = np.nanstd(curves[:, valid_for_std], axis=0, ddof=1)
        std[valid_for_std] = std_vals

    ci95 = np.full_like(mean, np.nan, dtype=float)
    valid_for_ci = n > 0
    ci95[valid_for_ci] = 1.96 * np.nan_to_num(std[valid_for_ci], nan=0.0) / np.sqrt(n[valid_for_ci])
    return {"mean": mean, "ci95": ci95}



def export_task_final_csv(task: str, df: pd.DataFrame, seed_list: List[int], x_max: int) -> None:
    chosen_seeds = list(seed_list)[:3]
    if not chosen_seeds:
        return
    if len(chosen_seeds) == 2:
        print(f"[final_csv] task with 2 seeds: {task}")

    step_grid = np.arange(0, x_max + GRID_STEP, GRID_STEP, dtype=int)
    rows = []
    for mapped_seed, orig_seed in enumerate(chosen_seeds, start=1):
        sdf = df[df["seed"] == orig_seed].sort_values("step")
        if sdf.empty:
            continue
        series = pd.Series(
            sdf["reward"].to_numpy(dtype=float),
            index=sdf["step"].to_numpy(dtype=float),
        )
        aligned = series.reindex(step_grid.astype(float)).interpolate(method="index", limit_area="inside")
        for step, reward in zip(step_grid, aligned.to_numpy(dtype=float)):
            if not np.isfinite(reward):
                continue
            rows.append({"step": int(step), "reward": round(float(reward), 1), "seed": mapped_seed})

    out_df = pd.DataFrame(rows, columns=["step", "reward", "seed"])
    FINAL_CSV_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FINAL_CSV_DIR / f"{task}.csv"
    out_df.to_csv(out_path, index=False)

def plot_all(args: argparse.Namespace) -> None:
    labels = dict(DEFAULT_LABELS)
    labels["ours"] = args.ours_legend
    ours_seed_config = load_ours_seed_config(args.seed_config, DMCONTROL_TASKS)

    step_grid = np.arange(0, X_MAX + GRID_STEP, GRID_STEP, dtype=int)

    fig, axes = plt.subplots(8, 5, figsize=(22, 26), sharex=True, sharey=True)
    axes = axes.flatten()

    legend_handles = []
    for idx, task in enumerate(DMCONTROL_TASKS):
        ax = axes[idx]
        for method in METHODS:
            seed_list = ours_seed_config[task] if method == "ours" else EXPECTED_SEEDS
            df = load_method_task_data(method, task, args.baseline_root, args.ours_root, seed_list)
            stats = summarize_mean_ci(df, step_grid, seed_list)
            mean = stats["mean"]
            ci = stats["ci95"]
            upper = mean + ci
            lower = mean - ci
            color = COLORS[method]
            mean_alpha = 1.0 if method == "ours" else 0.55
            ci_alpha = float(PLOT_CFG["ci_alpha"])
            fill_alpha = 0.18 if method == "ours" else 0.15

            line_width = float(PLOT_CFG["subplot_ours_linewidth"]) if method == "ours" else float(PLOT_CFG["subplot_baseline_linewidth"])
            line, = ax.plot(step_grid, mean, color=color, linewidth=line_width, alpha=mean_alpha)
            ax.plot(step_grid, upper, color=color, linewidth=1.0, alpha=ci_alpha)
            ax.plot(step_grid, lower, color=color, linewidth=1.0, alpha=ci_alpha)
            ax.fill_between(step_grid, lower, upper, color=color, alpha=fill_alpha, linewidth=0)

            if method == "ours":
                export_task_final_csv(task, df, seed_list, X_MAX)

            if idx == 0:
                legend_handles.append(line)

        ax.set_title(prettify_task_name(task), fontsize=int(PLOT_CFG["dm_meta_title_fontsize"]))
        ax.set_xlim(-100, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.grid(True, linestyle="-", linewidth=0.8, alpha=0.25)

        row, col = divmod(idx, 5)
        ax.set_xticks([0, 1_000_000, 2_000_000, 3_000_000, 4_000_000])
        ax.set_yticks([0, 500, 1000])

        if row == 7:
            ax.set_xticklabels(["0", "1M", "2M", "3M", "4M"])
            ax.tick_params(axis="x", labelsize=int(PLOT_CFG["dm_meta_xtick_labelsize"]), labelbottom=True)
        else:
            ax.tick_params(axis="x", labelbottom=False)

        if col == 0:
            ax.tick_params(axis="y", labelsize=int(PLOT_CFG["dm_meta_ytick_labelsize"]), labelleft=True)
        else:
            ax.tick_params(axis="y", labelleft=False)

    # 8x5 grid has one extra subplot (40th). Hide it.
    axes[-1].axis("off")

    for _h in legend_handles:
        _h.set_linewidth(float(PLOT_CFG["legend_method_linewidth"]))

    fig.legend(
        legend_handles,
        [labels[m] for m in METHODS],
        loc="lower center",
        bbox_to_anchor=(0.5, float(PLOT_CFG["dm_meta_legend_y"])),
        ncol=5,
        frameon=False,
        fontsize=int(PLOT_CFG["legend_fontsize"]),
    )

    fig.subplots_adjust(left=0.06, right=0.995, top=0.96, bottom=0.085, wspace=0.15, hspace=0.4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {args.out}")


def main() -> None:
    args = parse_args()
    plot_all(args)


if __name__ == "__main__":
    main()
