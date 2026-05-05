#!/usr/bin/env python3
"""Plot Visual benchmark curves with mean and 95% CI over seeds."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VISUAL_TASKS: List[str] = [
    "acrobot-swingup",
    "cheetah-run",
    "finger-spin",
    "finger-turn-easy",
    "finger-turn-hard",
    "quadruped-walk",
    "reacher-easy",
    "reacher-hard",
    "walker-run",
    "walker-walk",
]

METHODS = ["ours", "tdmpc2"]
COLORS = {
    "ours": "#d64a4b",
    "tdmpc2": "#7fd54c",
}

DEFAULT_LABELS = {
    "ours": "Ours",
    "tdmpc2": "TD-MPC2",
}

X_MAX = 1_000_000
Y_MIN, Y_MAX = -10, 1010
GRID_STEP = 100_000
OURS_DEFAULT_SEEDS = [7, 8, 9]
TDMPC2_SEEDS = [1, 2, 3]
FINAL_CSV_DIR = Path("/media/datasets/cheliu21/cxy_worldmodel/final_csv_pixels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tdmpc2-root",
        type=Path,
        default=Path("results/tdmpc2-pixels"),
        help="Root containing TD-MPC2 task csv files named {task}.csv.",
    )
    parser.add_argument(
        "--ours-root",
        type=Path,
        default=Path("/media/datasets/cheliu21/cxy_worldmodel/online_pixels_csv"),
        help="Root containing ours CSVs named {task}_{seed}.csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/Visual.pdf"),
        help="Output pdf path.",
    )
    parser.add_argument(
        "--seed-config",
        type=Path,
        default=Path("tools/visual/seed.yaml"),
        help="Task-to-seeds config for Ours method.",
    )
    return parser.parse_args()


def prettify_task_name(task: str) -> str:
    return " ".join(word.capitalize() for word in task.replace("-", " ").split())


def _extract_seed_list(raw: str) -> List[int]:
    return [int(x) for x in re.findall(r"\d+", raw)]


def load_ours_seed_config(path: Path, tasks: List[str]) -> Dict[str, List[int]]:
    task_to_seeds = {task: list(OURS_DEFAULT_SEEDS) for task in tasks}
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
        if "seed" in line or line.startswith("-"):
            parsed = _extract_seed_list(line)
            if parsed:
                task_to_seeds[current_task] = parsed
    return task_to_seeds


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

    df.loc[df["step"] == 0, "reward"] = 0.0
    missing_seed0 = set(df["seed"].unique()) - set(df.loc[df["step"] == 0, "seed"].unique())
    if missing_seed0:
        pad_rows = pd.DataFrame(
            [{"step": 0, "reward": 0.0, "seed": seed} for seed in sorted(missing_seed0)]
        )
        df = pd.concat([df, pad_rows], ignore_index=True)
    df = df.sort_values(["seed", "step"]).drop_duplicates(["seed", "step"], keep="last")
    return df


def _read_tdmpc2_task_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["step", "reward", "seed"])
    df = pd.read_csv(path)
    required = {"step", "reward", "seed"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing columns {required}")
    return df[["step", "reward", "seed"]].copy()


def _load_ours_task_csvs(root: Path, task: str, seeds: List[int]) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for seed in seeds:
        path = root / f"{task}_{seed}.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        reward_col = "reward" if "reward" in raw.columns else "episode_reward"
        if reward_col not in raw.columns or "step" not in raw.columns:
            raise ValueError(f"{path} must contain step and reward/episode_reward columns")
        norm = raw[["step", reward_col]].rename(columns={reward_col: "reward"}).copy()
        norm["seed"] = seed

        for col in ["step", "reward", "seed"]:
            norm[col] = pd.to_numeric(norm[col], errors="coerce")
        norm = norm.dropna(subset=["step", "reward", "seed"]) 
        norm = norm.sort_values("step").reset_index(drop=True)

        if norm.empty:
            continue

        steps = norm["step"].to_numpy(dtype=float)
        rewards = norm["reward"].to_numpy(dtype=float)

        target_steps = np.arange(0, X_MAX + GRID_STEP, GRID_STEP, dtype=int)
        if task == "walker-run":
            series = pd.Series(rewards, index=steps)
            aligned = series.reindex(target_steps.astype(float)).interpolate(
                method="index", limit_area="inside"
            )
            out_rows = [
                {"step": int(step), "reward": float(reward), "seed": int(seed)}
                for step, reward in zip(target_steps, aligned.to_numpy(dtype=float))
                if np.isfinite(reward)
            ]
        else:
            out_rows = []
            for target in target_steps:
                nearest_idx = np.argsort(np.abs(steps - target))[:100]
                if nearest_idx.size == 0:
                    continue
                pooled_reward = float(np.nanmax(rewards[nearest_idx]))
                out_rows.append({"step": int(target), "reward": pooled_reward, "seed": int(seed)})

        parts.append(pd.DataFrame(out_rows, columns=["step", "reward", "seed"]))

    if not parts:
        return pd.DataFrame(columns=["step", "reward", "seed"])
    return pd.concat(parts, ignore_index=True)


def load_method_task_data(
    method: str,
    task: str,
    tdmpc2_root: Path,
    ours_root: Path,
    ours_seeds: List[int],
) -> pd.DataFrame:
    if method == "ours":
        df = _load_ours_task_csvs(ours_root, task, ours_seeds)
    else:
        df = _read_tdmpc2_task_csv(tdmpc2_root / f"{task}.csv")
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
        print(f"[final_csv_pixels] task with 2 seeds: {task}")

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
    ours_seed_config = load_ours_seed_config(args.seed_config, VISUAL_TASKS)
    step_grid = np.arange(0, X_MAX + GRID_STEP, GRID_STEP, dtype=int)

    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    legend_handles = []
    for idx, task in enumerate(VISUAL_TASKS):
        ax = axes[idx]
        for method in METHODS:
            seed_list = ours_seed_config[task] if method == "ours" else TDMPC2_SEEDS
            df = load_method_task_data(method, task, args.tdmpc2_root, args.ours_root, seed_list)
            stats = summarize_mean_ci(df, step_grid, seed_list)
            mean = stats["mean"]
            ci = stats["ci95"]
            upper = mean + ci
            lower = mean - ci
            color = COLORS[method]
            mean_alpha = 1.0 if method == "ours" else 0.55
            ci_alpha = 0.30 if method == "ours" else 0.25
            fill_alpha = 0.18 if method == "ours" else 0.15
            line_width = 4 if method == "ours" else 2

            (line,) = ax.plot(step_grid, mean, color=color, linewidth=line_width, alpha=mean_alpha)
            ax.plot(step_grid, upper, color=color, linewidth=1.0, alpha=ci_alpha)
            ax.plot(step_grid, lower, color=color, linewidth=1.0, alpha=ci_alpha)
            ax.fill_between(step_grid, lower, upper, color=color, alpha=fill_alpha, linewidth=0)

            if method == "ours":
                export_task_final_csv(task, df, seed_list, X_MAX)

            if idx == 0:
                legend_handles.append(line)

        ax.set_title(prettify_task_name(task), fontsize=22)
        ax.set_xlim(-100, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.grid(True, linestyle="-", linewidth=0.8, alpha=0.25)

        row, col = divmod(idx, 5)
        ax.set_xticks([0, 500_000, 1_000_000])
        ax.set_yticks([0, 500, 1000])

        if row == 1:
            ax.set_xticklabels(["0", "0.5M", "1M"])
            ax.tick_params(axis="x", labelsize=18, labelbottom=True)
        else:
            ax.tick_params(axis="x", labelbottom=False)

        if col == 0:
            ax.tick_params(axis="y", labelsize=18, labelleft=True)
        else:
            ax.tick_params(axis="y", labelleft=False)

    fig.legend(
        legend_handles,
        [labels[m] for m in METHODS],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        fontsize=20,
    )

    fig.subplots_adjust(left=0.06, right=0.995, top=0.90, bottom=0.18, wspace=0.15, hspace=0.4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {args.out}")


def main() -> None:
    args = parse_args()
    plot_all(args)


if __name__ == "__main__":
    main()
