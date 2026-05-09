#!/usr/bin/env python3
"""Plot off2on finetuned-vs-from-scratch curves (mean ±95% CI) for 10 tasks."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_config import load_plot_config


TASK_ORDER: List[Tuple[str, str]] = [
    ("cheetah-run", "Cheetah Run"),
    ("hopper-hop", "Hopper Hop"),
    ("mw-bin-picking", "Bin Picking"),
    ("mw-box-close", "Box Close"),
    ("mw-door-lock", "Door Lock"),
    ("mw-door-unlock", "Door Unlock"),
    ("mw-hand-insert", "Hand Insert"),
    ("pendulum-swingup", "Pendulum Swingup"),
    ("reacher-hard", "Reacher Hard"),
    ("walker-run", "Walker Run"),
]
DMCONTROL_TASKS = {"cheetah-run", "hopper-hop", "pendulum-swingup", "reacher-hard", "walker-run"}

PLOT_CFG = load_plot_config()
FINETUNED_COLOR = "#d62728"  # red
SCRATCH_COLOR = str(PLOT_CFG["off2on_scratch_color"])
DEFAULT_SCRATCH_SEEDS = [1, 2, 3]
DEFAULT_FINETUNE_SEEDS = [4, 5, 6]
OUT_PATH = Path("figures/off2on_compare.pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finetuned-root",
        type=Path,
        default=Path("/media/datasets/cheliu21/cxy_worldmodel/off2on_outputs"),
        help="Root dir containing finetuned off2on outputs ({task}/eval_{seed}.csv).",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path("/media/datasets/cheliu21/cxy_worldmodel/off2on_outputs/no_load"),
        help="Root dir containing from-scratch (no-load) outputs ({task}/eval_{seed}.csv).",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=40_000,
        help="Max x-axis step for plotting.",
    )
    parser.add_argument(
        "--step-interval",
        type=int,
        default=1_000,
        help="Step grid interval for interpolation/alignment.",
    )
    parser.add_argument(
        "--plot-scratch",
        action="store_true",
        default=True,
        help="Whether to plot from-scratch(no_load) curves. Default: enabled.",
    )
    parser.add_argument(
        "--seed-config",
        type=Path,
        default=Path("tools/off2on/seed.yaml"),
        help="Task-to-seeds config yaml-like file.",
    )
    return parser.parse_args()


def _extract_seed_list(raw: str) -> List[int]:
    return [int(x) for x in re.findall(r"\d+", raw)]


def load_seed_config(path: Path, tasks: List[str]) -> Dict[str, Dict[str, List[int]]]:
    task_to_seeds = {
        task: {"scratch_seed": list(DEFAULT_SCRATCH_SEEDS), "finetune_seed": list(DEFAULT_FINETUNE_SEEDS)}
        for task in tasks
    }
    if not path.exists():
        return task_to_seeds

    current_task = None
    current_key = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if raw_line and not raw_line.startswith((" ", "\t")) and line.endswith(":"):
            current_task = line[:-1].strip()
            current_key = None
            continue
        if current_task is None:
            continue
        if line.startswith("scratch_seed:"):
            current_key = "scratch_seed"
            parsed = _extract_seed_list(line)
            if parsed and current_task in task_to_seeds:
                task_to_seeds[current_task][current_key] = parsed
            continue
        if line.startswith("finetune_seed:"):
            current_key = "finetune_seed"
            parsed = _extract_seed_list(line)
            if parsed and current_task in task_to_seeds:
                task_to_seeds[current_task][current_key] = parsed
            continue
        if "seed" in line and current_key is None:
            parsed = _extract_seed_list(line)
            if parsed and current_task in task_to_seeds:
                task_to_seeds[current_task]["scratch_seed"] = parsed
                task_to_seeds[current_task]["finetune_seed"] = parsed
            continue
        if line.startswith("-"):
            parsed = _extract_seed_list(line)
            if parsed and current_task in task_to_seeds:
                if current_key in ("scratch_seed", "finetune_seed"):
                    task_to_seeds[current_task][current_key] = parsed
                else:
                    task_to_seeds[current_task]["scratch_seed"] = parsed
                    task_to_seeds[current_task]["finetune_seed"] = parsed
    return task_to_seeds


def _read_eval_csv(path: Path, task: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["step", "reward"])
    df = pd.read_csv(path)
    if "step" not in df.columns:
        raise ValueError(f"{path} missing required column: step")

    # Meta-World should plot success rate, DMControl should plot episode reward.
    if task.startswith("mw-"):
        if "episode_success" not in df.columns:
            raise ValueError(f"{path} missing required column for Meta-World task: episode_success")
        out = df[["step", "episode_success"]].rename(columns={"episode_success": "reward"}).copy()
    else:
        if "episode_reward" not in df.columns:
            raise ValueError(f"{path} missing required column for DMControl task: episode_reward")
        out = df[["step", "episode_reward"]].rename(columns={"episode_reward": "reward"}).copy()

    out["step"] = pd.to_numeric(out["step"], errors="coerce")
    out["reward"] = pd.to_numeric(out["reward"], errors="coerce")
    out = out.dropna(subset=["step", "reward"]).sort_values("step")
    out["step"] = out["step"].astype(int)
    return out.drop_duplicates("step", keep="last")


def _normalize_reward(task: str, reward: np.ndarray) -> np.ndarray:
    # DMControl: normalize score to percentage by score / 1000 * 100.
    if task in DMCONTROL_TASKS:
        return reward / 10.0
    # Meta-World: success rate. Convert [0,1] to [0,100] when needed.
    if np.nanmax(np.abs(reward)) <= 1.5:
        return reward * 100.0
    return reward


def _load_task_seed_curves(root: Path, task: str, seeds: List[int], step_grid: np.ndarray) -> np.ndarray:
    curves = []
    for seed in seeds:
        fp = root / task / f"eval_{seed}.csv"
        df = _read_eval_csv(fp, task=task)
        if df.empty:
            curves.append(np.full_like(step_grid, np.nan, dtype=float))
            continue
        reward = _normalize_reward(task, df["reward"].to_numpy(dtype=float))
        series = pd.Series(reward, index=df["step"].to_numpy(dtype=float))
        aligned = series.reindex(step_grid.astype(float)).interpolate(method="index", limit_area="inside")
        curves.append(aligned.to_numpy(dtype=float))
    if not curves:
        return np.empty((0, len(step_grid)), dtype=float)
    return np.vstack(curves)


def _mean_ci95(curves: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if curves.size == 0:
        return np.array([]), np.array([])
    n = np.sum(np.isfinite(curves), axis=0)
    mean = np.nanmean(curves, axis=0)

    std = np.full_like(mean, np.nan, dtype=float)
    valid_std = n >= 2
    if np.any(valid_std):
        std[valid_std] = np.nanstd(curves[:, valid_std], axis=0, ddof=1)

    ci95 = np.full_like(mean, np.nan, dtype=float)
    valid = n > 0
    ci95[valid] = 1.96 * np.nan_to_num(std[valid], nan=0.0) / np.sqrt(n[valid])
    return mean, ci95


def _task_data(
    finetuned_root: Path,
    scratch_root: Path,
    task: str,
    step_grid: np.ndarray,
    finetune_seeds: List[int],
    scratch_seeds: List[int],
) -> Dict[str, np.ndarray]:
    ft_curves = _load_task_seed_curves(finetuned_root, task, finetune_seeds, step_grid)
    sc_curves = _load_task_seed_curves(scratch_root, task, scratch_seeds, step_grid)

    ft_mean, ft_ci = _mean_ci95(ft_curves)
    sc_mean, sc_ci = _mean_ci95(sc_curves)
    return {
        "ft_mean": ft_mean,
        "ft_ci": ft_ci,
        "sc_mean": sc_mean,
        "sc_ci": sc_ci,
    }


def _last_finite(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    valid_idx = np.where(np.isfinite(arr))[0]
    if len(valid_idx) == 0:
        return float("nan")
    return float(arr[valid_idx[-1]])


def _print_improvement_report(task_order: List[Tuple[str, str]], all_stats: Dict[str, Dict[str, np.ndarray]]) -> None:
    print("\n=== Off2On Improvement Report (final-step mean) ===")
    print("Definition: improvement_rate = (Finetuned - FromScratch) / |FromScratch| * 100%")
    print("-" * 88)
    print(f"{'Task':<20} {'Finetuned':>12} {'FromScratch':>12} {'AbsDelta':>12} {'Improve%':>12}")
    print("-" * 88)

    valid_rates = []
    for task, title in task_order:
        ft_final = _last_finite(all_stats[task]["ft_mean"])
        sc_final = _last_finite(all_stats[task]["sc_mean"])
        abs_delta = ft_final - sc_final if np.isfinite(ft_final) and np.isfinite(sc_final) else float("nan")
        if not np.isfinite(ft_final) or not np.isfinite(sc_final):
            rate = float("nan")
        elif abs(sc_final) < 1e-8:
            rate = float("nan")
        else:
            rate = (ft_final - sc_final) / abs(sc_final) * 100.0

        if np.isfinite(rate):
            valid_rates.append(rate)

        def _fmt(x: float) -> str:
            return f"{x:,.2f}" if np.isfinite(x) else "nan"

        print(f"{title:<20} {_fmt(ft_final):>12} {_fmt(sc_final):>12} {_fmt(abs_delta):>12} {_fmt(rate):>12}")

    print("-" * 88)
    avg_rate = float(np.mean(valid_rates)) if valid_rates else float("nan")
    print(f"{'Average Improve%':<58} {avg_rate:>12.2f}" if np.isfinite(avg_rate) else f"{'Average Improve%':<58} {'nan':>12}")
    print("=" * 88)


def plot(args: argparse.Namespace) -> None:
    step_grid = np.arange(0, args.max_step + args.step_interval, args.step_interval, dtype=int)
    task_seed_cfg = load_seed_config(args.seed_config, [t for t, _ in TASK_ORDER])

    all_stats: Dict[str, Dict[str, np.ndarray]] = {}
    for task, _ in TASK_ORDER:
        seed_cfg = task_seed_cfg.get(
            task,
            {"scratch_seed": list(DEFAULT_SCRATCH_SEEDS), "finetune_seed": list(DEFAULT_FINETUNE_SEEDS)},
        )
        stats = _task_data(
            args.finetuned_root,
            args.scratch_root,
            task,
            step_grid,
            seed_cfg["finetune_seed"],
            seed_cfg["scratch_seed"],
        )
        all_stats[task] = stats

    _print_improvement_report(TASK_ORDER, all_stats)

    plt.style.use("ggplot")
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey=True)
    axes = axes.flatten()

    legend_handles = []
    for idx, (task, title) in enumerate(TASK_ORDER):
        ax = axes[idx]
        stats = all_stats[task]

        ft_mean = stats["ft_mean"]
        ft_ci = stats["ft_ci"]
        sc_mean = stats["sc_mean"]
        sc_ci = stats["sc_ci"]

        # Match transparency style with tools/plot_DMcontrol.py
        ft_mean_alpha, ft_ci_alpha, ft_fill_alpha = 1.0, 0.55, 0.22
        sc_mean_alpha, sc_ci_alpha, sc_fill_alpha = 0.55, float(PLOT_CFG["ci_alpha"]), 0.15

        ft_upper, ft_lower = ft_mean + ft_ci, ft_mean - ft_ci

        line_sc = None
        if args.plot_scratch:
            sc_upper, sc_lower = sc_mean + sc_ci, sc_mean - sc_ci
            line_sc, = ax.plot(step_grid, sc_mean, color=SCRATCH_COLOR, linewidth=2, alpha=sc_mean_alpha)
            ax.fill_between(step_grid, sc_lower, sc_upper, color=SCRATCH_COLOR, alpha=sc_fill_alpha, linewidth=0)

        line_ft, = ax.plot(step_grid, ft_mean, color=FINETUNED_COLOR, linewidth=2.2, alpha=ft_mean_alpha)
        ax.fill_between(step_grid, ft_lower, ft_upper, color=FINETUNED_COLOR, alpha=ft_fill_alpha, linewidth=0)

        if idx == 0:
            if line_sc is None:
                legend_handles = [Line2D([], [], color=FINETUNED_COLOR, linewidth=float(PLOT_CFG["legend_method_linewidth"]), alpha=ft_mean_alpha)]
            else:
                legend_handles = [
                    Line2D([], [], color=SCRATCH_COLOR, linewidth=float(PLOT_CFG["legend_method_linewidth"]), alpha=sc_mean_alpha),
                    Line2D([], [], color=FINETUNED_COLOR, linewidth=float(PLOT_CFG["legend_method_linewidth"]), alpha=ft_mean_alpha),
                ]

        ax.set_title(title, fontsize=int(PLOT_CFG["off2on_title_fontsize"]))
        ax.set_xlim(-10, args.max_step + 10)
        ax.set_ylim(-5.0, 105.0)
        ax.set_facecolor(str(PLOT_CFG["off2on_bg_color"])); ax.grid(True, linestyle="-", color=str(PLOT_CFG["off2on_grid_color"]), linewidth=0.8, alpha=0.55)
        ax.set_yticks([0, 50, 100])

        row, col = divmod(idx, 5)
        if row == 1:
            xticks = [0, args.max_step // 2, args.max_step]
            ax.set_xticks(xticks)
            ax.set_xticklabels(["0", f"{args.max_step//2000}k", f"{args.max_step//1000}k"])
            ax.tick_params(axis="x", labelsize=int(PLOT_CFG["off2on_xtick_labelsize"]), labelbottom=True)
        else:
            ax.tick_params(axis="x", labelbottom=False)

        ax.tick_params(axis="y", labelsize=int(PLOT_CFG["ytick_labelsize"]))

    legend_labels = ["Finetuned"] if not args.plot_scratch else ["From scratch", "Finetuned"]
    for _h in legend_handles:
        _h.set_linewidth(float(PLOT_CFG["legend_method_linewidth"]))

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(legend_labels),
        frameon=False,
        fontsize=int(PLOT_CFG["off2on_legend_fontsize"]),
    )

    fig.subplots_adjust(left=0.06, right=0.995, top=0.93, bottom=0.18, wspace=0.15, hspace=0.38)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {OUT_PATH}")


def main() -> None:
    args = parse_args()
    plot(args)


if __name__ == "__main__":
    main()
