#!/usr/bin/env python3
"""Plot overall multi-domain aggregate curves with mean and 95% CI over 3 aggregate seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from plot_config import load_plot_config

METHODS = ["ours", "tdmpc2", "tdmpc", "dreamerv3", "sac"]
DRAW_ORDER = ["tdmpc2", "tdmpc", "dreamerv3", "sac", "ours"]
METHOD_DIR = {"tdmpc2": "tdmpc2", "tdmpc": "tdmpc", "dreamerv3": "dreamerv3", "sac": "sac"}
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

DOMAIN_SPECS = {
    "DMControl": {
        "tasks": [
            "acrobot-swingup","cartpole-balance","cartpole-balance-sparse","cartpole-swingup","cartpole-swingup-sparse","cheetah-jump","cheetah-run","cheetah-run-back","cheetah-run-backwards","cheetah-run-front","cup-catch","cup-spin","dog-run","dog-trot","dog-stand","dog-walk","finger-spin","finger-turn-easy","finger-turn-hard","fish-swim","hopper-hop","hopper-hop-backwards","hopper-stand","humanoid-run","humanoid-stand","humanoid-walk","pendulum-spin","pendulum-swingup","quadruped-run","quadruped-walk","reacher-easy","reacher-hard","reacher-three-easy","reacher-three-hard","walker-run","walker-run-backwards","walker-stand","walker-walk","walker-walk-backwards",
        ],
        "x_max": 4_000_000,
        "y_lim": (-20, 1020),
    },
    "Meta-World": {
        "tasks": [
            "mw-assembly","mw-basketball","mw-bin-picking","mw-box-close","mw-button-press-topdown-wall","mw-button-press-topdown","mw-button-press-wall","mw-button-press","mw-coffee-button","mw-coffee-pull","mw-coffee-push","mw-dial-turn","mw-disassemble","mw-door-close","mw-door-lock","mw-door-open","mw-door-unlock","mw-drawer-close","mw-drawer-open","mw-faucet-close","mw-faucet-open","mw-hammer","mw-hand-insert","mw-handle-press-side","mw-handle-press","mw-handle-pull-side","mw-handle-pull","mw-lever-pull","mw-peg-insert-side","mw-peg-unplug-side","mw-pick-out-of-hole","mw-pick-place-wall","mw-pick-place","mw-plate-slide-back-side","mw-plate-slide-back","mw-plate-slide-side","mw-plate-slide","mw-push-back","mw-push-wall","mw-push","mw-reach-wall","mw-reach","mw-shelf-place","mw-soccer","mw-stick-pull","mw-stick-push","mw-sweep-into","mw-sweep","mw-window-close","mw-window-open",
        ],
        "x_max": 2_000_000,
        "y_lim": (-2, 102),
    },
    "ManiSkill2": {"tasks": ["lift-cube", "pick-cube", "pick-ycb", "stack-cube", "turn-faucet"], "x_max": 4_000_000, "y_lim": (-2, 102)},
    "MyoSuite": {"tasks": ["myo-hand-key-turn","myo-hand-key-turn-hard","myo-hand-obj-hold","myo-hand-obj-hold-hard","myo-hand-pen-twirl","myo-hand-pen-twirl-hard","myo-hand-pose","myo-hand-pose-hard","myo-hand-reach","myo-hand-reach-hard"], "x_max": 2_000_000, "y_lim": (-2, 102)},
    "Visual RL": {
        "tasks": [
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
        ],
        "x_max": 1_000_000,
        "y_lim": (-20, 1020),
        "methods": ["tdmpc2", "ours"],
        "baseline_dir": "tdmpc2-pixels",
        "force_zero_start_baseline": True,
    },
}

SEEDS = [1, 2, 3]
GRID_STEP = 100_000
OVERALL_PLOT_CFG = {
    "title_fontsize": 18,
    "subtitle_fontsize": 15,
    "legend_fontsize": 16,
    "xtick_labelsize": 14,
    "ytick_labelsize": 14,
    "subplot_ours_linewidth": 3.2,
    "subplot_baseline_linewidth": 2.0,
    "legend_method_linewidth": 4.0,
    "legend_y": -0.16,
}
TASK_ALIASES = {
    # Keep backward compatibility with previously used task names.
    "myo-hand-key-turn": ["myo-key-turn"],
    "myo-hand-key-turn-hard": ["myo-key-turn-hard"],
    "myo-hand-obj-hold": ["myo-obj-hold"],
    "myo-hand-obj-hold-hard": ["myo-obj-hold-hard"],
    "myo-hand-pen-twirl": ["myo-pen-twirl"],
    "myo-hand-pen-twirl-hard": ["myo-pen-twirl-hard"],
    "myo-hand-pose": ["myo-pose"],
    "myo-hand-pose-hard": ["myo-pose-hard"],
    "myo-hand-reach": ["myo-reach"],
    "myo-hand-reach-hard": ["myo-reach-hard"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-root", type=Path, default=Path("/root/workspace/tdmpc2/results"))
    p.add_argument("--ours-root", type=Path, default=Path("/media/datasets/cheliu21/cxy_worldmodel/final_csv"))
    p.add_argument("--out", type=Path, default=Path("figures/Overall.pdf"))
    return p.parse_args()


def read_csv(path: Path, force_zero_start: bool = False) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["step", "reward", "seed"])
    df = pd.read_csv(path)
    if {"step", "reward", "seed"}.issubset(df.columns):
        out = df[["step", "reward", "seed"]].copy()
    elif {"step", "success", "seed"}.issubset(df.columns):
        out = df[["step", "success", "seed"]].rename(columns={"success": "reward"}).copy()
        # Meta-World / ManiSkill2 / MyoSuite baselines are often stored as
        # success ratio in [0, 1]. Convert to percentage [0, 100] so the
        # domain-scale matches Ours and the target y-axis.
        max_abs = np.nanmax(np.abs(pd.to_numeric(out["reward"], errors="coerce")))
        if np.isfinite(max_abs) and max_abs <= 1.5:
            out["reward"] = pd.to_numeric(out["reward"], errors="coerce") * 100.0
    else:
        raise ValueError(f"{path} missing columns")
    for c in ["step", "reward", "seed"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["step", "reward", "seed"])
    out["step"] = (np.round(out["step"] / GRID_STEP) * GRID_STEP).astype(int)
    out["seed"] = out["seed"].astype(int)
    out = out.sort_values(["seed", "step"]).drop_duplicates(["seed", "step"], keep="last")
    if force_zero_start and not out.empty:
        out.loc[out["step"] == 0, "reward"] = 0.0
        missing_seed0 = set(out["seed"].unique()) - set(out.loc[out["step"] == 0, "seed"].unique())
        if missing_seed0:
            pad = pd.DataFrame([{"step": 0, "reward": 0.0, "seed": s} for s in sorted(missing_seed0)])
            out = pd.concat([out, pad], ignore_index=True)
            out = out.sort_values(["seed", "step"]).drop_duplicates(["seed", "step"], keep="last")
    return out


def seed_curve(df: pd.DataFrame, seed: int, step_grid: np.ndarray) -> np.ndarray:
    sdf = df[df.seed == seed]
    if sdf.empty:
        return np.full_like(step_grid, np.nan, dtype=float)
    x = sdf.step.to_numpy()
    y = sdf.reward.to_numpy()
    if x[0] > 0:
        x = np.insert(x, 0, 0)
        y = np.insert(y, 0, 0.0)
    return np.interp(step_grid, x, y)


def resolve_task_csv(root: Path, task: str) -> Path:
    candidates = [task, *TASK_ALIASES.get(task, [])]
    for name in candidates:
        p = root / f"{name}.csv"
        if p.exists():
            return p
    return root / f"{task}.csv"


def domain_method_stats(method: str, domain: str, spec: dict, baseline_root: Path, ours_root: Path):
    step_grid = np.arange(0, spec["x_max"] + 1, GRID_STEP)
    agg_seed_curves = []
    for seed in SEEDS:
        task_curves = []
        for task in spec["tasks"]:
            if method == "ours":
                path = resolve_task_csv(ours_root, task)
            else:
                baseline_dir = spec.get("baseline_dir", METHOD_DIR[method])
                path = resolve_task_csv(baseline_root / baseline_dir, task)
            force_zero = method != "ours" and (domain == "MyoSuite" or spec.get("force_zero_start_baseline", False))
            df = read_csv(path, force_zero_start=force_zero)
            c = seed_curve(df, seed, step_grid)
            if not np.all(np.isnan(c)):
                task_curves.append(c)
        if task_curves:
            agg_seed_curves.append(np.nanmean(np.stack(task_curves, axis=0), axis=0))
    if not agg_seed_curves:
        nan = np.full_like(step_grid, np.nan, dtype=float)
        return step_grid, nan, nan, nan
    S = np.stack(agg_seed_curves, axis=0)
    mean = np.nanmean(S, axis=0)
    std = np.nanstd(S, axis=0, ddof=0)
    ci = 1.96 * std / np.sqrt(S.shape[0])
    return step_grid, mean, mean - ci, mean + ci


def main() -> None:
    args = parse_args()
    if not args.baseline_root.exists():
        alt_root = Path("/workspace/tdmpc2/results")
        if alt_root.exists():
            print(f"[plot_overall] baseline-root not found: {args.baseline_root}; fallback to {alt_root}")
            args.baseline_root = alt_root
    if not args.baseline_root.exists():
        raise FileNotFoundError(
            f"baseline-root does not exist: {args.baseline_root}; "
            "please set --baseline-root to the folder containing sac/dreamerv3/tdmpc/tdmpc2"
        )
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": int(OVERALL_PLOT_CFG["title_fontsize"]),
            "axes.labelsize": 16,
            "legend.fontsize": int(OVERALL_PLOT_CFG["legend_fontsize"]),
        }
    )
    fig, axes = plt.subplots(1, len(DOMAIN_SPECS), figsize=(20, 4), dpi=200)
    for ax, (domain, spec) in zip(axes, DOMAIN_SPECS.items()):
        methods = spec.get("methods", DRAW_ORDER)
        for method in methods:
            x, m, lo, hi = domain_method_stats(method, domain, spec, args.baseline_root, args.ours_root)
            line_alpha = 0.95 if method == "ours" else 0.55
            line_width = float(OVERALL_PLOT_CFG["subplot_ours_linewidth"]) if method == "ours" else float(OVERALL_PLOT_CFG["subplot_baseline_linewidth"])
            ax.plot(x, m, color=COLORS[method], lw=line_width, label=DEFAULT_LABELS[method], alpha=line_alpha)
            ax.fill_between(x, lo, hi, color=COLORS[method], alpha=0.12, linewidth=0)
        # Two-line title: bold domain name, normal task count
        ax.text(
            0.5, 1.2, domain,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=int(OVERALL_PLOT_CFG["title_fontsize"]),
            fontweight="bold",
        )

        ax.text(
            0.5, 1.055, f"{len(spec['tasks'])} tasks",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=int(OVERALL_PLOT_CFG["subtitle_fontsize"]),
            fontweight="normal",
        )
        ax.set_xlim(0, spec["x_max"])
        ax.set_ylim(*spec["y_lim"])
        xt = np.arange(0, spec["x_max"] + 1, 1_000_000)
        ax.set_xticks(xt)
        ax.set_xticklabels(["0"] + [f"{int(v/1_000_000)}M" for v in xt[1:]])
        ax.tick_params(axis="x", labelsize=int(OVERALL_PLOT_CFG["xtick_labelsize"]))
        ax.tick_params(axis="y", labelsize=int(OVERALL_PLOT_CFG["ytick_labelsize"]))
        ax.grid(True, alpha=0.25)
    handles = [
        Line2D([], [], color=COLORS[m], linewidth=float(OVERALL_PLOT_CFG["legend_method_linewidth"]), alpha=0.95 if m == "ours" else 0.55)
        for m in DRAW_ORDER
    ]
    labels = [DEFAULT_LABELS[m] for m in DRAW_ORDER]
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, float(OVERALL_PLOT_CFG["legend_y"])))
    plt.tight_layout(rect=[0, 0.18, 1, 1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
