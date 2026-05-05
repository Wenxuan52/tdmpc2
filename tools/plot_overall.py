#!/usr/bin/env python3
"""Plot overall multi-domain aggregate curves with mean and 95% CI over 3 aggregate seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ["ours", "tdmpc2", "tdmpc", "dreamerv3", "sac"]
METHOD_DIR = {"tdmpc2": "tdmpc2", "tdmpc": "tdmpc", "dreamerv3": "dreamerv3", "sac": "sac"}
COLORS = {
    "ours": "#d64a4b",
    "tdmpc2": "#7fd54c",
    "tdmpc": "#5da7df",
    "dreamerv3": "#8a6bc7",
    "sac": "#5ad7c3",
}
DEFAULT_LABELS = {
    "ours": "Ours",
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
    "MyoSuite": {"tasks": ["myo-key-turn","myo-key-turn-hard","myo-obj-hold","myo-obj-hold-hard","myo-pen-twirl","myo-pen-twirl-hard","myo-pose","myo-pose-hard","myo-reach","myo-reach-hard"], "x_max": 2_000_000, "y_lim": (-2, 102)},
}

SEEDS = [1, 2, 3]
GRID_STEP = 100_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-root", type=Path, default=Path("/workspace/tdmpc2/results"))
    p.add_argument("--ours-root", type=Path, default=Path("/media/datasets/cheliu21/cxy_worldmodel/final_csv"))
    p.add_argument("--out", type=Path, default=Path("figures/Overall.pdf"))
    return p.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["step", "reward", "seed"])
    df = pd.read_csv(path)
    if {"step", "reward", "seed"}.issubset(df.columns):
        out = df[["step", "reward", "seed"]].copy()
    elif {"step", "success", "seed"}.issubset(df.columns):
        out = df[["step", "success", "seed"]].rename(columns={"success": "reward"}).copy()
    else:
        raise ValueError(f"{path} missing columns")
    for c in ["step", "reward", "seed"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["step", "reward", "seed"])
    out["step"] = (np.round(out["step"] / GRID_STEP) * GRID_STEP).astype(int)
    out["seed"] = out["seed"].astype(int)
    return out.sort_values(["seed", "step"]).drop_duplicates(["seed", "step"], keep="last")


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


def domain_method_stats(method: str, domain: str, spec: dict, baseline_root: Path, ours_root: Path):
    step_grid = np.arange(0, spec["x_max"] + 1, GRID_STEP)
    agg_seed_curves = []
    for seed in SEEDS:
        task_curves = []
        for task in spec["tasks"]:
            path = (ours_root / f"{task}.csv") if method == "ours" else (baseline_root / METHOD_DIR[method] / f"{task}.csv")
            df = read_csv(path)
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
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 20, "axes.labelsize": 16, "legend.fontsize": 16})
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=200)
    for ax, (domain, spec) in zip(axes, DOMAIN_SPECS.items()):
        for method in METHODS:
            x, m, lo, hi = domain_method_stats(method, domain, spec, args.baseline_root, args.ours_root)
            ax.plot(x, m, color=COLORS[method], lw=2.8, label=DEFAULT_LABELS[method])
            ax.fill_between(x, lo, hi, color=COLORS[method], alpha=0.12, linewidth=0)
        ax.set_title(f"{domain}\n{len(spec['tasks'])} tasks", fontweight="bold")
        ax.set_xlim(0, spec["x_max"])
        ax.set_ylim(*spec["y_lim"])
        xt = np.arange(0, spec["x_max"] + 1, 1_000_000)
        ax.set_xticks(xt)
        ax.set_xticklabels(["0"] + [f"{int(v/1_000_000)}M" for v in xt[1:]])
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
