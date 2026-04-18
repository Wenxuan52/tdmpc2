#!/usr/bin/env python3
"""Plot MetaWorld single-task curves with mean and 95% CI over 3 seeds."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from handle_metaworld_multi_log import load_task_seed_logs

METAWORLD_TASKS: List[str] = [
    "mw-assembly",
    "mw-basketball",
    "mw-bin-picking",
    "mw-box-close",
    "mw-button-press-topdown-wall",
    "mw-button-press-topdown",
    "mw-button-press-wall",
    "mw-button-press",
    "mw-coffee-button",
    "mw-coffee-pull",
    "mw-coffee-push",
    "mw-dial-turn",
    "mw-disassemble",
    "mw-door-close",
    "mw-door-lock",
    "mw-door-open",
    "mw-door-unlock",
    "mw-drawer-close",
    "mw-drawer-open",
    "mw-faucet-close",
    "mw-faucet-open",
    "mw-hammer",
    "mw-hand-insert",
    "mw-handle-press-side",
    "mw-handle-press",
    "mw-handle-pull-side",
    "mw-handle-pull",
    "mw-lever-pull",
    "mw-peg-insert-side",
    "mw-peg-unplug-side",
    "mw-pick-out-of-hole",
    "mw-pick-place-wall",
    "mw-pick-place",
    "mw-plate-slide-back-side",
    "mw-plate-slide-back",
    "mw-plate-slide-side",
    "mw-plate-slide",
    "mw-push-back",
    "mw-push-wall",
    "mw-push",
    "mw-reach-wall",
    "mw-reach",
    "mw-shelf-place",
    "mw-soccer",
    "mw-stick-pull",
    "mw-stick-push",
    "mw-sweep-into",
    "mw-sweep",
    "mw-window-close",
    "mw-window-open",
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
    "ours": "Ours",
    "tdmpc2": "TD-MPC2",
    "tdmpc": "TD-MPC",
    "dreamerv3": "DreamerV3",
    "sac": "SAC",
}

X_MAX = 2_000_000
Y_MIN, Y_MAX = -3.0, 103.0
GRID_STEP = 100_000
EXPECTED_SEEDS = [1, 2, 3]
DEFAULT_OURS_SOURCE = "log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("/root/workspace/tdmpc2/results"),
        help="Root containing baseline folders (sac/dreamerv3/tdmpc/tdmpc2).",
    )
    parser.add_argument(
        "--ours-log-root",
        type=Path,
        default=Path("/media/datasets/cheliu21/cxy_worldmodel/replay/_logs"),
        help="Root containing ours logs in {task}/seed_{seed}.log.",
    )
    parser.add_argument(
        "--ours-legend",
        type=str,
        default="Ours",
        help="Legend label for the improved TD-MPC2 baseline.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/MetaWorld.pdf"),
        help="Output pdf path.",
    )
    parser.add_argument(
        "--seed-config",
        type=Path,
        default=Path("tools/metaworld/seed.yaml"),
        help="Task-to-seeds config for Ours method.",
    )
    parser.add_argument(
        "--ours-csv-root",
        type=Path,
        default=Path("/media/datasets/cheliu21/cxy_worldmodel/online_csv"),
        help="Root containing ours CSV files in {task}_*.csv when source=csv.",
    )
    return parser.parse_args()


def prettify_task_name(task: str) -> str:
    if task.startswith("mw-"):
        task = task[3:]
    return " ".join(word.capitalize() for word in task.replace("-", " ").split())


def _read_baseline_task_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["step", "reward", "seed"])
    df = pd.read_csv(path)
    # MetaWorld baselines are expected to store success in `success` column.
    # Keep compatibility with any existing `reward`-column exports.
    if {"step", "seed", "reward"}.issubset(df.columns):
        out = df[["step", "reward", "seed"]].copy()
    elif {"step", "seed", "success"}.issubset(df.columns):
        out = df[["step", "success", "seed"]].rename(columns={"success": "reward"}).copy()
        # Convert success ratio [0, 1] to percentage [0, 100] for plotting.
        max_abs = np.nanmax(np.abs(pd.to_numeric(out["reward"], errors="coerce")))
        if np.isfinite(max_abs) and max_abs <= 1.5:
            out["reward"] = pd.to_numeric(out["reward"], errors="coerce") * 100.0
    else:
        raise ValueError(f"{path} missing columns: expected step/seed with reward or success")
    return out


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

    # Ensure every existing seed curve starts at (step=0, reward=0.0) when missing.
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


def load_ours_seed_config(path: Path, tasks: List[str]) -> Dict[str, Dict[str, object]]:
    task_to_cfg: Dict[str, Dict[str, object]] = {
        task: {"seeds": list(EXPECTED_SEEDS), "source": DEFAULT_OURS_SOURCE} for task in tasks
    }
    if not path.exists():
        return task_to_cfg

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
                task_to_cfg[current_task]["seeds"] = parsed
            continue
        if "source" in line:
            source_match = re.search(r"(csv|log)", line.lower())
            if source_match:
                task_to_cfg[current_task]["source"] = source_match.group(1)
            continue
        if line.startswith("-"):
            parsed = _extract_seed_list(line)
            if parsed:
                task_to_cfg[current_task]["seeds"] = parsed
    return task_to_cfg


def _sort_csv_seed_files(paths: List[Path]) -> List[Path]:
    def key(path: Path):
        match = re.search(r"_(\d+)\.csv$", path.name)
        return (int(match.group(1)) if match else 10**9, path.name)

    return sorted(paths, key=key)


def _read_ours_task_csv(task: str, csv_root: Path, seeds: List[int]) -> pd.DataFrame:
    paths = _sort_csv_seed_files(list(csv_root.glob(f"{task}_*.csv")))
    if not paths:
        return pd.DataFrame(columns=["step", "reward", "seed"])

    # We intentionally map files to configured seeds by sorted file order so the
    # plotting seed selection remains stable even when filename suffixes are not
    # exactly equal to semantic seed id.
    selected_paths = paths[: len(seeds)]
    records: List[pd.DataFrame] = []
    for seed, path in zip(seeds, selected_paths):
        raw = pd.read_csv(path)
        if {"step", "episode_success"}.issubset(raw.columns):
            sdf = raw[["step", "episode_success"]].rename(columns={"episode_success": "reward"}).copy()
        elif {"step", "success"}.issubset(raw.columns):
            sdf = raw[["step", "success"]].rename(columns={"success": "reward"}).copy()
        elif {"step", "reward"}.issubset(raw.columns):
            sdf = raw[["step", "reward"]].copy()
        else:
            raise ValueError(f"{path} missing columns: expected step with episode_success/success/reward")

        sdf["seed"] = seed
        max_abs = np.nanmax(np.abs(pd.to_numeric(sdf["reward"], errors="coerce")))
        if np.isfinite(max_abs) and max_abs <= 1.5:
            sdf["reward"] = pd.to_numeric(sdf["reward"], errors="coerce") * 100.0
        records.append(sdf)

    if not records:
        return pd.DataFrame(columns=["step", "reward", "seed"])
    return pd.concat(records, ignore_index=True)


def load_method_task_data(
    method: str,
    task: str,
    baseline_root: Path,
    ours_log_root: Path,
    ours_csv_root: Path,
    ours_seeds: List[int],
    ours_source: str,
) -> pd.DataFrame:
    if method == "ours":
        if ours_source == "csv":
            df = _read_ours_task_csv(task=task, csv_root=ours_csv_root, seeds=ours_seeds)
        else:
            df = load_task_seed_logs(
                root=ours_log_root,
                task=task,
                seeds=ours_seeds,
                step_bucket=GRID_STEP,
                window_size=10,
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


def plot_all(args: argparse.Namespace) -> None:
    labels = dict(DEFAULT_LABELS)
    labels["ours"] = args.ours_legend
    ours_seed_config = load_ours_seed_config(args.seed_config, METAWORLD_TASKS)

    step_grid = np.arange(0, X_MAX + GRID_STEP, GRID_STEP, dtype=int)

    fig, axes = plt.subplots(10, 5, figsize=(20, 32), sharex=True, sharey=True)
    axes = axes.flatten()

    legend_handles = []
    for idx, task in enumerate(METAWORLD_TASKS):
        ax = axes[idx]
        for method in METHODS:
            task_cfg = ours_seed_config[task]
            seed_list = task_cfg["seeds"] if method == "ours" else EXPECTED_SEEDS
            source = task_cfg["source"] if method == "ours" else "log"
            df = load_method_task_data(
                method,
                task,
                args.baseline_root,
                args.ours_log_root,
                args.ours_csv_root,
                seed_list,
                source,
            )
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
            line, = ax.plot(step_grid, mean, color=color, linewidth=line_width, alpha=mean_alpha)
            ax.plot(step_grid, upper, color=color, linewidth=1.0, alpha=ci_alpha)
            ax.plot(step_grid, lower, color=color, linewidth=1.0, alpha=ci_alpha)
            ax.fill_between(step_grid, lower, upper, color=color, alpha=fill_alpha, linewidth=0)

            if idx == 0:
                legend_handles.append(line)

        ax.set_title(prettify_task_name(task), fontsize=26)
        ax.set_xlim(-1000, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.grid(True, linestyle="-", linewidth=0.8, alpha=0.25)

        row, col = divmod(idx, 5)
        ax.set_xticks([0, 1_000_000, 2_000_000])
        ax.set_yticks([0, 50, 100])

        if row == 9:
            ax.set_xticklabels(["0", "1M", "2M"])
            ax.tick_params(axis="x", labelsize=20, labelbottom=True)
        else:
            ax.tick_params(axis="x", labelsize=20, labelbottom=False)

        # Always render y-axis tick labels so each subplot keeps visible y ticks.
        # This avoids missing y-axis ticks when shared axes are used.
        ax.tick_params(axis="y", labelsize=20, labelleft=True)

    fig.legend(
        legend_handles,
        [labels[m] for m in METHODS],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=5,
        frameon=False,
        fontsize=20,
    )

    fig.subplots_adjust(left=0.06, right=0.995, top=0.98, bottom=0.07, wspace=0.15, hspace=0.4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {args.out}")


def main() -> None:
    args = parse_args()
    plot_all(args)


if __name__ == "__main__":
    main()
