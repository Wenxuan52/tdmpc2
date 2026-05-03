#!/usr/bin/env python3
"""Aggregate drift/gap plots for DMControl and MetaWorld."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_ROOT = Path("/media/datasets/cheliu21/cxy_worldmodel/diff_metric")
SEED_CONFIG = Path("tools/diff/all_seed.yaml")
PLOT_MODE = "Gap"  # choose from: "Drift", "Gap"
EXCLUDE_BETA01_HUMANOID_WALK = True  # can be overridden by `exclude_beta01_humanoid_walk` in seed config

EPS = 1e-8
POLICY_DENOM_FLOOR = 1e-3
X_MAX = 1_000_000
X_TICKS = np.linspace(0, X_MAX, 6)
X_TICK_LABELS = [f"{v:.1f}" for v in np.linspace(0.0, 1.0, 6)]

DM_TASKS = ["acrobot-swingup", "cheetah-run", "dog-trot", "humanoid-walk"]
MW_TASKS = ["mw-button-press-wall", "mw-handle-pull-side", "mw-pick-place", "mw-window-open"]

FONT = {"title": 20, "axis_label": 18, "ticks": 15, "legend": 16}
MEAN_ALPHA = 0.75
POLICY_REF_COLOR = "#666666"
METHOD_META = {
    "MPPI": {"drift_col": "action_drift/mppi", "gap_col": "planner_gap/mppi_to_policy", "label": "MPPI", "color": "#1f77b4"},
    "Beta0.0": {"drift_col": "action_drift/diffusion", "gap_col": "planner_gap/diffusion_to_policy", "label": "Diffusion (β=0.0)", "color": "#ff7f0e"},
    "Beta0.1": {"drift_col": "action_drift/diffusion", "gap_col": "planner_gap/diffusion_to_policy", "label": "Diffusion (β=0.1)", "color": "#2ca02c"},
}
METHODS_TO_PLOT = ["MPPI", "Beta0.0", "Beta0.1"]


def _extract_ints(line: str) -> List[int]:
    if "[" not in line or "]" not in line:
        return []
    inside = line[line.find("[") + 1 : line.find("]")]
    return [int(x.strip()) for x in inside.split(",") if x.strip()]


def _parse_seed_config(path: Path) -> Dict[str, Dict[str, List[int]]]:
    defaults = {
        "MPPI": {task: [7, 8, 9] for task in DM_TASKS + MW_TASKS},
        "Beta0.0": {task: [1, 2, 3] for task in DM_TASKS + MW_TASKS},
        "Beta0.1": {task: [14, 15, 16] for task in DM_TASKS + MW_TASKS},
    }
    if not path.exists():
        return defaults
    out = {k: dict(v) for k, v in defaults.items()}
    current = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if indent == 0 and line.endswith(":"):
            current = line[:-1].strip()
            if current not in out:
                out[current] = {}
            continue
        if current is None or ":" not in line:
            continue
        task, rhs = line.split(":", 1)
        seeds = _extract_ints(rhs)
        if seeds:
            out[current][task.strip()] = seeds
    return out


def _parse_bool_flag(path: Path, key: str, default: bool) -> bool:
    if not path.exists():
        return default
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        left, right = line.split(":", 1)
        if left.strip() != key:
            continue
        val = right.strip().lower()
        if val in {"true", "1", "yes", "on"}:
            return True
        if val in {"false", "0", "no", "off"}:
            return False
    return default


def _load_task_seed(task: str, seed: int) -> pd.DataFrame:
    path = DATA_ROOT / f"DIFF_metric_{task}_seed{seed}.csv"
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
    ser = pd.Series(sdf[col].to_numpy(dtype=float), index=sdf["step"].to_numpy(dtype=float))
    return ser.reindex(x_grid.astype(float)).interpolate(method="index", limit_area="inside").to_numpy(dtype=float)


def _mean_se(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.sum(np.isfinite(arr), axis=0)
    mean = np.nanmean(arr, axis=0)
    se = np.full_like(mean, np.nan)
    valid = n >= 2
    if np.any(valid):
        se[valid] = np.nanstd(arr[:, valid], axis=0, ddof=1) / np.sqrt(n[valid])
    return mean, se


def _collect_samples(tasks: List[str], method: str, mode: str, seed_cfg: Dict[str, Dict[str, List[int]]], x_grid: np.ndarray, domain: str) -> np.ndarray:
    samples: List[np.ndarray] = []
    for task in tasks:
        if domain == "DMC" and method == "Beta0.1" and EXCLUDE_BETA01_HUMANOID_WALK and task == "humanoid-walk":
            continue
        for seed in seed_cfg.get(method, {}).get(task, []):
            df = _load_task_seed(task, seed)
            if df.empty:
                continue
            if mode == "Drift":
                plan = _interp(df, METHOD_META[method]["drift_col"], x_grid)
                policy = _interp(df, "action_drift/pi", x_grid)
                finite_policy = policy[np.isfinite(policy) & (policy > 0)]
                adaptive_floor = float(np.nanpercentile(finite_policy, 10)) if finite_policy.size else POLICY_DENOM_FLOOR
                denom_floor = max(POLICY_DENOM_FLOOR, adaptive_floor)
                series = np.log10((plan + EPS) / (np.maximum(policy, denom_floor) + EPS))
            else:
                series = _interp(df, METHOD_META[method]["gap_col"], x_grid)
            if np.isfinite(series).any():
                samples.append(series)
    if not samples:
        return np.full((1, x_grid.size), np.nan)
    return np.vstack(samples)


def _style_axes(ax: plt.Axes, title: str, y_label: str, y_lim: tuple[float, float]) -> None:
    ax.set_title(title, fontsize=FONT["title"])
    ax.set_xlabel("Time steps (1M)", fontsize=FONT["axis_label"])
    ax.set_ylabel(y_label, fontsize=FONT["axis_label"])
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels(X_TICK_LABELS, fontsize=FONT["ticks"])
    ax.set_ylim(*y_lim)
    ax.tick_params(axis="y", labelsize=FONT["ticks"])
    ax.set_facecolor("#f2f2f2")
    ax.grid(color="#d9d9d9", linewidth=3.0)
    for s in ax.spines.values():
        s.set_visible(False)


def main() -> None:
    global EXCLUDE_BETA01_HUMANOID_WALK
    if PLOT_MODE not in {"Drift", "Gap"}:
        raise ValueError("PLOT_MODE must be 'Drift' or 'Gap'")

    out_path = Path(f"figures/drift_all_{PLOT_MODE}.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seed_cfg = _parse_seed_config(SEED_CONFIG)
    EXCLUDE_BETA01_HUMANOID_WALK = _parse_bool_flag(
        SEED_CONFIG, key="exclude_beta01_humanoid_walk", default=EXCLUDE_BETA01_HUMANOID_WALK
    )
    x_grid = np.linspace(0, X_MAX, 1001, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), sharex=True)
    domain_cfg = [("DMC", "DMControl", DM_TASKS), ("MetaWorld", "MetaWorld", MW_TASKS)]

    for ax, (domain_name, title, tasks) in zip(axes, domain_cfg):
        all_min = []
        all_max = []
        for method in METHODS_TO_PLOT:
            arr = _collect_samples(tasks, method, PLOT_MODE, seed_cfg, x_grid, domain_name)
            mean, se = _mean_se(arr)
            ax.plot(x_grid, mean, lw=2.0, color=METHOD_META[method]["color"], alpha=MEAN_ALPHA, label=METHOD_META[method]["label"])
            ax.fill_between(x_grid, mean - se, mean + se, color=METHOD_META[method]["color"], alpha=0.18)
            finite = mean[np.isfinite(mean)]
            if finite.size:
                all_min.append(float(np.min(finite)))
                all_max.append(float(np.max(finite)))

        if all_min and all_max:
            span = max(all_max) - min(all_min)
            pad = max(0.03 * span, 1e-3)
            y_min, y_max = min(all_min) - pad, max(all_max) + pad
        else:
            y_min, y_max = (-0.05, 0.05) if PLOT_MODE == "Drift" else (0.0, 0.05)

        if PLOT_MODE == "Drift":
            y_label = r"$\log_{10}\left(\frac{d_t^m+\epsilon}{d_t^\pi+\epsilon}\right)$"
        else:
            y_label = "Planner-policy gap"

        _style_axes(ax, title, y_label, (y_min, y_max))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.01), fontsize=FONT["legend"], frameon=False)
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 1.0])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
