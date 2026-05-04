#!/usr/bin/env python3
"""Aggregate drift/gap plots for DMControl and MetaWorld."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from matplotlib.patches import Patch
from matplotlib import colors as mcolors

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_ROOT = Path("/media/datasets/cheliu21/cxy_worldmodel/diff_metric")
SEED_CONFIG = Path("tools/diff/all_seed.yaml")
PLOT_MODE = "All"  # choose from: "Drift", "Gap", "All"
DRIFT_BOXPLOT_YLIM = (-0.13, 0.2)

EPS = 1e-8
POLICY_DENOM_FLOOR = 1e-3
X_MAX = 1_000_000
X_TICKS = np.linspace(0, X_MAX, 6)
X_TICK_LABELS = [f"{v:.1f}" for v in np.linspace(0.0, 1.0, 6)]

DM_TASKS = ["acrobot-swingup", "cheetah-run", "dog-trot", "humanoid-walk"]
MW_TASKS = ["mw-button-press-wall", "mw-handle-pull-side", "mw-pick-place", "mw-window-open"]

FONT = {"title": 24, "axis_label": 22, "ticks": 19, "legend": 20, "big_title": 30, "big_legend": 22}
MEAN_ALPHA = 0.75
EMA_WINDOW = 25
EMA_ALPHA = 0.2
METHOD_META = {
    "MPPI": {"drift_col": "action_drift/mppi", "gap_col": "planner_gap/mppi_to_policy", "label": "MPPI", "color": "#1f77b4"},
    "Beta0.0": {"drift_col": "action_drift/diffusion", "gap_col": "planner_gap/diffusion_to_policy", "label": "Diffusion (β=0.0)", "color": "#ff7f0e"},
    "Beta0.1": {"drift_col": "action_drift/diffusion", "gap_col": "planner_gap/diffusion_to_policy", "label": "Diffusion (β=0.1)", "color": "#2ca02c"},
}
METHODS_TO_PLOT = ["MPPI", "Beta0.0", "Beta0.1"]
STEP_STAGES = [(0, 250_000), (250_000, 500_000), (500_000, 750_000), (750_000, 1_000_000)]
STAGE_LABELS = ["0-250k", "250-500k", "500-750k", "750-1000k"]
CORRECTION_DELTA_STEPS = 5_000


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


def _shade_color(hex_color: str, level: int, total_levels: int = 4) -> str:
    base = np.array(mcolors.to_rgb(hex_color))
    mix = 0.88 - 0.58 * (level / max(total_levels - 1, 1))
    return mcolors.to_hex(base * (1.0 - mix) + np.ones(3) * mix)


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


def _ema_smooth(series: np.ndarray, window: int = EMA_WINDOW, alpha: float = EMA_ALPHA) -> np.ndarray:
    if series.size == 0:
        return series
    ser = pd.Series(series, dtype=float)
    interp = ser.interpolate(limit_direction="both")
    smoothed = interp.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    smoothed = smoothed.combine_first(interp.ewm(alpha=alpha, adjust=False).mean())
    return smoothed.to_numpy(dtype=float)


def _style_axes(ax: plt.Axes, title: str, y_label: str, y_lim: tuple[float, float], show_y_label: bool, use_time_axis: bool = True) -> None:
    ax.set_title(title, fontsize=FONT["title"])
    ax.set_xlabel("Time steps (1M)", fontsize=FONT["axis_label"])
    ax.set_ylabel(y_label if show_y_label else "", fontsize=FONT["axis_label"])
    if use_time_axis:
        ax.set_xticks(X_TICKS)
        ax.set_xticklabels(X_TICK_LABELS, fontsize=FONT["ticks"])
    else:
        ax.tick_params(axis="x", labelsize=FONT["ticks"])
    ax.set_ylim(*y_lim)
    ax.tick_params(axis="y", labelsize=FONT["ticks"])
    ax.set_facecolor("white")
    ax.grid(color="#d9d9d9", linewidth=1.8, axis="both")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.6)


def _collect_drift_stage_values(tasks: List[str], method: str, seed_cfg: Dict[str, Dict[str, List[int]]], domain: str) -> List[np.ndarray]:
    grouped: List[List[float]] = [[] for _ in STEP_STAGES]
    for task in tasks:
        for seed in seed_cfg.get(method, {}).get(task, []):
            df = _load_task_seed(task, seed)
            if df.empty or METHOD_META[method]["drift_col"] not in df.columns or "action_drift/pi" not in df.columns:
                continue
            step = pd.to_numeric(df["step"], errors="coerce").to_numpy(dtype=float)
            plan = pd.to_numeric(df[METHOD_META[method]["drift_col"]], errors="coerce").to_numpy(dtype=float)
            policy = pd.to_numeric(df["action_drift/pi"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(step) & np.isfinite(plan) & np.isfinite(policy)
            if not np.any(valid):
                continue
            step = step[valid]
            plan = plan[valid]
            policy = policy[valid]
            order = np.argsort(step)
            step = step[order]
            plan = plan[order]
            policy = policy[order]
            finite_policy = policy[np.isfinite(policy) & (policy > 0)]
            adaptive_floor = float(np.nanpercentile(finite_policy, 10)) if finite_policy.size else POLICY_DENOM_FLOOR
            denom_floor = max(POLICY_DENOM_FLOOR, adaptive_floor)
            drift = np.log10((plan + EPS) / (np.maximum(policy, denom_floor) + EPS))
            finite_drift = np.isfinite(drift)
            if not np.any(finite_drift):
                continue
            step = step[finite_drift]
            drift = drift[finite_drift]
            for i, (lo, hi) in enumerate(STEP_STAGES):
                in_stage = (step >= lo) & (step < hi)
                if np.any(in_stage):
                    grouped[i].extend(drift[in_stage].tolist())
    return [np.asarray(v, dtype=float) for v in grouped]


def _plot_drift_box(ax: plt.Axes, tasks: List[str], seed_cfg: Dict[str, Dict[str, List[int]]], domain_name: str, show_y_label: bool, title: str = "", show_xlabel: bool = True, y_lim: tuple[float, float] | None = None) -> None:
    centers = np.arange(len(METHODS_TO_PLOT), dtype=float)
    width = 0.16
    offsets = np.array([-0.24, -0.08, 0.08, 0.24])
    all_values = []
    for m_idx, method in enumerate(METHODS_TO_PLOT):
        stage_vals = _collect_drift_stage_values(tasks, method, seed_cfg, domain_name)
        for s_idx, values in enumerate(stage_vals):
            if values.size == 0:
                continue
            pos = centers[m_idx] + offsets[s_idx]
            color = _shade_color(METHOD_META[method]["color"], s_idx, len(STEP_STAGES))
            ax.boxplot(values, positions=[pos], widths=width, patch_artist=True,
                boxprops=dict(facecolor=color, edgecolor="black", linewidth=1.8),
                medianprops=dict(color="black", linewidth=1.8), whiskerprops=dict(color="black", linewidth=1.8),
                capprops=dict(color="black", linewidth=1.8),
                flierprops=dict(marker="D", markersize=4, markerfacecolor="#666", markeredgecolor="#666", alpha=0.7))
            all_values.append(values)
    if y_lim is not None:
        y_min, y_max = y_lim
    elif all_values:
        merged = np.concatenate(all_values)
        y_min, y_max = float(np.nanmin(merged)), float(np.nanmax(merged))
        pad = max(0.03 * (y_max - y_min), 1e-3)
        y_min, y_max = y_min - pad, y_max + pad
    else:
        y_min, y_max = (0.0, 0.05)
    ax.set_xticks(centers)
    ax.set_xticklabels(["MPPI", "Beta0.0", "Beta0.1"], fontsize=FONT["ticks"])
    _style_axes(ax, title, "Normalized Drift", (y_min, y_max), show_y_label=show_y_label, use_time_axis=False)
    if not show_xlabel:
        ax.set_xlabel("")
    ax.grid(axis="x", visible=False)
    gray_handles = [Patch(facecolor=_shade_color("#666666", i, len(STEP_STAGES)), edgecolor="black", label=lab) for i, lab in enumerate(STAGE_LABELS)]
    ax.legend(handles=gray_handles, fontsize=FONT["legend"], loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=True)


def _plot_drift_line(ax: plt.Axes, tasks: List[str], seed_cfg: Dict[str, Dict[str, List[int]]], x_grid: np.ndarray, domain_name: str, show_y_label: bool, title: str = "") -> None:
    all_min, all_max = [], []
    for method in METHODS_TO_PLOT:
        arr = _collect_samples(tasks, method, "Drift", seed_cfg, x_grid, domain_name)
        mean, se = _mean_se(arr)
        mean, se = _ema_smooth(mean), _ema_smooth(se)
        ax.plot(x_grid, mean, lw=2.0, color=METHOD_META[method]["color"], alpha=MEAN_ALPHA, label=METHOD_META[method]["label"])
        ax.fill_between(x_grid, mean - se, mean + se, color=METHOD_META[method]["color"], alpha=0.18)
        finite = mean[np.isfinite(mean)]
        if finite.size:
            all_min.append(float(np.min(finite))); all_max.append(float(np.max(finite)))
    if all_min and all_max:
        span = max(all_max) - min(all_min); pad = max(0.03 * span, 1e-3)
        y_min, y_max = min(all_min) - pad, max(all_max) + pad
    else:
        y_min, y_max = (-0.05, 0.05)
    _style_axes(ax, title, "Normalized Drift", (y_min, y_max), show_y_label=show_y_label, use_time_axis=True)

def main() -> None:
    if PLOT_MODE not in {"Drift", "Gap", "All"}:
        raise ValueError("PLOT_MODE must be 'Drift', 'Gap' or 'All'")

    out_path = Path(f"figures/drift_all_{PLOT_MODE}.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seed_cfg = _parse_seed_config(SEED_CONFIG)
    x_grid = np.linspace(0, X_MAX, 1001, dtype=float)
    domain_cfg = [("DMC", "DMControl", DM_TASKS), ("MetaWorld", "Meta World", MW_TASKS)]

    if PLOT_MODE == "All":
        fig, ax = plt.subplots(1, 1, figsize=(12.0, 7.5), sharex=False)
        domain_name, _, tasks = domain_cfg[1]  # Meta World only
        _plot_drift_box(
            ax, tasks, seed_cfg, domain_name, show_y_label=False, title="Normalized Drift",
            show_xlabel=False, y_lim=DRIFT_BOXPLOT_YLIM,
        )
        fig.tight_layout(rect=[0.02, 0.10, 0.98, 1.0])
    elif PLOT_MODE == "Gap":
        fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), sharex=False)
        for idx, (ax, (domain_name, title, tasks)) in enumerate(zip(axes, domain_cfg)):
            _plot_drift_box(ax, tasks, seed_cfg, domain_name, show_y_label=(idx == 0), title=title, show_xlabel=False, y_lim=DRIFT_BOXPLOT_YLIM)
        fig.tight_layout(rect=[0.02, 0.10, 0.98, 1.0]); fig.subplots_adjust(wspace=0.18)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), sharex=True)
        for idx, (ax, (domain_name, title, tasks)) in enumerate(zip(axes, domain_cfg)):
            _plot_drift_line(ax, tasks, seed_cfg, x_grid, domain_name, show_y_label=(idx == 0), title=title)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.01), fontsize=FONT["legend"], frameon=False)
        fig.tight_layout(rect=[0.02, 0.08, 0.98, 1.0]); fig.subplots_adjust(wspace=0.18)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
