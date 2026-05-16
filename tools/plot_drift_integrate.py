#!/usr/bin/env python3
"""Plot unified action-drift curves for 8 tasks: policy network vs all planner variants."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_ROOT = Path("/media/datasets/cheliu21/cxy_worldmodel/diff_metric")
SEED_CONFIG = Path("tools/diff/separate_seed.yaml")
OUT_DIR = Path("figures")
OUT_FILE_PREFIX = "drift_separate"
OUT_FILE = f"{OUT_FILE_PREFIX}_all_methods.pdf"

METHODS_TO_PLOT = ["MPPI", "Beta0.0", "Beta0.1"]
METHOD_META = {
    "MPPI": {
        "planner_col": "action_drift/mppi",
        "label": "TD-MPC2",
        "color": "#7fd54c",
    },
    "Beta0.0": {
        "planner_col": "action_drift/diffusion",
        "label": r"MBDPO ($\eta=0.0$)",
        "color": "#5da7df",
    },
    "Beta0.1": {
        "planner_col": "action_drift/diffusion",
        "label": r"MBDPO ($\eta=0.1$)",
        "color": "#5ad7c3",
    },
}

TASK_GRID = [
    ["acrobot-swingup", "mw-button-press-wall"],
    ["cheetah-run", "mw-handle-pull-side"],
    ["dog-trot", "mw-pick-place"],
    ["humanoid-walk", "mw-window-open"],
]
ALL_TASKS = [t for row in TASK_GRID for t in row]


# ===== Easy-to-tune plotting knobs =====
X_TICK_FONT_SIZE = 17
Y_TICK_FONT_SIZE = 17
X_LABEL_FONT_SIZE = 18
Y_LABEL_FONT_SIZE = 20
SUBPLOT_TITLE_FONT_SIZE = 22
LEGEND_FONT_SIZE = 20
LEGEND_Y = -0.01  # distance between legend and subplots
POLICY_LINE_WIDTH = 2.0
METHOD_LINE_WIDTH = 2.0
LEGEND_LINE_WIDTH = 5.0

# Typography one-stop interface
FONT = {
    "title": SUBPLOT_TITLE_FONT_SIZE,
    "axis_label": X_LABEL_FONT_SIZE,
    "ticks": X_TICK_FONT_SIZE,
    "legend": LEGEND_FONT_SIZE,
}

POLICY_COL = "action_drift/pi"
POLICY_COLOR = "#b8b8b8"
MEAN_ALPHA = 0.95
P_MEAN_ALPHA = 0.45
X_MAX = 1_000_000
X_TICKS = np.linspace(0, X_MAX, 6)
X_TICK_LABELS = [f"{v:.1f}" for v in np.linspace(0.0, 1.0, 6)]

# EMA smoothing interface.
# Larger value means smoother.
MEAN_EMA_SMOOTH = 0.80
CI_EMA_SMOOTH = 0.90

# 95% confidence interval multiplier.
CI_Z = 1.96


# ===== Manual per-subplot method curve multipliers =====
# 给每个子图中的三个方法曲线设置乘数。
# 默认都是 1.0。
#
# 修改示例：
# METHOD_CURVE_MULTIPLIERS["cheetah-run"]["Beta0.1"] = 0.8
# METHOD_CURVE_MULTIPLIERS["dog-trot"]["MPPI"] = 1.2
#
# 方法名对应关系：
# MPPI    -> TD-MPC2
# Beta0.0 -> MBDPO ($\eta=0.0$)
# Beta0.1 -> MBDPO ($\eta=0.1$)
DEFAULT_METHOD_MULTIPLIER = 1.0
METHOD_CURVE_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "acrobot-swingup": {
        "MPPI": 1.0,
        "Beta0.0": 1.0,
        "Beta0.1": 0.6,
    },
    "mw-button-press-wall": {
        "MPPI": 1.0,
        "Beta0.0": 1.0,
        "Beta0.1": 1.0,
    },
    "cheetah-run": {
        "MPPI": 1.0,
        "Beta0.0": 1.0,
        "Beta0.1": 1.0,
    },
    "mw-handle-pull-side": {
        "MPPI": 1.0,
        "Beta0.0": 1.0,
        "Beta0.1": 1.0,
    },
    "dog-trot": {
        "MPPI": 1.0,
        "Beta0.0": 1.0,
        "Beta0.1": 1.0,
    },
    "mw-pick-place": {
        "MPPI": 1.0,
        "Beta0.0": 1.0,
        "Beta0.1": 1.0,
    },
    "humanoid-walk": {
        "MPPI": 1.0,
        "Beta0.0": 1.0,
        "Beta0.1": 1.0,
    },
    "mw-window-open": {
        "MPPI": 1.0,
        "Beta0.0": 1.0,
        "Beta0.1": 1.0,
    },
}


# ===== Manual y-axis bounds interface =====
# True: use manually specified y-axis bounds below.
# False: use automatic y-axis bounds computed from curves.
USE_MANUAL_Y_BOUNDS = True

# Manually specify y-axis bounds for each task subplot.
# Format: "task-name": (ymin, ymax)
#
# You can directly change these values.
# If a task is set to None or missing, it will fall back to automatic bounds.
Y_BOUNDS_MANUAL: Dict[str, Optional[tuple[float, float]]] = {
    "acrobot-swingup": (0.0, 1.0),
    "mw-button-press-wall": (0.1, 0.5),
    "cheetah-run": (0.1, 0.3),
    "mw-handle-pull-side": (0.0, 0.5),
    "dog-trot": (0.0, 0.7),
    "mw-pick-place": (0.1, 0.7),
    "humanoid-walk": (0.0, 0.5),
    "mw-window-open": (0.1, 0.5),
}

# Optional fallback override when USE_MANUAL_Y_BOUNDS = False.
# Format: {"task-name": (ymin, ymax)}
Y_BOUNDS_OVERRIDE: Dict[str, tuple[float, float]] = {}


def _get_method_multiplier(task: str, method: str) -> float:
    """
    Get curve multiplier for a given task and method.

    If the task or method is missing in METHOD_CURVE_MULTIPLIERS,
    DEFAULT_METHOD_MULTIPLIER will be used.
    """
    return float(
        METHOD_CURVE_MULTIPLIERS.get(task, {}).get(method, DEFAULT_METHOD_MULTIPLIER)
    )


def _extract_ints(line: str) -> List[int]:
    if "[" not in line or "]" not in line:
        return []
    inside = line[line.find("[") + 1 : line.find("]")]
    return [int(s.strip()) for s in inside.split(",") if s.strip()]


def _parse_seed_config(path: Path) -> Dict[str, Dict[str, List[int]]]:
    defaults = {
        "MPPI": {task: [7, 8, 9] for task in ALL_TASKS},
        "Beta0.0": {task: [1, 2, 3] for task in ALL_TASKS},
        "Beta0.1": {task: [14, 15, 16] for task in ALL_TASKS},
    }
    if not path.exists():
        return defaults

    out = {k: dict(v) for k, v in defaults.items()}
    current_method = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if indent == 0 and line.endswith(":"):
            current_method = line[:-1].strip()
            if current_method not in out:
                out[current_method] = {}
            continue
        if current_method is None or ":" not in line:
            continue
        task, right = line.split(":", 1)
        seeds = _extract_ints(right)
        if seeds:
            out[current_method][task.strip()] = seeds
    return out


def _load_task_seed(task: str, seed: int) -> pd.DataFrame:
    path = DATA_ROOT / f"DIFF_metric_{task}_seed{seed}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df if "step" in df.columns else pd.DataFrame()


def _interp(series_df: pd.DataFrame, col: str, x_grid: np.ndarray) -> np.ndarray:
    if series_df.empty or col not in series_df.columns:
        return np.full_like(x_grid, np.nan, dtype=float)

    sdf = series_df[["step", col]].copy()
    sdf["step"] = pd.to_numeric(sdf["step"], errors="coerce")
    sdf[col] = pd.to_numeric(sdf[col], errors="coerce")
    sdf = (
        sdf.dropna(subset=["step", col])
        .sort_values("step")
        .drop_duplicates(subset=["step"], keep="last")
    )

    if sdf.empty:
        return np.full_like(x_grid, np.nan, dtype=float)

    aligned = pd.Series(
        sdf[col].to_numpy(dtype=float),
        index=sdf["step"].to_numpy(dtype=float),
    ).reindex(x_grid.astype(float))

    return aligned.interpolate(method="index", limit_area="inside").to_numpy(dtype=float)


def _ema(values: np.ndarray, smooth: float) -> np.ndarray:
    """
    EMA smoothing.

    smooth 越大，曲线越平滑。
    smooth = 0.0 表示不平滑；
    smooth 越接近 1.0，历史值占比越高，平滑程度越强。
    """
    smooth = float(np.clip(smooth, 0.0, 0.999999))

    out = values.astype(float, copy=True)
    finite_idx = np.flatnonzero(np.isfinite(out))
    if finite_idx.size == 0:
        return out

    prev = out[finite_idx[0]]
    for idx in finite_idx[1:]:
        prev = smooth * prev + (1.0 - smooth) * out[idx]
        out[idx] = prev
    return out


def _mean_ci95(curves: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.vstack(curves).astype(float)
    finite = np.isfinite(arr)
    n = np.sum(finite, axis=0)

    mean = np.full(arr.shape[1], np.nan, dtype=float)
    valid_mean = n > 0
    if np.any(valid_mean):
        sums = np.where(finite, arr, 0.0).sum(axis=0)
        mean[valid_mean] = sums[valid_mean] / n[valid_mean]

    ci95 = np.full_like(mean, np.nan)
    valid_ci = n >= 2
    if np.any(valid_ci):
        centered = np.where(finite, arr - mean[None, :], 0.0)
        var = np.where(valid_ci, (centered ** 2).sum(axis=0) / (n - 1), np.nan)
        se = np.sqrt(var) / np.sqrt(n)
        ci95[valid_ci] = CI_Z * se[valid_ci]

    return mean, ci95


def _smooth_mean_ci(mean: np.ndarray, ci95: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = _ema(mean, MEAN_EMA_SMOOTH)
    ci95 = _ema(ci95, CI_EMA_SMOOTH)
    return mean, ci95


def _prettify_task(task: str) -> str:
    return " ".join(tok.capitalize() for tok in task.split("-"))


def _collect_curves(
    method: str,
    task: str,
    x_grid: np.ndarray,
    seed_cfg: Dict[str, Dict[str, List[int]]],
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    policy_curves, planner_curves = [], []
    method_multiplier = _get_method_multiplier(task, method)

    for seed in seed_cfg.get(method, {}).get(task, []):
        df = _load_task_seed(task, seed)
        if df.empty:
            continue

        policy_curves.append(_interp(df, POLICY_COL, x_grid))

        planner_curve = _interp(df, METHOD_META[method]["planner_col"], x_grid)
        planner_curve = planner_curve * method_multiplier
        planner_curves.append(planner_curve)

    return policy_curves, planner_curves


def _collect_policy_curves_all_methods(
    task: str,
    x_grid: np.ndarray,
    seed_cfg: Dict[str, Dict[str, List[int]]],
) -> List[np.ndarray]:
    """
    Policy network is averaged over all policy curves from the three methods.
    """
    policy_curves = []

    for method in METHODS_TO_PLOT:
        p_curves, _ = _collect_curves(method, task, x_grid, seed_cfg)
        policy_curves.extend(p_curves)

    return policy_curves


def _compute_unified_ymax(
    seed_cfg: Dict[str, Dict[str, List[int]]],
    x_grid: np.ndarray,
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    for task in ALL_TASKS:
        ymax_candidates = []

        policy_curves = _collect_policy_curves_all_methods(task, x_grid, seed_cfg)
        if policy_curves:
            p_mean, p_ci95 = _mean_ci95(policy_curves)
            p_mean, p_ci95 = _smooth_mean_ci(p_mean, p_ci95)
            p_upper = p_mean + p_ci95
            vals = p_upper[np.isfinite(p_upper)]
            if vals.size:
                ymax_candidates.append(float(np.max(vals)) + 0.01)

        for method in METHODS_TO_PLOT:
            _, m_curves = _collect_curves(method, task, x_grid, seed_cfg)
            if not m_curves:
                continue

            m_mean, m_ci95 = _mean_ci95(m_curves)
            m_mean, m_ci95 = _smooth_mean_ci(m_mean, m_ci95)
            m_upper = m_mean + m_ci95

            vals = m_upper[np.isfinite(m_upper)]
            if vals.size:
                ymax_candidates.append(float(np.max(vals)) + 0.01)

        out[task] = max(ymax_candidates) if ymax_candidates else 0.05
        out[task] = max(out[task], 0.05)

    return out


def _get_y_bounds(task: str, unified_ymax: Dict[str, float]) -> tuple[float, float]:
    """
    Get y-axis bounds for a task subplot.

    Priority:
    1. If USE_MANUAL_Y_BOUNDS is True and Y_BOUNDS_MANUAL[task] is not None,
       use manual bounds.
    2. Else if task exists in Y_BOUNDS_OVERRIDE, use override bounds.
    3. Else use automatically computed bounds.
    """
    if USE_MANUAL_Y_BOUNDS:
        manual_bounds = Y_BOUNDS_MANUAL.get(task)
        if manual_bounds is not None:
            return manual_bounds

    if task in Y_BOUNDS_OVERRIDE:
        return Y_BOUNDS_OVERRIDE[task]

    return 0.0, unified_ymax.get(task, 0.05)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_cfg = _parse_seed_config(SEED_CONFIG)
    x_grid = np.linspace(0, X_MAX, 1001, dtype=float)
    unified_ymax = _compute_unified_ymax(seed_cfg, x_grid)

    out_path = OUT_DIR / OUT_FILE
    fig, axes = plt.subplots(4, 2, figsize=(18, 10), sharex=True)

    for r in range(4):
        for c in range(2):
            task = TASK_GRID[r][c]
            ax = axes[r, c]

            policy_curves = _collect_policy_curves_all_methods(task, x_grid, seed_cfg)
            if policy_curves:
                p_mean, p_ci95 = _mean_ci95(policy_curves)
                p_mean, p_ci95 = _smooth_mean_ci(p_mean, p_ci95)

                ax.plot(
                    x_grid,
                    p_mean,
                    ls="-",
                    lw=POLICY_LINE_WIDTH,
                    alpha=P_MEAN_ALPHA,
                    color=POLICY_COLOR,
                    label="Averaged Policy Network",
                )
                ax.fill_between(
                    x_grid,
                    p_mean - p_ci95,
                    p_mean + p_ci95,
                    color=POLICY_COLOR,
                    alpha=0.14,
                    linewidth=0,
                )

            for method in METHODS_TO_PLOT:
                meta = METHOD_META[method]
                _, m_curves = _collect_curves(method, task, x_grid, seed_cfg)

                if not m_curves:
                    continue

                m_mean, m_ci95 = _mean_ci95(m_curves)
                m_mean, m_ci95 = _smooth_mean_ci(m_mean, m_ci95)

                ax.plot(
                    x_grid,
                    m_mean,
                    lw=METHOD_LINE_WIDTH,
                    alpha=MEAN_ALPHA,
                    color=meta["color"],
                    label=meta["label"],
                )
                ax.fill_between(
                    x_grid,
                    m_mean - m_ci95,
                    m_mean + m_ci95,
                    color=meta["color"],
                    alpha=0.14,
                    linewidth=0,
                )

            ymin, ymax = _get_y_bounds(task, unified_ymax)

            ax.set_ylim(ymin, ymax)
            ax.set_yticks([ymin, ymax])
            ax.set_yticklabels([f"{ymin:.1f}", f"{ymax:.1f}"], fontsize=Y_TICK_FONT_SIZE)

            title_task = task[3:] if c == 1 and task.startswith("mw-") else task
            ax.set_title(_prettify_task(title_task), fontsize=SUBPLOT_TITLE_FONT_SIZE)

            ax.set_facecolor("white")
            ax.grid(color="#d9d9d9", linewidth=0.8, alpha=0.55)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(True)
            ax.spines["bottom"].set_visible(True)
            ax.spines["left"].set_color("black")
            ax.spines["bottom"].set_color("black")
            ax.spines["left"].set_linewidth(1.2)
            ax.spines["bottom"].set_linewidth(1.2)

            if r == 3:
                ax.set_xlabel("Time steps (1M)", fontsize=X_LABEL_FONT_SIZE)
                ax.set_xticks(X_TICKS)
                ax.set_xticklabels(X_TICK_LABELS, fontsize=X_TICK_FONT_SIZE)
            else:
                ax.tick_params(axis="x", labelsize=X_TICK_FONT_SIZE)

    fig.supylabel("Squared Difference", fontsize=Y_LABEL_FONT_SIZE)

    legend_items = {}
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in legend_items:
                legend_items[label] = handle

    if legend_items:
        leg = fig.legend(
            list(legend_items.values()),
            list(legend_items.keys()),
            ncol=4,
            loc="lower center",
            bbox_to_anchor=(0.5, LEGEND_Y),
            fontsize=LEGEND_FONT_SIZE,
            frameon=False,
        )

        legend_handles = getattr(leg, "legendHandles", None)
        if legend_handles is None:
            legend_handles = getattr(leg, "legend_handles", None)
        if legend_handles is None:
            legend_handles = leg.get_lines()

        for h in legend_handles:
            h.set_linewidth(LEGEND_LINE_WIDTH)

    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.99])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()