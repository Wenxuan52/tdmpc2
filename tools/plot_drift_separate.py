#!/usr/bin/env python3
"""Plot per-method action-drift curves (policy vs planner variant) for 8 tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

DATA_ROOT = Path("/media/datasets/cheliu21/cxy_worldmodel/diff_metric")
SEED_CONFIG = Path("tools/diff/separate_seed.yaml")
OUT_PATH = Path("figures/drift_separate_mppi.pdf")

# METHODS_TO_PLOT = ["MPPI", "Beta0.0", "Beta0.1"]
METHODS_TO_PLOT = ["MPPI"]
METHOD_META = {
    "MPPI": {"planner_col": "action_drift/mppi", "label": "MPPI", "color": "#4C9A2A"},
    "Beta0.0": {"planner_col": "action_drift/diffusion", "label": "Beta 0.0", "color": "#2F78B7"},
    "Beta0.1": {"planner_col": "action_drift/diffusion", "label": "Beta 0.1", "color": "#1FAE9A"},
}

TASK_GRID = [
    ["acrobot-swingup", "mw-button-press-wall"],
    ["cheetah-run", "mw-handle-pull-side"],
    ["dog-trot", "mw-pick-place"],
    ["humanoid-walk", "mw-window-open"],
]
ALL_TASKS = [t for row in TASK_GRID for t in row]

# Typography one-stop interface
FONT = {
    "title": 20,
    "axis_label": 18,
    "ticks": 15,
    "legend": 16,
}

POLICY_COL = "action_drift/pi"
POLICY_COLOR = "#9e9e9e"
MEAN_ALPHA = 0.65
P_MEAN_ALPHA = 0.45
X_MAX = 1_000_000
X_TICKS = np.linspace(0, X_MAX, 6)
X_TICK_LABELS = [f"{v:.1f}" for v in np.linspace(0.0, 1.0, 6)]
EMA_ALPHA = 0.4

# Optional per-task y-axis bounds override: {"task-name": (ymin, ymax)}
Y_BOUNDS_OVERRIDE: Dict[str, tuple[float, float]] = {}


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
    sdf = sdf.dropna(subset=["step", col]).sort_values("step").drop_duplicates(subset=["step"], keep="last")
    if sdf.empty:
        return np.full_like(x_grid, np.nan, dtype=float)
    aligned = pd.Series(sdf[col].to_numpy(dtype=float), index=sdf["step"].to_numpy(dtype=float)).reindex(x_grid.astype(float))
    return aligned.interpolate(method="index", limit_area="inside").to_numpy(dtype=float)


def _ema(values: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    out = values.astype(float, copy=True)
    finite_idx = np.flatnonzero(np.isfinite(out))
    if finite_idx.size == 0:
        return out

    prev = out[finite_idx[0]]
    for idx in finite_idx[1:]:
        prev = alpha * out[idx] + (1.0 - alpha) * prev
        out[idx] = prev
    return out


def _mean_se(curves: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.vstack(curves)
    n = np.sum(np.isfinite(arr), axis=0)
    mean = np.nanmean(arr, axis=0)
    se = np.full_like(mean, np.nan)
    valid = n >= 2
    if np.any(valid):
        se[valid] = np.nanstd(arr[:, valid], axis=0, ddof=1) / np.sqrt(n[valid])
    return mean, se


def _prettify_task(task: str) -> str:
    return " ".join(tok.capitalize() for tok in task.split("-"))


def _collect_curves(method: str, task: str, x_grid: np.ndarray, seed_cfg: Dict[str, Dict[str, List[int]]]) -> tuple[List[np.ndarray], List[np.ndarray]]:
    policy_curves, planner_curves = [], []
    for seed in seed_cfg.get(method, {}).get(task, []):
        df = _load_task_seed(task, seed)
        if df.empty:
            continue
        policy_curves.append(_interp(df, POLICY_COL, x_grid))
        planner_curves.append(_interp(df, METHOD_META[method]["planner_col"], x_grid))
    return policy_curves, planner_curves


def _compute_unified_ymax(seed_cfg: Dict[str, Dict[str, List[int]]], x_grid: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for task in ALL_TASKS:
        method_ymax = []
        for method in METHODS_TO_PLOT:
            p_curves, m_curves = _collect_curves(method, task, x_grid, seed_cfg)
            if not p_curves or not m_curves:
                method_ymax.append(0.05)
                continue
            p_mean, _ = _mean_se(p_curves)
            m_mean, _ = _mean_se(m_curves)
            p_mean = _ema(p_mean)
            m_mean = _ema(m_mean)
            vals = np.concatenate([p_mean[np.isfinite(p_mean)], m_mean[np.isfinite(m_mean)]])
            method_ymax.append((float(np.max(vals)) + 0.01) if vals.size else 0.05)
        out[task] = max(min(method_ymax), 0.05)
    return out


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    seed_cfg = _parse_seed_config(SEED_CONFIG)
    x_grid = np.linspace(0, X_MAX, 1001, dtype=float)
    unified_ymax = _compute_unified_ymax(seed_cfg, x_grid)

    with PdfPages(OUT_PATH) as pdf:
        for method in METHODS_TO_PLOT:
            meta = METHOD_META[method]
            fig, axes = plt.subplots(4, 2, figsize=(18, 10), sharex=True)

            for r in range(4):
                for c in range(2):
                    task = TASK_GRID[r][c]
                    ax = axes[r, c]
                    p_curves, m_curves = _collect_curves(method, task, x_grid, seed_cfg)

                    if p_curves and m_curves:
                        p_mean, p_se = _mean_se(p_curves)
                        m_mean, m_se = _mean_se(m_curves)
                        p_mean = _ema(p_mean)
                        m_mean = _ema(m_mean)
                        ax.plot(x_grid, p_mean, ls="-", lw=2.0, alpha=P_MEAN_ALPHA, color=POLICY_COLOR, label="Policy network")
                        ax.fill_between(x_grid, p_mean - p_se, p_mean + p_se, color=POLICY_COLOR, alpha=0.14, linewidth=0)
                        ax.plot(x_grid, m_mean, lw=2.0, alpha=MEAN_ALPHA, color=meta["color"], label=meta["label"])
                        ax.fill_between(x_grid, m_mean - m_se, m_mean + m_se, color=meta["color"], alpha=0.14, linewidth=0)

                    ymin, ymax = (0.0, unified_ymax.get(task, 0.05))
                    if task in Y_BOUNDS_OVERRIDE:
                        ymin, ymax = Y_BOUNDS_OVERRIDE[task]
                    ax.set_ylim(ymin, ymax)
                    ax.set_yticks([ymin, ymax])
                    ax.set_yticklabels([f"{ymin:.1f}", f"{ymax:.1f}"], fontsize=FONT["ticks"])
                    ax.set_title(_prettify_task(task), fontsize=FONT["title"])
                    ax.set_facecolor("#f2f2f2")
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
                        ax.set_xlabel("Time steps (1M)", fontsize=FONT["axis_label"])
                        ax.set_xticks(X_TICKS)
                        ax.set_xticklabels(X_TICK_LABELS, fontsize=FONT["ticks"])
                    else:
                        ax.tick_params(axis="x", labelsize=FONT["ticks"])

            fig.supylabel("selected-action squared difference", fontsize=FONT["axis_label"])

            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.005), fontsize=FONT["legend"], frameon=False)
            fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.99])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved plot to {OUT_PATH}")


if __name__ == "__main__":
    main()
