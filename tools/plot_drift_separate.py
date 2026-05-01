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
OUT_PATH = Path("figures/drift_separate.pdf")

# Methods to render (edit this list to control what gets plotted).
METHODS_TO_PLOT = ["MPPI", "Beta0.0", "Beta0.1"]

METHOD_META = {
    "MPPI": {
        "planner_col": "action_drift/mppi",
        "label": "MPPI",
        "color": "#1f77b4",
    },
    "Beta0.0": {
        "planner_col": "action_drift/diffusion",
        "label": "Beta 0.0",
        "color": "#ff7f0e",
    },
    "Beta0.1": {
        "planner_col": "action_drift/diffusion",
        "label": "Beta 0.1",
        "color": "#2ca02c",
    },
}

TASK_GRID = [
    ["acrobot-swingup", "mw-button-press-wall"],
    ["cheetah-run", "mw-handle-pull-side"],
    ["dog-trot", "mw-pick-place"],
    ["humanoid-walk", "mw-window-open"],
]

POLICY_COL = "action_drift/pi"
X_MAX = 1_000_000
X_TICKS = np.linspace(0, X_MAX, 6)
X_TICK_LABELS = [f"{v:.1f}" for v in np.linspace(0.0, 1.0, 6)]

# Optional per-task y-axis bounds override: {"task-name": (ymin, ymax)}
Y_BOUNDS_OVERRIDE: Dict[str, tuple[float, float]] = {}


def _parse_seed_config(path: Path) -> Dict[str, List[int]]:
    defaults = {"MPPI": [7, 8, 9], "Beta0.0": [1, 2, 3], "Beta0.1": [14, 15, 16]}
    if not path.exists():
        return defaults

    out = dict(defaults)
    current = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not raw.startswith((" ", "\t")) and line.endswith(":"):
            current = line[:-1].strip()
            continue
        if current is None:
            continue
        if "seed" in line and "[" in line and "]" in line:
            inside = line[line.find("[") + 1 : line.find("]")]
            seeds = [int(s.strip()) for s in inside.split(",") if s.strip()]
            if seeds:
                out[current] = seeds
    return out


def _load_task_seed(task: str, seed: int) -> pd.DataFrame:
    path = DATA_ROOT / f"DIFF_metric_{task}_seed{seed}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "step" not in df.columns:
        return pd.DataFrame()
    return df


def _interp(series_df: pd.DataFrame, col: str, x_grid: np.ndarray) -> np.ndarray:
    if series_df.empty or col not in series_df.columns:
        return np.full_like(x_grid, np.nan, dtype=float)
    sdf = series_df[["step", col]].copy()
    sdf["step"] = pd.to_numeric(sdf["step"], errors="coerce")
    sdf[col] = pd.to_numeric(sdf[col], errors="coerce")
    sdf = sdf.dropna(subset=["step", col]).sort_values("step")
    if sdf.empty:
        return np.full_like(x_grid, np.nan, dtype=float)
    sdf = sdf.drop_duplicates(subset=["step"], keep="last")
    idx = sdf["step"].to_numpy(dtype=float)
    vals = sdf[col].to_numpy(dtype=float)
    aligned = pd.Series(vals, index=idx).reindex(x_grid.astype(float)).interpolate(method="index", limit_area="inside")
    return aligned.to_numpy(dtype=float)


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


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    seed_cfg = _parse_seed_config(SEED_CONFIG)
    x_grid = np.linspace(0, X_MAX, 1001, dtype=float)

    with PdfPages(OUT_PATH) as pdf:
        for method in METHODS_TO_PLOT:
            meta = METHOD_META[method]
            fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)

            for r in range(4):
                for c in range(2):
                    task = TASK_GRID[r][c]
                    ax = axes[r, c]
                    seeds = seed_cfg.get(method, [])
                    policy_curves, planner_curves = [], []
                    for seed in seeds:
                        df = _load_task_seed(task, seed)
                        if df.empty:
                            continue
                        policy_curves.append(_interp(df, POLICY_COL, x_grid))
                        planner_curves.append(_interp(df, meta["planner_col"], x_grid))

                    if policy_curves and planner_curves:
                        p_mean, p_se = _mean_se(policy_curves)
                        m_mean, m_se = _mean_se(planner_curves)

                        ax.plot(x_grid, p_mean, ls="--", lw=2.0, color="#4d4d4d", label="Policy network")
                        ax.fill_between(x_grid, p_mean - p_se, p_mean + p_se, color="#8d8d8d", alpha=0.2)

                        ax.plot(x_grid, m_mean, lw=2.0, color=meta["color"], label=meta["label"])
                        ax.fill_between(x_grid, m_mean - m_se, m_mean + m_se, color=meta["color"], alpha=0.2)

                        all_vals = np.concatenate([p_mean[np.isfinite(p_mean)], m_mean[np.isfinite(m_mean)]])
                        ymax = (float(np.max(all_vals)) + 0.01) if all_vals.size else 0.05
                    else:
                        ymax = 0.05

                    ymin = 0.0
                    if task in Y_BOUNDS_OVERRIDE:
                        ymin, ymax = Y_BOUNDS_OVERRIDE[task]
                    else:
                        ymax = max(ymax, 0.05)

                    ax.set_ylim(ymin, ymax)
                    ax.set_yticks([ymin, ymax])
                    ax.set_title(_prettify_task(task), fontsize=12)
                    ax.grid(alpha=0.2, ls=":")

                    if c == 0:
                        ax.set_ylabel("selected-action squared difference")
                    if r == 3:
                        ax.set_xlabel("Time steps (1M)")
                        ax.set_xticks(X_TICKS)
                        ax.set_xticklabels(X_TICK_LABELS)

            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.98))
            fig.suptitle(f"Action drift curves: {meta['label']}", fontsize=16, y=0.995)
            fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved plot to {OUT_PATH}")


if __name__ == "__main__":
    main()
