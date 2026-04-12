#!/usr/bin/env python3
"""Utilities for reading per-seed MetaWorld training logs and extracting success curves."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

# Example train line fragment:
# train ... I: 100 ... S: 0.0 ...
TRAIN_LINE_RE = re.compile(r"\bI:\s*([\d,]+).*?\bS:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _parse_single_seed_log(path: Path, seed: int) -> pd.DataFrame:
    """Parse one `seed_{seed}.log` into columns: step, reward, seed.

    `reward` is used as the unified metric name for compatibility with plotting code,
    and corresponds to Success rate (%).
    """
    if not path.exists():
        return pd.DataFrame(columns=["step", "reward", "seed"])

    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = TRAIN_LINE_RE.search(line)
            if not m:
                continue
            step = int(m.group(1).replace(",", ""))
            success = float(m.group(2))
            rows.append({"step": step, "reward": success, "seed": seed})

    if not rows:
        return pd.DataFrame(columns=["step", "reward", "seed"])

    df = pd.DataFrame(rows, columns=["step", "reward", "seed"])
    for col in ("step", "reward", "seed"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["step", "reward", "seed"])
    if df.empty:
        return pd.DataFrame(columns=["step", "reward", "seed"])

    # Convert success ratio [0, 1] to percentage [0, 100].
    max_abs = np.nanmax(np.abs(df["reward"].to_numpy(dtype=float)))
    if np.isfinite(max_abs) and max_abs <= 1.5:
        df["reward"] = df["reward"] * 100.0

    df["step"] = df["step"].astype(int)
    df["seed"] = df["seed"].astype(int)
    return df.sort_values("step").drop_duplicates(["step", "seed"], keep="last")


def _coarsen_success_curve(df: pd.DataFrame, step_bucket: int, window_size: int = 10) -> pd.DataFrame:
    """Downsample a single-seed curve by local max pooling around bucketed steps."""
    if df.empty:
        return df

    steps = df["step"].to_numpy(dtype=float)
    rewards = df["reward"].to_numpy(dtype=float)
    seed = int(df["seed"].iloc[0])

    start = int(np.floor(steps.min() / step_bucket) * step_bucket)
    end = int(np.floor(steps.max() / step_bucket) * step_bucket)
    targets = np.arange(start, end + step_bucket, step_bucket, dtype=int)

    out_rows = []
    for target in targets:
        nearest_idx = np.argsort(np.abs(steps - target))[:window_size]
        if nearest_idx.size == 0:
            continue
        pooled_reward = float(np.nanmax(rewards[nearest_idx]))
        out_rows.append({"step": int(target), "reward": pooled_reward, "seed": seed})

    return pd.DataFrame(out_rows, columns=["step", "reward", "seed"])


def load_task_seed_logs(
    root: Path,
    task: str,
    seeds: Iterable[int],
    step_bucket: int = 100_000,
    window_size: int = 10,
) -> pd.DataFrame:
    """Load MetaWorld logs from `{root}/{task}/seed_{seed}.log` for the given seeds."""
    parts: List[pd.DataFrame] = []
    for seed in seeds:
        log_path = root / task / f"seed_{seed}.log"
        parsed = _parse_single_seed_log(log_path, seed=seed)
        if parsed.empty:
            continue
        coarse = _coarsen_success_curve(parsed, step_bucket=step_bucket, window_size=window_size)
        parts.append(coarse)

    if not parts:
        return pd.DataFrame(columns=["step", "reward", "seed"])

    out = pd.concat(parts, ignore_index=True)
    out["step"] = pd.to_numeric(out["step"], errors="coerce").astype(int)
    out["reward"] = pd.to_numeric(out["reward"], errors="coerce")
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype(int)
    return out.dropna(subset=["step", "reward", "seed"]).sort_values(["seed", "step"])
