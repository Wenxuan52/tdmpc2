#!/usr/bin/env python3
"""Utilities for reading per-seed multi-CSV experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


def _normalize_columns(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Normalize possible CSV schemas into columns: step, reward, seed."""
    reward_col = None
    for candidate in ("reward", "episode_reward"):
        if candidate in df.columns:
            reward_col = candidate
            break
    if reward_col is None:
        raise ValueError("CSV must contain one of: reward, episode_reward")
    if "step" not in df.columns:
        raise ValueError("CSV must contain step column")

    out = df[["step", reward_col]].copy()
    out = out.rename(columns={reward_col: "reward"})
    if "seed" in df.columns:
        out["seed"] = df["seed"]
    else:
        out["seed"] = seed
    return out[["step", "reward", "seed"]]


def _coarsen_step_granularity(df: pd.DataFrame, step_bucket: int) -> pd.DataFrame:
    """Downsample dense logging with local-window max pooling around each bucketed step."""
    for col in ("step", "reward", "seed"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["step", "reward", "seed"])
    if df.empty:
        return pd.DataFrame(columns=["step", "reward", "seed"])

    if step_bucket > 0:
        df = df.sort_values("step").reset_index(drop=True)
        steps = df["step"].to_numpy(dtype=float)
        rewards = df["reward"].to_numpy(dtype=float)
        seeds = df["seed"].to_numpy(dtype=float)

        start = int(np.floor(steps.min() / step_bucket) * step_bucket)
        end = int(np.floor(steps.max() / step_bucket) * step_bucket)
        target_steps = np.arange(start, end + step_bucket, step_bucket, dtype=int)

        out_rows = []
        for target in target_steps:
            nearest_idx = np.argsort(np.abs(steps - target))[:10]
            if nearest_idx.size == 0:
                continue
            pooled_reward = float(np.nanmax(rewards[nearest_idx]))
            pooled_seed = int(seeds[nearest_idx[0]])
            out_rows.append({"step": int(target), "reward": pooled_reward, "seed": pooled_seed})
        df = pd.DataFrame(out_rows, columns=["step", "reward", "seed"])
    return df[["step", "reward", "seed"]]


def load_task_seed_csvs(
    root: Path,
    task: str,
    seeds: Iterable[int],
    step_bucket: int = 100_000,
) -> pd.DataFrame:
    """Load `{task}_{seed}.csv` files and return concatenated normalized dataframe."""
    parts: List[pd.DataFrame] = []
    for seed in seeds:
        path = root / f"{task}_{seed}.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        norm = _normalize_columns(raw, seed=seed)
        coarse = _coarsen_step_granularity(norm, step_bucket=step_bucket)
        parts.append(coarse)

    if not parts:
        return pd.DataFrame(columns=["step", "reward", "seed"])
    return pd.concat(parts, ignore_index=True)
