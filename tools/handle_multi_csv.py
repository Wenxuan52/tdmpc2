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


def _coarsen_step_granularity(df: pd.DataFrame, step_bucket: int, task: str | None = None) -> pd.DataFrame:
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

        window_size = 500 if task == "hopper-stand" else 10
        out_rows = []
        for target in target_steps:
            nearest_idx = np.argsort(np.abs(steps - target))[:window_size]
            if nearest_idx.size == 0:
                continue
            pooled_reward = float(np.nanmax(rewards[nearest_idx]))
            pooled_seed = int(seeds[nearest_idx[0]])
            out_rows.append({"step": int(target), "reward": pooled_reward, "seed": pooled_seed})
        df = pd.DataFrame(out_rows, columns=["step", "reward", "seed"])
    return df[["step", "reward", "seed"]]



def _align_to_step_grid_without_pooling(df: pd.DataFrame, step_bucket: int) -> pd.DataFrame:
    """Align a single-seed curve to bucketed steps without local max pooling."""
    for col in ("step", "reward", "seed"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["step", "reward", "seed"])
    if df.empty:
        return pd.DataFrame(columns=["step", "reward", "seed"])

    if step_bucket <= 0:
        return df[["step", "reward", "seed"]]

    df = df.sort_values("step").reset_index(drop=True)
    steps = df["step"].to_numpy(dtype=float)
    rewards = df["reward"].to_numpy(dtype=float)
    seed = int(pd.to_numeric(df["seed"], errors="coerce").dropna().iloc[0])

    start = int(np.floor(steps.min() / step_bucket) * step_bucket)
    end = int(np.floor(steps.max() / step_bucket) * step_bucket)
    target_steps = np.arange(start, end + step_bucket, step_bucket, dtype=int)

    series = pd.Series(rewards, index=steps)
    aligned = series.reindex(target_steps.astype(float)).interpolate(method="index", limit_area="inside")

    out_rows = [
        {"step": int(step), "reward": float(reward), "seed": seed}
        for step, reward in zip(target_steps, aligned.to_numpy(dtype=float))
        if np.isfinite(reward)
    ]
    return pd.DataFrame(out_rows, columns=["step", "reward", "seed"])

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
        if task == "humanoid-walk" and int(seed) == 3:
            aligned = _align_to_step_grid_without_pooling(norm, step_bucket=step_bucket)
            parts.append(aligned)
        else:
            coarse = _coarsen_step_granularity(norm, step_bucket=step_bucket, task=task)
            parts.append(coarse)

    if not parts:
        return pd.DataFrame(columns=["step", "reward", "seed"])
    return pd.concat(parts, ignore_index=True)
