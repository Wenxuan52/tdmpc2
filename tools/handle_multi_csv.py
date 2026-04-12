#!/usr/bin/env python3
"""Utilities for reading per-seed multi-CSV experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

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
    """Downsample dense logging by keeping the last point in each step bucket."""
    for col in ("step", "reward", "seed"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["step", "reward", "seed"])
    if df.empty:
        return pd.DataFrame(columns=["step", "reward", "seed"])

    if step_bucket > 0:
        # Avoid creating duplicate "step" columns after grouping: compute bucketed step
        # and explicitly aggregate reward/seed onto that new step key.
        df["step"] = (df["step"] // step_bucket).astype(int) * step_bucket
        df = (
            df.sort_values("step")
            .groupby("step", as_index=False)
            .agg({"reward": "last", "seed": "last"})
        )
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
