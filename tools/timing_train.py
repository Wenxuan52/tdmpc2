#!/usr/bin/env python3
"""Recompute training timing statistics using iterations 2~10.

Usage: edit `CSV_ROOT_DIRS` below and run this script.
No CLI arguments are required.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

# Fill in one or more profiling directories here before running.
CSV_ROOT_DIRS = [
    # "/media/datasets/cheliu21/cxy_worldmodel/profiling",
    # "/media/datasets/cheliu21/cxy_worldmodel/profiling_pixels",
    # "/media/datasets/cheliu21/cxy_worldmodel/profiling_mppi",
    # "/media/datasets/cheliu21/cxy_worldmodel/profiling_mppi_pixels",
]

ITERATION_START = 2
ITERATION_END = 10


def _load_training_values(timing_csv_path: Path) -> List[float]:
    """Load training_cycle_time from iteration 2..10 (inclusive)."""
    values: List[float] = []
    with timing_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"iteration", "training_cycle_time"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"{timing_csv_path} missing required columns {sorted(required_cols)}"
            )

        for row in reader:
            it = int(float(row["iteration"]))
            if ITERATION_START <= it <= ITERATION_END:
                values.append(float(row["training_cycle_time"]))

    if not values:
        raise ValueError(
            f"{timing_csv_path} has no rows in iteration [{ITERATION_START}, {ITERATION_END}]"
        )
    return values


def _mean_std(values: List[float]) -> Tuple[float, float]:
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)  # population std (ddof=0)
    return mean, math.sqrt(var)


def _recompute_domain_training_file(domain_training_path: Path) -> None:
    root = domain_training_path.parent

    rows: List[Dict[str, str]] = []
    with domain_training_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["task_name", "mean", "std"]:
            raise ValueError(
                f"Unexpected columns in {domain_training_path}: {reader.fieldnames}"
            )
        rows = list(reader)

    updated_rows: List[Dict[str, str]] = []
    for row in rows:
        task = row["task_name"]
        timing_path = root / f"{task}_timing.csv"
        if not timing_path.exists():
            raise FileNotFoundError(f"Cannot find timing file for task '{task}': {timing_path}")

        values = _load_training_values(timing_path)
        mean, std = _mean_std(values)
        updated_rows.append(
            {
                "task_name": task,
                "mean": repr(mean),
                "std": repr(std),
            }
        )

    with domain_training_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_name", "mean", "std"])
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"Updated {domain_training_path} ({len(updated_rows)} tasks)")


def recompute_all(root_dir: Path) -> None:
    if not root_dir.exists() or not root_dir.is_dir():
        raise NotADirectoryError(f"Invalid directory: {root_dir}")

    training_files = sorted(root_dir.glob("*_training.csv"))
    if not training_files:
        print(f"No *_training.csv files found in {root_dir}")
        return

    for training_path in training_files:
        _recompute_domain_training_file(training_path)


def main() -> None:
    if not CSV_ROOT_DIRS:
        raise ValueError("Please set CSV_ROOT_DIRS in tools/timing_train.py before running.")

    for root in CSV_ROOT_DIRS:
        recompute_all(Path(root))


if __name__ == "__main__":
    main()
