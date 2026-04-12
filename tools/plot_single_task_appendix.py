#!/usr/bin/env python3
"""TD-MPC2 Appendix D style single-task plotting utility.

Usage examples:
  python tools/plot_single_task_appendix.py
  python tools/plot_single_task_appendix.py --categories dmcontrol metaworld
  python tools/plot_single_task_appendix.py --skip-missing
  python tools/plot_single_task_appendix.py --only-coverage-report

This script ingests CSV logs from a modified TD-MPC2 method and baseline methods,
normalizes them into a long-form table, computes mean +/- 95% CI across seeds with
in-range interpolation only, and saves publication-style single-task figures.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

DEFAULT_OUR_ROOT = Path("/media/datasets/cheliu21/cxy_worldmodel/online_csv")
DEFAULT_BASELINE_ROOT = Path("/root/workspace/tdmpc2/results")
DEFAULT_OUTDIR = Path("figures/single_task")

METHOD_ORDER = ["sac", "dreamerv3", "tdmpc", "tdmpc2", "tdmpc2_mod"]
EXPECTED_SEEDS = 3


@dataclass(frozen=True)
class CategoryDef:
    name: str
    domain: str
    tasks: List[str]
    xlim: int
    ylabel: str
    nrows: int
    ncols: int


def get_category_definitions() -> Dict[str, CategoryDef]:
    """Return all task groups in the required exact order."""
    dmcontrol_tasks = [
        "acrobot-swingup", "cartpole-balance", "cartpole-balance-sparse", "cartpole-swingup",
        "cartpole-swingup-sparse", "cheetah-jump", "cheetah-run", "cheetah-run-back",
        "cheetah-run-backwards", "cheetah-run-front", "cup-catch", "cup-spin", "dog-run",
        "dog-stand", "dog-trot", "dog-walk", "finger-spin", "finger-turn-easy",
        "finger-turn-hard", "fish-swim", "hopper-hop", "hopper-hop-backwards", "hopper-stand",
        "humanoid-run", "humanoid-stand", "humanoid-walk", "pendulum-spin", "pendulum-swingup",
        "quadruped-run", "quadruped-walk", "reacher-easy", "reacher-hard",
        "reacher-three-easy", "reacher-three-hard", "walker-run", "walker-run-backwards",
        "walker-stand", "walker-walk", "walker-walk-backwards",
    ]
    locomotion_tasks = [
        "dog-run", "dog-stand", "dog-trot", "dog-walk", "humanoid-run", "humanoid-stand", "humanoid-walk",
    ]
    metaworld_tasks = [
        "mw-assembly", "mw-basketball", "mw-bin-picking", "mw-box-close", "mw-button-press-topdown-wall",
        "mw-button-press-topdown", "mw-button-press-wall", "mw-button-press", "mw-coffee-button",
        "mw-coffee-pull", "mw-coffee-push", "mw-dial-turn", "mw-disassemble", "mw-door-close",
        "mw-door-lock", "mw-door-open", "mw-door-unlock", "mw-drawer-close", "mw-drawer-open",
        "mw-faucet-close", "mw-faucet-open", "mw-hammer", "mw-hand-insert", "mw-handle-press-side",
        "mw-handle-press", "mw-handle-pull-side", "mw-handle-pull", "mw-lever-pull", "mw-peg-insert-side",
        "mw-peg-unplug-side", "mw-pick-out-of-hole", "mw-pick-place-wall", "mw-pick-place",
        "mw-plate-slide-back-side", "mw-plate-slide-back", "mw-plate-slide-side", "mw-plate-slide",
        "mw-push-back", "mw-push-wall", "mw-push", "mw-reach-wall", "mw-reach", "mw-shelf-place",
        "mw-soccer", "mw-stick-pull", "mw-stick-push", "mw-sweep-into", "mw-sweep", "mw-window-close",
        "mw-window-open",
    ]
    myosuite_tasks = [
        "myo-key-turn", "myo-key-turn-hard", "myo-obj-hold", "myo-obj-hold-hard", "myo-pen-twirl",
        "myo-pen-twirl-hard", "myo-pose", "myo-pose-hard", "myo-reach", "myo-reach-hard",
    ]

    return {
        "dmcontrol": CategoryDef("dmcontrol", "DMControl", dmcontrol_tasks, 4_000_000, "Episode return", 8, 5),
        "metaworld": CategoryDef("metaworld", "Meta-World", metaworld_tasks, 2_000_000, "Success rate (%)", 10, 5),
        "myosuite": CategoryDef("myosuite", "MyoSuite", myosuite_tasks, 2_000_000, "Success rate (%)", 2, 5),
        "locomotion": CategoryDef("locomotion", "Locomotion", locomotion_tasks, 14_000_000, "Episode return", 2, 4),
    }


def build_task_domain_map(categories: Dict[str, CategoryDef]) -> Dict[str, str]:
    task_to_domain: Dict[str, str] = {}
    for cat in categories.values():
        for task in cat.tasks:
            task_to_domain[task] = cat.domain
    return task_to_domain


def is_success_domain(domain: str) -> bool:
    return domain in {"Meta-World", "MyoSuite"}


def prettify_task_name(task: str) -> str:
    name = task
    if name.startswith("mw-"):
        name = name[3:]
    if name.startswith("myo-"):
        name = name[4:]
    tokens = name.replace("-", " ").split()
    return " ".join(tok.capitalize() for tok in tokens)


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception as exc:  # recoverable input errors should not crash the script
        print(f"[WARN] Failed to read CSV {path}: {exc}")
        return None


def load_our_method_logs(root: Path, task_to_domain: Dict[str, str]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    if not root.exists():
        print(f"[WARN] Our method root not found: {root}")
        return pd.DataFrame(columns=["method", "domain", "task", "seed", "step", "value"])

    pattern = re.compile(r"^(?P<task>.+)_(?P<seed>\d+)\.csv$")
    for csv_path in sorted(root.glob("*.csv")):
        match = pattern.match(csv_path.name)
        if not match:
            continue
        task = match.group("task")
        domain = task_to_domain.get(task)
        if domain is None:
            continue

        seed_from_name = int(match.group("seed"))
        df = safe_read_csv(csv_path)
        if df is None or df.empty:
            continue
        if "step" not in df.columns or "reward" not in df.columns:
            print(f"[WARN] Missing required columns in {csv_path}, expected at least step,reward")
            continue

        seed_col = df["seed"] if "seed" in df.columns else seed_from_name
        part = pd.DataFrame(
            {
                "method": "tdmpc2_mod",
                "domain": domain,
                "task": task,
                "seed": seed_col,
                "step": df["step"],
                "value": df["reward"],
            }
        )
        rows.append(part)

    if not rows:
        return pd.DataFrame(columns=["method", "domain", "task", "seed", "step", "value"])
    return pd.concat(rows, ignore_index=True)


def load_baseline_logs(root: Path, task_to_domain: Dict[str, str]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for method in METHOD_ORDER:
        if method == "tdmpc2_mod":
            continue
        method_dir = root / method
        if not method_dir.exists():
            print(f"[WARN] Baseline directory not found: {method_dir}")
            continue

        for csv_path in sorted(method_dir.glob("*.csv")):
            task = csv_path.stem
            domain = task_to_domain.get(task)
            if domain is None:
                continue

            df = safe_read_csv(csv_path)
            if df is None or df.empty:
                continue
            if not {"step", "reward", "seed"}.issubset(df.columns):
                print(f"[WARN] Missing columns in {csv_path}, expected step,reward,seed")
                continue

            part = pd.DataFrame(
                {
                    "method": method,
                    "domain": domain,
                    "task": task,
                    "seed": df["seed"],
                    "step": df["step"],
                    "value": df["reward"],
                }
            )
            rows.append(part)

    if not rows:
        return pd.DataFrame(columns=["method", "domain", "task", "seed", "step", "value"])
    return pd.concat(rows, ignore_index=True)


def normalize_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    for col in ["seed", "step", "value"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["method", "domain", "task", "seed", "step", "value"]).copy()
    out["seed"] = out["seed"].astype(int)
    out["step"] = out["step"].astype(float)
    out["value"] = out["value"].astype(float)
    out = out[out["step"] >= 0]

    out = out.sort_values(["method", "task", "seed", "step"]).drop_duplicates(
        subset=["method", "task", "seed", "step"], keep="last"
    )

    success_mask = out["domain"].map(is_success_domain)
    if success_mask.any():
        grouped = out[success_mask].groupby(["method", "task", "seed"], sort=False)
        for (method, task, seed), idx in grouped.groups.items():
            vals = out.loc[idx, "value"]
            finite = vals[np.isfinite(vals)]
            if finite.empty:
                continue
            # Heuristic: if nearly all values are in [0, 1.2], treat as [0,1] success and convert to percent.
            q95 = np.nanpercentile(finite, 95)
            q05 = np.nanpercentile(finite, 5)
            if q95 <= 1.2 and q05 >= -0.2:
                out.loc[idx, "value"] = vals * 100.0

    return out.reset_index(drop=True)


def truncate_to_xlim(df: pd.DataFrame, xlim_by_domain: Dict[str, int]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out = out[out.apply(lambda r: r["step"] <= xlim_by_domain.get(r["domain"], np.inf), axis=1)]
    return out.reset_index(drop=True)


def summarize_seed_curves(group: pd.DataFrame, xlim: int) -> pd.DataFrame:
    """Align seeds on union step grid and compute mean + 95% CI without right-tail extrapolation."""
    if group.empty:
        return pd.DataFrame(columns=["step", "mean", "ci95", "n"])

    steps = np.sort(group["step"].unique())
    steps = steps[steps <= xlim]
    if len(steps) == 0:
        return pd.DataFrame(columns=["step", "mean", "ci95", "n"])

    seed_series = []
    for seed, sdf in group.groupby("seed"):
        sdf = sdf.sort_values("step")
        s = pd.Series(sdf["value"].to_numpy(), index=sdf["step"].to_numpy(), dtype=float)
        s = s[~s.index.duplicated(keep="last")]
        re = s.reindex(steps)
        re = re.interpolate(method="index", limit_area="inside")
        seed_series.append(re.rename(seed))

    aligned = pd.concat(seed_series, axis=1)
    mean = aligned.mean(axis=1, skipna=True)
    std = aligned.std(axis=1, skipna=True, ddof=1)
    n = aligned.count(axis=1)
    ci95 = 1.96 * (std / np.sqrt(n))
    ci95 = ci95.where(n > 1, 0.0)

    out = pd.DataFrame({"step": steps, "mean": mean.values, "ci95": ci95.values, "n": n.values})
    out.loc[out["n"] <= 0, ["mean", "ci95"]] = np.nan
    return out


def generate_coverage_report(
    raw_df: pd.DataFrame,
    categories: Dict[str, CategoryDef],
    methods: Sequence[str],
    expected_seeds: int = EXPECTED_SEEDS,
) -> pd.DataFrame:
    rows = []
    for cat in categories.values():
        for task in cat.tasks:
            for method in methods:
                subset = raw_df[(raw_df["domain"] == cat.domain) & (raw_df["task"] == task) & (raw_df["method"] == method)]
                seeds = sorted(subset["seed"].unique().tolist()) if not subset.empty else []
                seeds_found = len(seeds)
                min_seed = seeds[0] if seeds else np.nan
                max_seed = seeds[-1] if seeds else np.nan
                max_step = float(subset["step"].max()) if not subset.empty else np.nan
                flags: List[str] = []
                if subset.empty:
                    flags.append("missing_task")
                else:
                    if seeds_found < expected_seeds:
                        flags.append("missing_seed")
                    if max_step < cat.xlim:
                        flags.append("insufficient_max_step")
                    if max_step > cat.xlim:
                        flags.append("exceeded_expected_step")

                status = "ok" if not flags else (flags[0] if len(flags) == 1 else "mixed_issues")
                rows.append(
                    {
                        "method": method,
                        "domain": cat.domain,
                        "task": task,
                        "seeds_found": seeds_found,
                        "min_seed": min_seed,
                        "max_seed": max_seed,
                        "max_step": max_step,
                        "expected_max_step": cat.xlim,
                        "status": status,
                    }
                )

    report = pd.DataFrame(rows)
    return report


def print_summary(report: pd.DataFrame, categories: Dict[str, CategoryDef], methods: Sequence[str], expected_seeds: int) -> None:
    print("\n=== Coverage summary ===")
    for cat_key, cat in categories.items():
        print(f"\n[{cat_key}] {cat.domain} ({len(cat.tasks)} tasks, target max step={cat.xlim:,})")
        sub = report[report["domain"] == cat.domain]
        for method in methods:
            msub = sub[sub["method"] == method]
            found = int((msub["seeds_found"] > 0).sum())
            missing = int((msub["status"] == "missing_task").sum())
            fewer_seed = int((msub["seeds_found"] > 0).mul(msub["seeds_found"] < expected_seeds).sum())
            low_step = int((msub["seeds_found"] > 0).mul(msub["max_step"] < msub["expected_max_step"]).sum())
            high_step = int((msub["seeds_found"] > 0).mul(msub["max_step"] > msub["expected_max_step"]).sum())
            print(
                f"  - {method:10s} found={found:2d} missing={missing:2d} "
                f"<seeds({expected_seeds})={fewer_seed:2d} below_x={low_step:2d} above_x={high_step:2d}"
            )

        missing_tasks = sorted(sub.groupby("task")["seeds_found"].sum().loc[lambda s: s == 0].index.tolist())
        if missing_tasks:
            print(f"  * Tasks missing across all methods ({len(missing_tasks)}): {', '.join(missing_tasks[:8])}" + (" ..." if len(missing_tasks) > 8 else ""))


def m_formatter(x: float, _pos: int) -> str:
    if abs(x) < 1e-9:
        return "0"
    return f"{int(round(x / 1_000_000.0))}M"


def plot_category(
    clean_df: pd.DataFrame,
    cat: CategoryDef,
    outdir: Path,
    method_order: Sequence[str],
    skip_missing: bool,
) -> None:
    fig, axes = plt.subplots(cat.nrows, cat.ncols, figsize=(cat.ncols * 3.2, cat.nrows * 2.1), squeeze=False)
    axes_flat = axes.flatten()

    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(method_order)}
    legend_handles = {}

    for i, task in enumerate(cat.tasks):
        ax = axes_flat[i]
        task_df = clean_df[(clean_df["domain"] == cat.domain) & (clean_df["task"] == task)]
        any_curve = False

        for method in method_order:
            mt = task_df[task_df["method"] == method]
            if mt.empty:
                continue
            stats = summarize_seed_curves(mt, xlim=cat.xlim)
            stats = stats.dropna(subset=["mean"])
            if stats.empty:
                continue
            any_curve = True
            (line,) = ax.plot(stats["step"], stats["mean"], lw=1.8, color=colors[method], label=method)
            ax.fill_between(
                stats["step"].to_numpy(dtype=float),
                (stats["mean"] - stats["ci95"]).to_numpy(dtype=float),
                (stats["mean"] + stats["ci95"]).to_numpy(dtype=float),
                color=colors[method],
                alpha=0.18,
                linewidth=0,
            )
            legend_handles[method] = line

        if not any_curve and skip_missing:
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center", alpha=0.45, fontsize=9)

        ax.set_title(prettify_task_name(task), fontsize=9)
        ax.set_xlim(0, cat.xlim)
        ax.xaxis.set_major_formatter(FuncFormatter(m_formatter))
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.3)
        ax.tick_params(labelsize=8)

    # Remove trailing empty axes where layout has one extra slot.
    for j in range(len(cat.tasks), len(axes_flat)):
        fig.delaxes(axes_flat[j])

    fig.supxlabel("Environment steps", fontsize=12)
    fig.supylabel(cat.ylabel, fontsize=12)

    ordered_handles = [legend_handles[m] for m in method_order if m in legend_handles]
    ordered_labels = [m for m in method_order if m in legend_handles]
    if ordered_handles:
        fig.legend(ordered_handles, ordered_labels, loc="lower center", ncol=min(5, len(ordered_handles)), frameon=False, fontsize=10)
        plt.subplots_adjust(bottom=0.08)

    fig.tight_layout(rect=(0, 0.05, 1, 1))

    pdf_path = outdir / f"{cat.name}_single_task.pdf"
    png_path = outdir / f"{cat.name}_single_task.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {pdf_path}")
    print(f"[INFO] Saved {png_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TD-MPC2 Appendix D single-task comparisons from CSV logs.")
    parser.add_argument("--our-root", type=Path, default=DEFAULT_OUR_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["dmcontrol", "metaworld", "myosuite", "locomotion"],
        default=["dmcontrol", "metaworld", "myosuite", "locomotion"],
    )
    parser.add_argument("--skip-missing", action="store_true", help="Annotate fully missing tasks lightly instead of failing.")
    parser.add_argument("--only-coverage-report", action="store_true", help="Only build and save coverage report CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_categories = get_category_definitions()
    selected_categories = {k: all_categories[k] for k in args.categories}
    args.outdir.mkdir(parents=True, exist_ok=True)

    task_to_domain = build_task_domain_map(all_categories)
    raw_ours = load_our_method_logs(args.our_root, task_to_domain)
    raw_baselines = load_baseline_logs(args.baseline_root, task_to_domain)
    raw_df = pd.concat([raw_baselines, raw_ours], ignore_index=True)

    clean_df = normalize_and_clean(raw_df)

    coverage = generate_coverage_report(raw_df=clean_df, categories=selected_categories, methods=METHOD_ORDER)
    coverage_path = args.outdir / "coverage_report.csv"
    coverage.to_csv(coverage_path, index=False)
    print(f"[INFO] Saved {coverage_path}")

    print_summary(coverage, selected_categories, METHOD_ORDER, expected_seeds=EXPECTED_SEEDS)

    if args.only_coverage_report:
        return

    # Strictly truncate values shown beyond each category budget.
    xlim_by_domain = {cat.domain: cat.xlim for cat in selected_categories.values()}
    truncated = truncate_to_xlim(clean_df, xlim_by_domain)

    for cat in selected_categories.values():
        plot_category(
            clean_df=truncated,
            cat=cat,
            outdir=args.outdir,
            method_order=METHOD_ORDER,
            skip_missing=args.skip_missing,
        )


if __name__ == "__main__":
    main()
