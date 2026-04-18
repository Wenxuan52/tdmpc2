from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# Keep text editable in vector editors.
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'DejaVu Sans'


def build_colors() -> list[str]:
    """Pastel palette with higher distinction for beta=0.0, 0.1, and 5.0."""
    return [
        '#cfe8a9',  # beta=0.0  light green
        '#ffd9a8',  # beta=0.1  light apricot
        '#7fd54c',  # beta=0.2
        '#5da7df',  # beta=0.3
        '#8a6bc7',  # beta=0.5
        '#5ad7c3',  # beta=0.8
        '#f2a65a',  # beta=1.0  extra similar warm tone
        '#f6bfd8',  # beta=5.0  light pink
    ]


def main() -> None:
    betas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 5.0]
    means = [76.29, 79.15, 78.84, 81.39, 79.82, 79.37, 80.89, 77.50]
    stds = [1.94, 0.59, 0.83, 0.33, 0.97, 1.40, 0.81, 0.51]

    x = np.arange(len(betas))
    colors = build_colors()

    fig, ax = plt.subplots(figsize=(8.6, 5.6))

    bars = ax.bar(
        x,
        means,
        width=0.82,
        color=colors,
        edgecolor='none',
        yerr=stds,
        ecolor='black',
        capsize=4,
        error_kw={'elinewidth': 1.3, 'capthick': 1.3},
    )

    # Axis styling.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['bottom'].set_linewidth(1.6)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{b:.1f}' for b in betas], fontsize=12)
    ax.set_xlabel(r'$\beta$', fontsize=16)
    ax.set_ylabel('Score', fontsize=18)
    ax.set_title(r'$\beta$ Ablation', fontsize=20, fontweight='bold', pad=12)

    # Truncated y-axis to emphasize relative differences.
    ymin = min(m - s for m, s in zip(means, stds)) - 1.0
    ymax = max(m + s for m, s in zip(means, stds)) + 1.2
    ymin = float(np.floor(ymin))
    ymax = float(np.ceil(ymax))
    ax.set_ylim(70, 85)
    ax.tick_params(axis='both', labelsize=12, width=1.4, length=6)

    # Mean labels inside bars, rotated by 90 degrees.
    visible_base = ax.get_ylim()[0]
    for rect, value in zip(bars, means):
        y_pos = visible_base + (value - visible_base) * 0.42
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            y_pos,
            f'{value:.2f}',
            ha='center',
            va='center',
            rotation=90,
            fontsize=11,
            fontweight='bold',
            color='#2f2f2f',
        )

    fig.tight_layout()

    out_path = Path('/root/workspace/tdmpc2/figures/ablation.pdf')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved figure to: {out_path}')


if __name__ == '__main__':
    main()
