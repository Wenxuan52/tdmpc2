from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


# Keep text editable in vector editors.
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'DejaVu Sans'


def build_colors(n: int) -> list[str]:
    """Use viridis colormap, avoiding colors that are too light."""
    cmap = plt.get_cmap('viridis')

    # Avoid the very light end of viridis.
    color_values = np.linspace(0.30, 0.90, n)

    return [mcolors.to_hex(cmap(v)) for v in color_values]


def main() -> None:
    etas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 5.0]
    means = [75.29, 79.15, 78.84, 81.39, 79.82, 79.37, 80.89, 69.74]
    stds = [1.94, 0.59, 0.83, 0.53, 0.97, 1.40, 0.81, 1.45]

    # Add a larger visual gap before eta=5.0.
    x = np.array([0, 1, 2, 3, 4, 5, 6, 8.3], dtype=float)

    colors = build_colors(len(etas))

    # Make the whole figure shorter.
    fig, ax = plt.subplots(figsize=(8.8, 4.6))

    # Light horizontal grid for readability.
    ax.set_axisbelow(True)
    ax.grid(
        axis='y',
        linestyle='--',
        linewidth=0.8,
        alpha=0.25,
        zorder=0,
    )

    # Bar plot: same mean values as the line plot.
    # No std/error bars on bars. Bars are narrower and use corresponding point colors.
    ax.bar(
        x,
        means,
        width=0.33,
        color=colors,
        alpha=0.55,
        edgecolor='none',
        zorder=1,
    )

    # Main trend line from eta=0.0 to eta=1.0.
    ax.plot(
        x[:-1],
        means[:-1],
        color='#5f5f5f',
        linewidth=2.2,
        alpha=0.75,
        zorder=2,
    )

    # Dashed segment from eta=1.0 to eta=5.0.
    ax.plot(
        x[-2:],
        means[-2:],
        color='#5f5f5f',
        linewidth=2.2,
        linestyle=(0, (4, 3)),
        alpha=0.75,
        zorder=2,
    )

    # Colored points and std error bars for the line plot.
    for xi, mean, std, color in zip(x, means, stds, colors):
        ax.errorbar(
            xi,
            mean,
            yerr=std,
            fmt='o',
            markersize=13,
            markerfacecolor=mcolors.to_rgba(color, 0.96),
            markeredgecolor=mcolors.to_rgba('#222222', 0.55),
            markeredgewidth=1.35,
            ecolor=mcolors.to_rgba('black', 0.85),
            elinewidth=1.8,
            capsize=5.2,
            capthick=1.8,
            zorder=4,
        )

    # Axis styling.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_linewidth(1.3)

    # Hide default x-axis spine, then draw a custom x-axis manually.
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{e:.1f}' for e in etas], fontsize=15)

    ax.set_xlabel(r'$\eta$ Ablation', fontsize=20, labelpad=8)
    ax.set_ylabel('Score', fontsize=20, labelpad=8)

    # Y-axis range and ticks.
    # Set ylim explicitly so bars do not force the y-axis to start from 0.
    ax.set_ylim(65, 83.5)
    ax.set_yticks(np.arange(66, 84, 3))

    # Larger tick labels.
    ax.tick_params(
        axis='both',
        labelsize=15,
        width=1.2,
        length=5,
    )

    # Value labels above the std error bars.
    label_offset = 0.35

    for xi, mean, std in zip(x, means, stds):
        ax.text(
            xi,
            mean + std + label_offset,
            f'{mean:.2f}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='semibold',
            color='#333333',
            zorder=5,
            clip_on=False,
        )

    ax.set_xlim(x[0] - 0.45, x[-1] + 0.55)

    # Custom x-axis:
    # solid from the left edge to eta=1.0,
    # dashed from eta=1.0 to eta=5.0,
    # solid again after eta=5.0.
    x_left, x_right = ax.get_xlim()
    x_eta_1 = x[-2]
    x_eta_5 = x[-1]

    axis_transform = ax.get_xaxis_transform()

    ax.plot(
        [x_left, x_eta_1],
        [0, 0],
        color='black',
        linewidth=1.3,
        transform=axis_transform,
        clip_on=False,
        zorder=6,
    )

    ax.plot(
        [x_eta_1, x_eta_5],
        [0, 0],
        color='black',
        linewidth=1.3,
        linestyle=(0, (4, 3)),
        transform=axis_transform,
        clip_on=False,
        zorder=6,
    )

    ax.plot(
        [x_eta_5, x_right],
        [0, 0],
        color='black',
        linewidth=1.3,
        transform=axis_transform,
        clip_on=False,
        zorder=6,
    )

    fig.tight_layout()

    out_path = Path('/root/workspace/tdmpc2/figures/Ablation_bar.pdf')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f'Saved figure to: {out_path}')


if __name__ == '__main__':
    main()