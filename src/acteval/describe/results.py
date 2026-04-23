"""Visualisation helpers for acteval compare/evaluate results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from acteval.describe.plot import _POPULATION_PALETTE, BG, _save, _subplots


def heatmap(
    result,
    name: str = "heatmap.svg",
    output_dir: Path | None = None,
    bg: str = BG,
) -> plt.Figure:
    """Domain-level distance heatmap for a compare result.

    Parameters
    ----------
    result:
        Return value of ``acteval.compare()``.
    name:
        File name used when saving.
    output_dir:
        Directory to save the figure.  If ``None`` the figure is not saved.
    bg:
        Background colour for axes and figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    summary = result.summary()
    fig, ax = _subplots(
        figsize=(max(4, len(summary.columns) * 1.2), max(3, len(summary) * 0.9)), bg=bg
    )
    sns.heatmap(
        summary,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=0.5,
        ax=ax,
        square=True,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "distance", "shrink": 0.8},
        annot_kws={"size": 9},
    )
    ax.set_title(
        "Domain-level distances from observed", fontsize=11, fontweight="bold", pad=10
    )
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30, labelsize=9, length=0)
    ax.tick_params(axis="y", rotation=0, labelsize=9, length=0)
    fig.tight_layout()
    _save(fig, name, output_dir)
    return fig


def groups(
    result,
    palette: list[str] | None = None,
    name: str = "groups.svg",
    output_dir: Path | None = None,
    bg: str = BG,
) -> plt.Figure:
    """Horizontal bar chart of feature-group distances for a compare result.

    Parameters
    ----------
    result:
        Return value of ``acteval.compare()``.
    palette:
        Colours for each model in order.  Defaults to the built-in population
        palette.
    name:
        File name used when saving.
    output_dir:
        Directory to save the figure.  If ``None`` the figure is not saved.
    bg:
        Background colour for axes and figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    palette = palette or _POPULATION_PALETTE
    gd = result.group_distances[result.model_names]
    # sort rows by mean distance descending so the largest distances appear at top
    gd = (
        gd.assign(_mean=gd.mean(axis=1))
        .sort_values("_mean", ascending=True)
        .drop(columns="_mean")
    )
    fig, ax = _subplots(figsize=(7, max(4, len(gd) * 0.45)), bg=bg)
    y = np.arange(len(gd))
    for i, model in enumerate(result.model_names):
        offset = (i - (len(result.model_names) - 1) / 2) * 0.28
        ax.barh(
            y + offset,
            gd[model],
            height=0.24,
            label=model,
            color=palette[i % len(palette)],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
    labels = [f"{d}  ·  {f}" for d, f in gd.index]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, None)
    ax.set_xlabel("Distance", fontsize=9, labelpad=6)
    ax.xaxis.grid(True, color="white", linewidth=1.0, zorder=0)
    ax.tick_params(length=0)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    _save(fig, name, output_dir)
    return fig
