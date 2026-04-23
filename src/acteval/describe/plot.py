"""Population-level visualisation helpers for activity schedule data."""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from acteval.describe.utils import _to_population
from acteval.population import Population

_DEFAULT_PALETTE = [
    "#5b8dd9",
    "#d95b5b",
    "#e8a838",
    "#5db85d",
    "#a05bb8",
    "#5bbcbc",
    "#e07b39",
    "#7b5ea7",
    "#3aa3a3",
    "#c45b8d",
]

# Palette for distinguishing multiple populations in the same plot
_POPULATION_PALETTE = [
    "#4e9af1",
    "#e05c5c",
    "#6dbf67",
    "#e8a838",
    "#a05bb8",
    "#5bbcbc",
]

BG = "#f2f2f2"


def _style_ax(ax, bg: str = BG):
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _subplots(*args, bg: str = BG, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(bg)
    for ax in np.atleast_1d(axes).flat:
        _style_ax(ax, bg)
    return fig, axes


def _save(fig, name: str, output_dir: Path | None):
    if output_dir is None:
        return
    path = Path(output_dir) / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved {path}")
    plt.close(fig)


def _build_act_colors(acts: list[str], colors: dict[str, str] | None) -> dict[str, str]:
    """Return a color mapping for the given acts.

    Uses ``colors`` for any acts that appear in it; assigns palette colors to
    the rest in the order they appear in ``acts``.
    """
    colors = colors or {}
    palette_iter = iter(c for c in _DEFAULT_PALETTE if c not in colors.values())
    result = {}
    for act in acts:
        if act in colors:
            result[act] = colors[act]
        else:
            result[act] = next(palette_iter, "#aaaaaa")
    return result


def gantt(
    populations: dict[str, "Population | pd.DataFrame"],
    sample_pids: list | None = None,
    name: str = "gantt.svg",
    output_dir: Path | None = None,
    include_pids: bool = False,
    act_colors: dict[str, str] | None = None,
    acts: list[str] | None = None,
    bg: str = BG,
) -> plt.Figure:
    """Draw a Gantt chart for one or more activity schedule populations.

    Parameters
    ----------
    populations:
        Mapping of label → Population or DataFrame with columns ``pid``,
        ``act``, ``start``, ``duration``.  Each population is drawn in its
        own row of subplots.
    sample_pids:
        Person IDs to display (default: first 8 original person IDs from the
        first population).
    name:
        File name used when saving (e.g. ``"fig1-gantt.svg"``).
    output_dir:
        Directory to save the figure.  If ``None`` the figure is not saved.
    include_pids:
        Whether to label the y-axis with person IDs.
    act_colors:
        Optional mapping of activity name → hex colour.  Any activities not
        present in this mapping receive colours from the built-in palette.
    acts:
        Ordered list of activities to include in the legend.  Defaults to all
        unique activities found across all populations, sorted alphabetically.
    bg:
        Background colour for axes and figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pops = {label: _to_population(v) for label, v in populations.items()}

    if sample_pids is None:
        first_pop = next(iter(pops.values()))
        sample_pids = first_pop.unique_pids_original[:8].tolist()

    if acts is None:
        all_acts: set[str] = set()
        for pop in pops.values():
            all_acts.update(pop.unique_acts)
        acts = sorted(all_acts)

    colors = _build_act_colors(acts, act_colors)

    n_pops = len(pops)
    n_persons = len(sample_pids)

    fig, axes = plt.subplots(
        n_pops,
        1,
        figsize=(12, (1.2 + (n_pops * (0.6 + (0.2 * n_persons))))),
        sharex=True,
    )
    fig.patch.set_facecolor(bg)

    for ax, (label, pop) in zip(np.atleast_1d(axes), pops.items()):
        _style_ax(ax, bg)
        orig_pids_per_row = pop.unique_pids_original[pop.pids]
        for y, pid in enumerate(sample_pids):
            mask = orig_pids_per_row == pid
            for start, duration, act in zip(
                pop.starts[mask], pop.durations[mask], pop.acts[mask]
            ):
                ax.broken_barh(
                    [(start, duration)],
                    (y - 0.4, 0.8),
                    facecolors=colors.get(act, "#aaaaaa"),
                    edgecolor="white",
                    linewidth=0.5,
                )
        ax.set_xlim(0, 1440)

        if include_pids:
            ax.set_yticks(range(n_persons))
            ax.set_yticklabels([str(p) for p in sample_pids], fontsize=10)
        else:
            ax.set_yticks([])

        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="y", left=False)

    bottom_ax = np.atleast_1d(axes)[-1]
    bottom_ax.set_xticks([0, 180, 360, 540, 720, 900, 1080, 1260, 1440])
    bottom_ax.set_xticklabels(
        [
            "00:00",
            "03:00",
            "06:00",
            "09:00",
            "12:00",
            "15:00",
            "18:00",
            "21:00",
            "24:00",
        ],
        fontsize=10,
    )

    legend_handles = [mpatches.Patch(color=colors[a], label=a) for a in acts]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(acts), 6),
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    if output_dir is not None:
        path = Path(output_dir) / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved {path}")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Stacked area / bar — population-average time use
# ---------------------------------------------------------------------------

_X_TICKS = [0, 180, 360, 540, 720, 900, 1080, 1260, 1440]
_X_LABELS = [
    "00:00",
    "03:00",
    "06:00",
    "09:00",
    "12:00",
    "15:00",
    "18:00",
    "21:00",
    "24:00",
]


def _time_use(pop: Population, step: int = 10) -> pd.DataFrame:
    minutes = np.arange(0, 1440, step)
    records = []
    for t in minutes:
        mask = (pop.starts <= t) & (pop.ends > t)
        active_acts = pop.acts[mask]
        active_pids = pop.pids[mask]
        row = {}
        for act in pop.unique_acts:
            act_mask = active_acts == act
            row[act] = np.unique(active_pids[act_mask]).size / pop.n
        records.append(row)
    return pd.DataFrame(records, index=minutes).fillna(0)


def _draw_timeuse_ax(ax, pop: Population, acts, colors, step, bar_step):
    tu = _time_use(pop, step=step if bar_step is None else bar_step)
    present_acts = sorted(
        (a for a in acts if a in tu.columns), key=lambda a: tu[a].sum()
    )

    if bar_step is None:
        ax.stackplot(
            tu.index,
            [tu[a] for a in present_acts],
            colors=[colors[a] for a in present_acts],
            labels=present_acts,
        )
    else:
        bottom = np.zeros(len(tu))
        for act in present_acts:
            ax.bar(
                tu.index,
                tu[act].values,
                width=bar_step * 0.98,
                bottom=bottom,
                color=colors[act],
                label=act,
                edgecolor="white",
                linewidth=0.4,
                align="edge",
            )
            bottom += tu[act].values

    ax.set_xlim(0, 1440)
    ax.set_ylim(0, 1)
    ax.set_xticks(_X_TICKS)
    ax.set_xticklabels(_X_LABELS, fontsize=9)
    ax.set_xlabel("Time of day", fontsize=9, labelpad=4)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
    ax.tick_params(length=0)


def timeuse(
    populations: dict[str, "Population | pd.DataFrame"],
    acts: list[str] | None = None,
    act_colors: dict[str, str] | None = None,
    step: int = 10,
    time_step: int | None = None,
    name: str = "timeuse.svg",
    output_dir: Path | None = None,
    bg: str = BG,
) -> plt.Figure:
    """Stacked area or bar chart of population-average time use across the day.

    Parameters
    ----------
    populations:
        Mapping of label → Population or DataFrame with columns ``pid``,
        ``act``, ``start``, ``end``.
    acts:
        Ordered list of activities to stack.  Defaults to all activities found
        across all populations, sorted alphabetically.
    act_colors:
        Optional mapping of activity name → hex colour.  Unspecified activities
        receive colours from the built-in palette.
    step:
        Time resolution in minutes for the stacked area chart (default: 10).
        Ignored when ``bar_step`` is set.
    bar_step:
        When set, aggregate time use into bins of this width (in minutes) and
        render as stacked bars instead of a stacked area.  Default: ``None``
        (use stacked area).
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
    pops = {label: _to_population(v) for label, v in populations.items()}

    if acts is None:
        all_acts: set[str] = set()
        for pop in pops.values():
            all_acts.update(pop.unique_acts)
        acts = sorted(all_acts)

    colors = _build_act_colors(acts, act_colors)

    fig, axes = _subplots(1, len(pops), figsize=(5 * len(pops), 4), sharey=True, bg=bg)
    for ax, (label, pop) in zip(np.atleast_1d(axes), pops.items()):
        _draw_timeuse_ax(ax, pop, acts, colors, step, time_step)
        ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
    np.atleast_1d(axes)[0].set_ylabel("Fraction of population", fontsize=9, labelpad=6)

    legend_handles = [mpatches.Patch(color=colors[a], label=a) for a in acts]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(acts), 6),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    _save(fig, name, output_dir)
    return fig


# ---------------------------------------------------------------------------
# Participation rate bar chart
# ---------------------------------------------------------------------------


def participation(
    populations: dict[str, "Population | pd.DataFrame"],
    acts: list[str] | None = None,
    palette: list[str] | None = None,
    name: str = "participation.svg",
    output_dir: Path | None = None,
    bg: str = BG,
) -> plt.Figure:
    """Bar chart of activity participation rates across multiple populations.

    Parameters
    ----------
    populations:
        Mapping of label → Population or DataFrame with columns ``pid``, ``act``.
    acts:
        Activities to plot.  Defaults to all activities found across all
        populations, sorted alphabetically.
    palette:
        Colours for each population in order.  Defaults to the built-in
        population palette.
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
    pops = {label: _to_population(v) for label, v in populations.items()}

    if acts is None:
        all_acts: set[str] = set()
        for pop in pops.values():
            all_acts.update(pop.unique_acts)
        acts = sorted(all_acts)

    palette = palette or _POPULATION_PALETTE

    def _rates(pop: Population) -> pd.Series:
        rates = {}
        for act in acts:
            if act in pop.unique_acts:
                idx = list(pop.unique_acts).index(act)
                rates[act] = pop.act_count_matrix[:, idx].mean()
            else:
                rates[act] = 0.0
        return pd.Series(rates)

    rate_df = pd.DataFrame({label: _rates(pop) for label, pop in pops.items()})
    x = np.arange(len(acts))
    width = 0.6 / len(populations)

    fig, ax = _subplots(figsize=(max(6, len(acts) * 1.2), 4), bg=bg)
    for i, (label, col) in enumerate(rate_df.items()):
        ax.bar(
            x + i * width,
            col.values,
            width,
            label=label,
            color=palette[i % len(palette)],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    ax.set_xticks(x + width * (len(populations) - 1) / 2)
    ax.set_xticklabels(acts, fontsize=9)
    ax.set_ylabel("Participation rate", fontsize=9, labelpad=6)
    ax.yaxis.grid(True, color="white", linewidth=1.0, zorder=0)
    ax.tick_params(length=0)

    if len(populations) > 1:
        ax.legend(fontsize=9, frameon=False)

    fig.tight_layout()
    _save(fig, name, output_dir)
    return fig


# ---------------------------------------------------------------------------
# Bigram transition heatmap
# ---------------------------------------------------------------------------


def bigrams(
    populations: dict[str, "Population | pd.DataFrame"],
    acts: list[str] | None = None,
    name: str = "bigrams.svg",
    output_dir: Path | None = None,
    bg: str = BG,
) -> plt.Figure:
    """Heatmap of population-average bigram (activity-to-activity) transition rates.

    Parameters
    ----------
    populations:
        Mapping of label → Population or DataFrame with columns ``pid``,
        ``act``, ``start``.
    acts:
        Ordered list of activities for heatmap rows/columns.  Defaults to all
        activities found across all populations, sorted alphabetically.
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
    from acteval.features.transitions import ngrams

    pops = {label: _to_population(v) for label, v in populations.items()}

    if acts is None:
        all_acts: set[str] = set()
        for pop in pops.values():
            all_acts.update(pop.unique_acts)
        acts = sorted(all_acts)

    act_idx = {a: i for i, a in enumerate(acts)}

    def _matrix(pop: Population) -> np.ndarray:
        bigrams_data = ngrams(pop, n=2).aggregate()
        mat = np.zeros((len(acts), len(acts)))
        for key, (_, counts) in bigrams_data.items():
            parts = key.split(">")
            if len(parts) == 2 and parts[0] in act_idx and parts[1] in act_idx:
                mat[act_idx[parts[0]], act_idx[parts[1]]] = counts.sum() / pop.n
        return mat

    fig, axes = _subplots(1, len(pops), figsize=(5.5 * len(pops), 4.5), bg=bg)
    for ax, (label, pop) in zip(np.atleast_1d(axes), pops.items()):
        sns.heatmap(
            _matrix(pop),
            annot=True,
            fmt=".2f",
            xticklabels=acts,
            yticklabels=acts,
            cmap="Blues",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar=False,
            square=True,
            linewidths=0.8,
            linecolor="white",
            annot_kws={"size": 8},
        )
        ax.set_title(label, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("to", fontsize=9, labelpad=6)
        ax.set_ylabel("from", fontsize=9, labelpad=6)
        ax.tick_params(axis="x", rotation=30, labelsize=8, length=0)
        ax.tick_params(axis="y", rotation=0, labelsize=8, length=0)
    fig.tight_layout()
    _save(fig, name, output_dir)
    return fig


# ---------------------------------------------------------------------------
# Sequence length bar chart
# ---------------------------------------------------------------------------


def sequence_lengths(
    populations: dict[str, "Population | pd.DataFrame"],
    palette: list[str] | None = None,
    name: str = "sequence_lengths.svg",
    output_dir: Path | None = None,
    bg: str = BG,
) -> plt.Figure:
    """Overlaid bar chart of per-person sequence length distributions.

    Parameters
    ----------
    populations:
        Mapping of label → Population or DataFrame with columns ``pid``, ``act``.
    palette:
        Colours for each population in order.  Defaults to the built-in
        population palette.
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
    from acteval.features.participation import sequence_lengths as _seq_lengths

    palette = palette or _POPULATION_PALETTE

    # collect (lengths_array, proportions_array) per population
    series: list[tuple[str, np.ndarray, np.ndarray]] = []
    all_lengths: set[int] = set()
    for label, df_or_pop in populations.items():
        pop = _to_population(df_or_pop)
        vals, counts = _seq_lengths(pop).aggregate()["sequence lengths"]
        props = counts / counts.sum()
        series.append((label, vals.astype(int), props))
        all_lengths.update(vals.astype(int).tolist())

    x_min, x_max = min(all_lengths), max(all_lengths)
    x_all = np.arange(x_min, x_max + 1)

    n = len(series)
    bar_width = 0.8 / n

    fig, ax = _subplots(figsize=(max(6, len(x_all) * 0.5 * n), 4), bg=bg)

    for i, (label, vals, props) in enumerate(series):
        offset = (i - (n - 1) / 2) * bar_width
        # expand to the full x range, filling gaps with 0
        prop_full = np.zeros(len(x_all))
        for v, p in zip(vals, props):
            prop_full[v - x_min] = p
        ax.bar(
            x_all + offset,
            prop_full,
            width=bar_width,
            align="center",
            color=palette[i % len(palette)],
            alpha=0.75,
            edgecolor="white",
            linewidth=0.5,
            label=label,
            zorder=3,
        )

    ax.set_xticks(x_all)
    ax.set_xticklabels(x_all.tolist(), fontsize=9)
    ax.set_xlabel("Activities per person", fontsize=9, labelpad=4)
    ax.set_ylabel("Proportion", fontsize=9, labelpad=6)
    ax.yaxis.grid(True, color="white", linewidth=1.0, zorder=0)
    ax.tick_params(length=0)
    if len(populations) > 1:
        ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    _save(fig, name, output_dir)
    return fig
