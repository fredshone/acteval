from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from acteval.describe.utils import _to_population
from acteval.features.frequency import binned_activity_density


def frequency_plots(observed, ys: Optional[dict], **kwargs):
    if ys is None:
        ys = dict()
    pop_obs = _to_population(observed)
    act_order = np.argsort(pop_obs.act_count_matrix.sum(0))[::-1]
    acts = [pop_obs.unique_acts[i] for i in act_order]
    class_map = {n: i for i, n in enumerate(acts)}

    n_plots = len(ys) + 2
    ratios = [1 for _ in range(n_plots)]
    ratios[-1] = 0.3

    cmap = kwargs.pop("cmap", None)
    if cmap is None:
        cmap = plt.cm.Set3
        colors = cmap.colors
        factor = (len(acts) // len(colors)) + 1
        cmap = dict(zip(acts, colors * factor))

    fig, axs = plt.subplots(
        sharex=True,
        sharey=True,
        nrows=1,
        ncols=n_plots,
        constrained_layout=True,
        figsize=kwargs.pop("figsize", (15, 4)),
        gridspec_kw={"width_ratios": ratios},
    )

    name = kwargs.pop("observed_title", "Observed")

    plot_agg_acts(name, pop_obs, class_map, ax=axs[0], legend=False, **kwargs)

    for i, (name, y) in enumerate(ys.items()):
        ax = axs[i + 1]
        plot_agg_acts(name, _to_population(y), class_map, ax=ax, legend=False, **kwargs)

    elements = [Patch(facecolor=cmap[act], label=act.title()) for act in acts]
    axs[-1].axis("off")
    axs[-1].legend(handles=elements, loc="center left", frameon=False)

    return fig


def plot_agg_acts(
    name: str,
    population,
    class_map: dict,
    duration: int = 1440,
    step: int = 10,
    ax=None,
    legend=True,
    **kwargs,
):
    interval = kwargs.pop("interval", 240)
    pop = _to_population(population)
    df = pd.DataFrame({
        "pid": pop.pids,
        "act": pop.acts,
        "start": pop.starts,
        "end": pop.ends,
        "duration": pop.durations,
    })
    bins = binned_activity_density(
        df, duration=duration, step=step, class_map=class_map
    )
    columns = list(class_map.keys())
    totals = bins.sum(0)
    sorted_cols = [x for _, x in sorted(zip(totals, columns))]
    df_plot = pd.DataFrame(bins, columns=columns)[sorted_cols]
    df_plot.index = [
        datetime(2021, 11, 1, 0) + timedelta(minutes=i * step)
        for i in range(len(df_plot.index))
    ]
    fig = df_plot.plot(kind="bar", stacked=True, width=1, ax=ax, legend=legend, **kwargs)
    if legend:
        ax.legend(loc="upper right")
    ax = fig.axes
    ax.set_xticks(
        [i / step for i in [0, 240, 480, 720, 960, 1200, 1440]],
        labels=["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"],
        rotation=90,
        fontsize=8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_xlabel("Time of day")
    ax.set_ylabel("Activity Proportion")
    ax.set_title(name)
    return ax
