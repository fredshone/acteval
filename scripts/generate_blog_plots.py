"""Generate all plots for the acteval distances blog post.

Usage:
    uv run python scripts/generate_blog_plots.py [output_dir]

Writes 10 SVG files to output_dir (default: assets/img/acteval/).
Also overridable with the OUTPUT_DIR environment variable.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import ot
import pandas as pd
import seaborn as sns

from acteval import Population, compare, compare_splits
from acteval.describe.plot import (
    _POPULATION_PALETTE,
    _save,
    _subplots,
    bigrams,
    gantt,
    participation,
    sequence_lengths,
    timeuse,
)
from acteval.describe.results import groups, heatmap
from acteval.describe.utils import PopulationGenerator
from acteval.features.participation import sequence_lengths as sequence_lengths_feature
from acteval.features.times import start_times_by_act_plan_enum_per_pid

ACT_COLORS = {
    "home": "#5b8dd9",
    "work": "#d95b5b",
    "eat_out": "#e8a838",
    "shop": "#5db85d",
    "leisure": "#a05bb8",
    "education": "#5bbcbc",
}
ACTS = ["home", "work", "eat_out", "shop", "leisure", "education"]
# ---------------------------------------------------------------------------
# Population generators
# ---------------------------------------------------------------------------

# Observed reference: conventional mixed urban workforce
urban_workers = PopulationGenerator(
    profile_weights={
        "office": 4,
        "flexible": 2,
        "student": 2,
        "part_time": 1,
        "late_worker": 1,
    },
    seed=42,
)

# Population A: education- and study-leaning — more students and part-time workers,
# shorter work blocks, more leisure after structured activities
education_leaning = PopulationGenerator(
    profile_weights={
        "student": 5,
        "part_time": 3,
        "flexible": 2,
        "leisure_shopper": 2,
        "late_worker": 1,
        "office": 1,
    },
    seed=7,
)

# Population B: leisure- and shopping-dominant — fewer workers, more discretionary time
leisure_dominant = PopulationGenerator(
    profile_weights={
        "leisure_shopper": 5,
        "home_heavy": 3,
        "flexible": 2,
        "part_time": 2,
        "late_worker": 1,
        "office": 1,
    },
    seed=13,
)


# ---------------------------------------------------------------------------
# Figure 5: work0 start-time histogram
# ---------------------------------------------------------------------------


def starttimes(
    populations: dict[str, pd.DataFrame],
    name: str = "fig5-starttimes.svg",
    output_dir: Path = None,
):
    print("Fig 5: Work start-time histograms")
    fig, ax = _subplots(figsize=(7, 4))
    for i, (label, df) in enumerate(populations.items()):
        starts = df[df.act == "work"].groupby("pid").start.min().values / 60
        ax.hist(
            starts,
            bins=30,
            alpha=0.55,
            color=_POPULATION_PALETTE[i % len(_POPULATION_PALETTE)],
            label=label,
            density=True,
        )
    ax.set_xlabel("Work start (hour of day)")
    ax.set_ylabel("Density")
    ax.set_xticks(range(4, 16))
    ax.legend()
    fig.tight_layout()
    _save(fig, name, output_dir)


# ---------------------------------------------------------------------------
# Figure 6: 2D scatter (work start, total work duration)
# ---------------------------------------------------------------------------


def joint(
    populations: dict[str, pd.DataFrame],
    name: str = "fig6-joint.svg",
    output_dir: Path = None,
):
    print("Fig 6: (work start, duration) scatter")
    fig, ax = _subplots(figsize=(6, 5))
    for i, (label, df) in enumerate(populations.items()):
        work = (
            df[df.act == "work"]
            .groupby("pid")
            .agg(
                start=("start", "min"),
                duration=("duration", "sum"),
            )
        )
        ax.scatter(
            work.start / 60,
            work.duration / 60,
            alpha=0.25,
            color=_POPULATION_PALETTE[i % len(_POPULATION_PALETTE)],
            label=label,
            s=14,
        )
    ax.set_xlabel("Work start (h)")
    ax.set_ylabel("Total work duration (h)")
    ax.legend()
    fig.tight_layout()
    _save(fig, name, output_dir)


# ---------------------------------------------------------------------------
# Figure 7: EMD transport-plan diagram
# ---------------------------------------------------------------------------


def emd(
    populations: dict[str, pd.DataFrame],
    name: str = "fig7-emd.svg",
    output_dir: Path = None,
):
    print("Fig 7: EMD transport plan")
    pop_labels = list(populations.keys())
    obs_label, model_label = pop_labels[0], pop_labels[1]

    obs_agg = start_times_by_act_plan_enum_per_pid(
        Population(populations[obs_label])
    ).aggregate()
    model_agg = start_times_by_act_plan_enum_per_pid(
        Population(populations[model_label])
    ).aggregate()

    u_vals, u_wts = obs_agg["work0"]
    s_vals, s_wts = model_agg["work0"]

    bin_edges = np.arange(360, 630, 30) / 1440
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    def _to_hist(vals, wts):
        h = np.zeros(len(bin_centres))
        for v, w in zip(vals, wts):
            idx = np.searchsorted(bin_edges[1:], v, side="left")
            if 0 <= idx < len(h):
                h[idx] += w
        total = h.sum()
        return h / total if total > 0 else h

    u_hist = _to_hist(u_vals, u_wts)
    s_hist = _to_hist(s_vals, s_wts)

    M = ot.dist(bin_centres[:, None], bin_centres[:, None], metric="euclidean")
    T = ot.emd(u_hist, s_hist, M)
    emd_val = (T * M).sum()

    fig, axes = _subplots(1, 2, figsize=(12, 4))

    w = (bin_edges[1] - bin_edges[0]) * 1440 / 60 * 0.4
    centres_h = bin_centres * 24
    axes[0].bar(
        centres_h - w / 2,
        u_hist,
        width=w * 0.9,
        alpha=0.75,
        label=obs_label,
        color=_POPULATION_PALETTE[0],
    )
    axes[0].bar(
        centres_h + w / 2,
        s_hist,
        width=w * 0.9,
        alpha=0.75,
        label=model_label,
        color=_POPULATION_PALETTE[1],
    )
    axes[0].set_xlabel("Work start (hour)")
    axes[0].set_ylabel("Mass")
    axes[0].set_title("Start-time distributions")
    axes[0].legend()

    labels_h = [f"{c * 24:.1f}h" for c in bin_centres]
    sns.heatmap(
        T,
        xticklabels=labels_h,
        yticklabels=labels_h,
        ax=axes[1],
        cmap="Blues",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        cbar=False,
    )
    axes[1].set_xlabel(f"{model_label} bin")
    axes[1].set_ylabel(f"{obs_label} bin")
    axes[1].set_title("Optimal transport plan  T")
    axes[1].tick_params(axis="x", rotation=30, labelsize=8)
    axes[1].tick_params(axis="y", rotation=0, labelsize=8)

    fig.suptitle(f"Earth Mover's Distance (work0 start) = {emd_val:.4f}", fontsize=11)
    fig.tight_layout()
    _save(fig, name, output_dir)


# ---------------------------------------------------------------------------
# Figure 10: Attribute-split bar chart
# ---------------------------------------------------------------------------


def splits(
    populations: dict[str, pd.DataFrame],
    name: str = "fig10-splits.svg",
    output_dir: Path = None,
):
    print("Fig 10: Attribute splits")
    pop_labels = list(populations.keys())
    obs_label, model_label = pop_labels[0], pop_labels[1]
    observed_df = populations[obs_label]
    synthetic_df = populations[model_label]

    n_obs = observed_df.pid.nunique()
    n_syn = synthetic_df.pid.nunique()

    n_ft = int(round(n_obs * 0.85))
    target_attrs = pd.DataFrame(
        {
            "pid": observed_df.pid.unique(),
            "work_status": (["full_time"] * n_ft) + (["student"] * (n_obs - n_ft)),
        }
    )
    synthetic_attrs = pd.DataFrame(
        {
            "pid": synthetic_df.pid.unique(),
            "work_status": ["full_time"] * n_syn,
        }
    )

    result = compare_splits(
        observed=observed_df,
        synthetic_schedules={model_label: synthetic_df},
        synthetic_attributes={model_label: synthetic_attrs},
        target_attributes=target_attrs,
        split_on=["work_status"],
        report_stats=False,
    )

    ld = result.label_domain_distances
    if ld is None:
        print("  label_domain_distances not available — skipping fig10")
        return

    ld_reset = ld.reset_index()

    if "label" in ld_reset.columns and "cat" in ld_reset.columns:
        ws = ld_reset[ld_reset["label"] == "work_status"].copy()
    else:
        ws = ld_reset.copy()

    if ws.empty:
        print("  no work_status rows found — skipping fig10")
        return

    if "domain" not in ws.columns:
        print("  unexpected index structure — skipping fig10")
        return

    cat_col = "cat" if "cat" in ws.columns else ws.columns[1]
    categories = ws[cat_col].unique()
    domains = ws["domain"].unique()
    x = np.arange(len(domains))
    width = 0.35
    n_cats = len(categories)

    fig, ax = _subplots(figsize=(9, 4))
    for i, cat in enumerate(sorted(categories)):
        subset = ws[ws[cat_col] == cat].set_index("domain")
        vals = [
            subset.loc[d, model_label] if d in subset.index else 0.0 for d in domains
        ]
        offset = (i - (n_cats - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width=width * 0.9,
            label=cat,
            color=_POPULATION_PALETTE[i % len(_POPULATION_PALETTE)],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=15)
    ax.set_ylabel("Distance from observed")
    ax.set_title(f"{model_label} — distances by work_status")
    ax.legend(title="work_status", fontsize=9)
    fig.tight_layout()
    _save(fig, name, output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=os.environ.get("OUTPUT_DIR", "assets/img/acteval"),
        help="Directory to write SVG files (default: assets/img/acteval)",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 450, "duration": 450},
            {"pid": 0, "act": "work", "start": 450, "end": 810, "duration": 360},
            {"pid": 0, "act": "eat_out", "start": 810, "end": 870, "duration": 60},
            {"pid": 0, "act": "work", "start": 870, "end": 1080, "duration": 210},
            {"pid": 0, "act": "home", "start": 1080, "end": 1440, "duration": 360},
        ]
    )

    gantt(
        populations={"Sample": df},
        output_dir=output_dir,
        name="fig1-sample-gantt.svg",
        act_colors=ACT_COLORS,
        acts=ACTS,
    )

    A = education_leaning(1000)
    B = leisure_dominant(1000)
    C = urban_workers(1000)

    populations = {
        "Population A": A,
        # "Population B": B,
        # "Population C": C,
    }

    gantt(
        populations=populations,
        output_dir=output_dir,
        name="fig2-samples-gantt.svg",
        act_colors=ACT_COLORS,
        acts=ACTS,
    )

    pop = Population(urban_workers(1000))
    print(pop.act_count_matrix[-3:])
    print(pop.int_to_act)

    print(sequence_lengths_feature(pop).aggregate())

    sequence_lengths({"A": A}, output_dir=output_dir, name="fig2-sequence-lengths.svg")

    participation(
        populations=populations,
        acts=ACTS,
        name="fig4-participation.svg",
        output_dir=output_dir,
    )

    sequence_lengths(
        {"A": A, "B": B}, output_dir=output_dir, name="fig3-sequence-lengths.svg"
    )

    participation(
        populations=populations,
        acts=ACTS,
        name="fig4-participation.svg",
        output_dir=output_dir,
    )

    print("\nRunning compare()...")

    result = compare(
        observed=A,
        synthetic=populations,
        report_stats=False,
    )
    print(result)

    print(f"\nBest model: {result.best_model}")

    print("\nGenerating plots...")
    timeuse(
        populations=populations,
        acts=ACTS,
        act_colors=ACT_COLORS,
        name="fig3-timeuse.svg",
        output_dir=output_dir,
    )
    participation(
        populations=populations,
        acts=ACTS,
        name="fig4-participation.svg",
        output_dir=output_dir,
    )
    bigrams(
        populations=populations,
        acts=ACTS,
        name="fig5-bigrams.svg",
        output_dir=output_dir,
    )
    starttimes(populations=populations, output_dir=output_dir)
    joint(populations=populations, output_dir=output_dir)
    emd(populations=populations, output_dir=output_dir)
    heatmap(result, name="fig8-heatmap.svg", output_dir=output_dir)
    groups(result, name="fig9-groups.svg", output_dir=output_dir)
    splits(populations=populations, output_dir=output_dir)

    print(f"\nDone. All plots written to {output_dir}/")


if __name__ == "__main__":
    main()
