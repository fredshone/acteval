"""Generate all plots for the acteval distances blog post.

Usage:
    uv run python scripts/generate_blog_plots.py

Writes 10 SVG files to OUTPUT_DIR (default: assets/img/acteval/).
Override with:
    OUTPUT_DIR=path/to/dir uv run python scripts/generate_blog_plots.py
"""

import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import seaborn as sns

from acteval import Population, compare, compare_splits
from acteval.features.participation import participation_rates_by_act_per_pid
from acteval.features.times import start_times_by_act_plan_enum_per_pid
from acteval.features.transitions import ngrams_per_pid

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "assets/img/acteval"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

def _clip_normal(rng, mean, std, lo, hi):
    return int(round(np.clip(rng.normal(mean, std), lo, hi)))


def generate_urban_workers(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Observed population: office workers, flexible workers, students."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n):
        profile = rng.choice(["office", "flexible", "student"], p=[0.60, 0.25, 0.15])
        if profile == "office":
            work_start  = _clip_normal(rng, 450, 30, 390, 510)
            lunch_start = _clip_normal(rng, 810, 15, max(work_start + 60, 750), 870)
            lunch_end   = lunch_start + 60
            work_end    = _clip_normal(rng, 1080, 20, max(lunch_end + 60, 1020), 1140)
            episodes = [
                ("home",    0,           work_start),
                ("work",    work_start,  lunch_start),
                ("eat_out", lunch_start, lunch_end),
                ("work",    lunch_end,   work_end),
                ("home",    work_end,    1440),
            ]
        elif profile == "flexible":
            work_start = _clip_normal(rng, 480, 45, 360, 600)
            work_end   = _clip_normal(rng, 960, 30, max(work_start + 60, 900), 1080)
            shop_end   = min(work_end + 60, 1440)
            episodes = [
                ("home", 0,          work_start),
                ("work", work_start, work_end),
                ("shop", work_end,   shop_end),
                ("home", shop_end,   1440),
            ]
        else:  # student
            edu_start = _clip_normal(rng, 540, 30, 480, 600)
            edu_end   = _clip_normal(rng, 840, 20, max(edu_start + 60, 780), 900)
            lei_end   = min(edu_end + 120, 1440)
            episodes = [
                ("home",      0,         edu_start),
                ("education", edu_start, edu_end),
                ("leisure",   edu_end,   lei_end),
                ("home",      lei_end,   1440),
            ]
        for act, start, end in episodes:
            rows.append({"pid": pid, "act": act,
                         "start": start, "end": end, "duration": end - start})
    return pd.DataFrame(rows)


def generate_suburban_workers(n: int = 500, seed: int = 7) -> pd.DataFrame:
    """Model A: work starts ~45 min later, reduced eat_out, no education."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n):
        profile = rng.choice(["office", "flexible", "part_time"], p=[0.60, 0.25, 0.15])
        if profile == "office":
            work_start = _clip_normal(rng, 495, 50, 390, 600)
            work_end   = _clip_normal(rng, 1080, 20, 1020, 1140)
            if rng.random() < 0.60:
                lunch_start = _clip_normal(rng, 840, 15, max(work_start + 60, 780), 900)
                lunch_end   = lunch_start + 60
                episodes = [
                    ("home",    0,           work_start),
                    ("work",    work_start,  lunch_start),
                    ("eat_out", lunch_start, lunch_end),
                    ("work",    lunch_end,   work_end),
                    ("home",    work_end,    1440),
                ]
            else:
                episodes = [
                    ("home", 0,          work_start),
                    ("work", work_start, work_end),
                    ("home", work_end,   1440),
                ]
        elif profile == "flexible":
            work_start = _clip_normal(rng, 495, 50, 360, 630)
            work_end   = _clip_normal(rng, 975, 30, max(work_start + 60, 900), 1095)
            shop_end   = min(work_end + 60, 1440)
            episodes = [
                ("home", 0,          work_start),
                ("work", work_start, work_end),
                ("shop", work_end,   shop_end),
                ("home", shop_end,   1440),
            ]
        else:  # part_time
            work_start = _clip_normal(rng, 540, 30, 480, 600)
            work_end   = _clip_normal(rng, 900, 30, max(work_start + 60, 840), 960)
            episodes = [
                ("home", 0,          work_start),
                ("work", work_start, work_end),
                ("home", work_end,   1440),
            ]
        for act, start, end in episodes:
            rows.append({"pid": pid, "act": act,
                         "start": start, "end": end, "duration": end - start})
    return pd.DataFrame(rows)


def generate_lifestyle(n: int = 500, seed: int = 13) -> pd.DataFrame:
    """Model B: leisure-dominant, late/spread work, inverted activity mix."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n):
        profile = rng.choice(["leisure_shopper", "late_worker", "home_heavy"],
                             p=[0.40, 0.35, 0.25])
        if profile == "leisure_shopper":
            lei_start = _clip_normal(rng, 420, 60, 300, 540)
            lei_end   = _clip_normal(rng, 780, 60, max(lei_start + 120, 600), 900)
            shop_end  = min(lei_end + 90, 1440)
            episodes = [
                ("home",    0,         lei_start),
                ("leisure", lei_start, lei_end),
                ("shop",    lei_end,   shop_end),
                ("home",    shop_end,  1440),
            ]
        elif profile == "late_worker":
            work_start = _clip_normal(rng, 600, 90, 360, 900)
            work_end   = _clip_normal(rng, 1080, 60, max(work_start + 120, 960), 1380)
            episodes = [
                ("home", 0,          work_start),
                ("work", work_start, work_end),
                ("home", work_end,   1440),
            ]
        else:  # home_heavy
            lei_start = _clip_normal(rng, 600, 60, 480, 720)
            lei_end   = _clip_normal(rng, 1200, 60, max(lei_start + 120, 1080), 1380)
            episodes = [
                ("home",    0,         lei_start),
                ("leisure", lei_start, lei_end),
                ("home",    lei_end,   1440),
            ]
        for act, start, end in episodes:
            rows.append({"pid": pid, "act": act,
                         "start": start, "end": end, "duration": end - start})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _save(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Gantt chart
# ---------------------------------------------------------------------------

def fig1_gantt(urban_df, suburban_df):
    print("Fig 1: Gantt chart")
    # Pick 8 pids that cover all profiles
    sample_pids = list(range(8))
    pops = {"Urban (observed)": urban_df, "Suburban (model A)": suburban_df}

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, (label, df) in zip(axes, pops.items()):
        for y, pid in enumerate(sample_pids):
            person = df[df.pid == pid]
            for _, row in person.iterrows():
                ax.broken_barh(
                    [(row.start, row.duration)], (y - 0.4, 0.8),
                    facecolors=ACT_COLORS.get(row.act, "#aaa"),
                    edgecolor="white", linewidth=0.5,
                )
        ax.set_xlim(0, 1440)
        ax.set_yticks(range(len(sample_pids)))
        ax.set_yticklabels([f"person {p}" for p in sample_pids], fontsize=8)
        ax.set_title(label, fontsize=10)

    axes[-1].set_xticks([0, 360, 720, 1080, 1440])
    axes[-1].set_xticklabels(["midnight", "6 am", "noon", "6 pm", "midnight"])

    legend_handles = [
        mpatches.Patch(color=c, label=a) for a, c in ACT_COLORS.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=6,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, "fig1-gantt.svg")


# ---------------------------------------------------------------------------
# Figure 2: Stacked area — population-average time use
# ---------------------------------------------------------------------------

def _time_use(df: pd.DataFrame, step: int = 10) -> pd.DataFrame:
    minutes = np.arange(0, 1440, step)
    pids = df.pid.nunique()
    records = []
    for t in minutes:
        active = df[(df.start <= t) & (df.end > t)]
        counts = active.groupby("act").pid.nunique()
        records.append(counts / pids)
    return pd.DataFrame(records, index=minutes).fillna(0)


def fig2_timeuse(urban_df, suburban_df, lifestyle_df):
    print("Fig 2: Stacked area time use")
    datasets = [
        ("Urban (observed)", urban_df),
        ("Suburban (model A)", suburban_df),
        ("Lifestyle (model B)", lifestyle_df),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (label, df) in zip(axes, datasets):
        tu = _time_use(df)
        present_acts = [a for a in ACTS if a in tu.columns]
        ax.stackplot(
            tu.index,
            [tu[a] for a in present_acts],
            colors=[ACT_COLORS[a] for a in present_acts],
            labels=present_acts,
            alpha=0.85,
        )
        ax.set_xlim(0, 1440)
        ax.set_xticks([0, 360, 720, 1080, 1440])
        ax.set_xticklabels(["0", "6", "12", "18", "24"])
        ax.set_xlabel("Hour of day")
        ax.set_title(label, fontsize=10)
    axes[0].set_ylabel("Fraction of population")

    legend_handles = [
        mpatches.Patch(color=ACT_COLORS[a], label=a) for a in ACTS
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=6,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    _save(fig, "fig2-timeuse.svg")


# ---------------------------------------------------------------------------
# Figure 3: Participation rate bar chart
# ---------------------------------------------------------------------------

def _participation_rates(df: pd.DataFrame) -> pd.Series:
    pop = Population(df)
    rates = {}
    for act in ACTS:
        if act in pop.unique_acts:
            idx = list(pop.unique_acts).index(act)
            rates[act] = (pop.count_matrix[:, idx] > 0).mean()
        else:
            rates[act] = 0.0
    return pd.Series(rates)


def fig3_participation(urban_df, suburban_df, lifestyle_df):
    print("Fig 3: Participation rates")
    rate_df = pd.DataFrame({
        "Urban (observed)":    _participation_rates(urban_df),
        "Suburban (model A)":  _participation_rates(suburban_df),
        "Lifestyle (model B)": _participation_rates(lifestyle_df),
    })
    x = np.arange(len(ACTS))
    width = 0.25
    colors = ["#4e9af1", "#e05c5c", "#6dbf67"]

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (label, col) in enumerate(rate_df.items()):
        ax.bar(x + i * width, col.values, width, label=label,
               color=colors[i], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(ACTS)
    ax.set_ylabel("Participation rate")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "fig3-participation.svg")


# ---------------------------------------------------------------------------
# Figure 4: Bigram transition heatmap
# ---------------------------------------------------------------------------

def _bigram_matrix(df: pd.DataFrame) -> np.ndarray:
    pop = Population(df)
    bigrams = ngrams_per_pid(pop, n=2).aggregate()
    act_idx = {a: i for i, a in enumerate(ACTS)}
    mat = np.zeros((len(ACTS), len(ACTS)))
    for key, (_, counts) in bigrams.items():
        parts = key.split(">")
        if len(parts) == 2 and parts[0] in act_idx and parts[1] in act_idx:
            mat[act_idx[parts[0]], act_idx[parts[1]]] = counts.sum() / pop.n
    return mat


def fig4_bigrams(urban_df, suburban_df):
    print("Fig 4: Bigram heatmaps")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (label, df) in zip(axes, [("Urban (observed)", urban_df),
                                       ("Suburban (model A)", suburban_df)]):
        mat = _bigram_matrix(df)
        sns.heatmap(
            mat, annot=True, fmt=".2f",
            xticklabels=ACTS, yticklabels=ACTS,
            cmap="Blues", vmin=0, vmax=1,
            ax=ax, cbar=False, annot_kws={"size": 7},
        )
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("to", fontsize=9)
        ax.set_ylabel("from", fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    _save(fig, "fig4-bigrams.svg")


# ---------------------------------------------------------------------------
# Figure 5: work0 start-time histogram
# ---------------------------------------------------------------------------

def fig5_starttimes(urban_df, suburban_df):
    print("Fig 5: Work start-time histograms")
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, df, color in [
        ("Urban (observed)", urban_df, "#4e9af1"),
        ("Suburban (model A)", suburban_df, "#e05c5c"),
    ]:
        # first work episode per person, in hours
        starts = (
            df[df.act == "work"]
            .groupby("pid")
            .start.min()
            .values / 60
        )
        ax.hist(starts, bins=30, alpha=0.55, color=color, label=label, density=True)
    ax.set_xlabel("Work start (hour of day)")
    ax.set_ylabel("Density")
    ax.set_xticks(range(4, 16))
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig5-starttimes.svg")


# ---------------------------------------------------------------------------
# Figure 6: 2D scatter (work start, total work duration)
# ---------------------------------------------------------------------------

def fig6_joint(urban_df, suburban_df):
    print("Fig 6: (work start, duration) scatter")
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, df, color in [
        ("Urban (observed)", urban_df, "#4e9af1"),
        ("Suburban (model A)", suburban_df, "#e05c5c"),
    ]:
        work = df[df.act == "work"].groupby("pid").agg(
            start=("start", "min"),
            duration=("duration", "sum"),
        )
        ax.scatter(work.start / 60, work.duration / 60,
                   alpha=0.25, color=color, label=label, s=14)
    ax.set_xlabel("Work start (h)")
    ax.set_ylabel("Total work duration (h)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig6-joint.svg")


# ---------------------------------------------------------------------------
# Figure 7: EMD transport-plan diagram
# ---------------------------------------------------------------------------

def fig7_emd(urban_df, suburban_df):
    print("Fig 7: EMD transport plan")
    # Get aggregated work0 start times from the library's feature functions
    # values are already divided by 1440 (factor=1440 inside PidFeatures)
    urban_agg    = start_times_by_act_plan_enum_per_pid(Population(urban_df)).aggregate()
    suburban_agg = start_times_by_act_plan_enum_per_pid(Population(suburban_df)).aggregate()

    u_vals, u_wts = urban_agg["work0"]   # values in [0, 1]
    s_vals, s_wts = suburban_agg["work0"]

    # Build coarse 30-min bins (30/1440 ≈ 0.0208 in normalised units)
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

    # Optimal transport plan
    M = ot.dist(bin_centres[:, None], bin_centres[:, None], metric="euclidean")
    T = ot.emd(u_hist, s_hist, M)
    emd_val = (T * M).sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: overlaid histograms
    w = (bin_edges[1] - bin_edges[0]) * 1440 / 60 * 0.4  # bar width in hours
    centres_h = bin_centres * 24  # convert to hours
    axes[0].bar(centres_h - w / 2, u_hist, width=w * 0.9, alpha=0.75,
                label="Urban (obs)", color="#4e9af1")
    axes[0].bar(centres_h + w / 2, s_hist, width=w * 0.9, alpha=0.75,
                label="Suburban (A)", color="#e05c5c")
    axes[0].set_xlabel("Work start (hour)")
    axes[0].set_ylabel("Mass")
    axes[0].set_title("Start-time distributions")
    axes[0].legend()

    # Right: transport plan as heatmap
    labels_h = [f"{c * 24:.1f}h" for c in bin_centres]
    sns.heatmap(T, xticklabels=labels_h, yticklabels=labels_h,
                ax=axes[1], cmap="Blues", annot=True, fmt=".2f",
                annot_kws={"size": 8}, cbar=False)
    axes[1].set_xlabel("Suburban bin")
    axes[1].set_ylabel("Urban bin")
    axes[1].set_title("Optimal transport plan  T")
    axes[1].tick_params(axis="x", rotation=30, labelsize=8)
    axes[1].tick_params(axis="y", rotation=0, labelsize=8)

    fig.suptitle(f"Earth Mover's Distance (work0 start) = {emd_val:.4f}", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig7-emd.svg")


# ---------------------------------------------------------------------------
# Figure 8: Domain heatmap
# ---------------------------------------------------------------------------

def fig8_heatmap(result):
    print("Fig 8: Domain heatmap")
    summary = result.summary()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        summary,
        annot=True, fmt=".2f",
        cmap="YlOrRd", vmin=0, vmax=0.5,
        ax=ax,
        cbar_kws={"label": "distance (0 = identical)"},
    )
    ax.set_title("Domain-level distances from observed population")
    ax.set_xlabel("")
    fig.tight_layout()
    _save(fig, "fig8-heatmap.svg")


# ---------------------------------------------------------------------------
# Figure 9: Feature-group horizontal bar chart
# ---------------------------------------------------------------------------

def fig9_groups(result):
    print("Fig 9: Feature-group bar chart")
    gd = result.group_distances[result.model_names]
    fig, ax = plt.subplots(figsize=(7, max(5, len(gd) * 0.4)))
    y = np.arange(len(gd))
    colors = ["#4e9af1", "#e05c5c"]
    for i, model in enumerate(result.model_names):
        offset = (i - (len(result.model_names) - 1) / 2) * 0.3
        ax.barh(y + offset, gd[model], height=0.25, label=model, color=colors[i])
    labels = [f"{d} / {f}" for d, f in gd.index]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, None)
    ax.set_xlabel("Distance")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "fig9-groups.svg")


# ---------------------------------------------------------------------------
# Figure 10: Attribute-split bar chart
# ---------------------------------------------------------------------------

def fig10_splits(urban_df, suburban_df):
    print("Fig 10: Attribute splits")
    n_urban    = urban_df.pid.nunique()
    n_suburban = suburban_df.pid.nunique()

    # Assign work_status labels matching the profile proportions used in generators
    # urban:    60% office + 25% flexible = 85% full_time, 15% student
    # suburban: 60% office + 25% flexible + 15% part_time = all full_time
    n_ft_urban = int(round(n_urban * 0.85))
    target_attrs = pd.DataFrame({
        "pid":         urban_df.pid.unique(),
        "work_status": (["full_time"] * n_ft_urban) + (["student"] * (n_urban - n_ft_urban)),
    })
    suburban_attrs = pd.DataFrame({
        "pid":         suburban_df.pid.unique(),
        "work_status": ["full_time"] * n_suburban,
    })

    result = compare_splits(
        observed=urban_df,
        synthetic_schedules={"suburban": suburban_df},
        synthetic_attributes={"suburban": suburban_attrs},
        target_attributes=target_attrs,
        split_on=["work_status"],
        report_stats=False,
    )

    ld = result.label_domain_distances
    if ld is None:
        print("  label_domain_distances not available — skipping fig10")
        return

    # Flatten MultiIndex and filter to work_status split
    ld_reset = ld.reset_index()
    # Find the level names
    level_names = list(ld_reset.columns[:ld_reset.columns.get_loc("suburban")])

    # Try to find domain, label, cat columns
    if "label" in ld_reset.columns and "cat" in ld_reset.columns:
        ws = ld_reset[ld_reset["label"] == "work_status"].copy()
    else:
        # fall back: use first split level
        ws = ld_reset.copy()

    if ws.empty:
        print("  no work_status rows found — skipping fig10")
        return

    if "domain" not in ws.columns:
        print("  unexpected index structure — skipping fig10")
        return

    categories = ws["cat"].unique() if "cat" in ws.columns else ws.iloc[:, 1].unique()
    cat_col = "cat" if "cat" in ws.columns else ws.columns[1]

    domains = ws["domain"].unique()
    x = np.arange(len(domains))
    width = 0.35
    n_cats = len(categories)
    palette = ["#4e9af1", "#e05c5c", "#6dbf67", "#e8a838"]

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, cat in enumerate(sorted(categories)):
        subset = ws[ws[cat_col] == cat].set_index("domain")
        vals = [subset.loc[d, "suburban"] if d in subset.index else 0.0 for d in domains]
        offset = (i - (n_cats - 1) / 2) * width
        ax.bar(x + offset, vals, width=width * 0.9,
               label=cat, color=palette[i % len(palette)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=15)
    ax.set_ylabel("Distance from observed")
    ax.set_title("Suburban model — distances by work_status")
    ax.legend(title="work_status", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig10-splits.svg")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating populations...")
    urban_df    = generate_urban_workers(500, seed=42)
    suburban_df = generate_suburban_workers(500, seed=7)
    lifestyle_df = generate_lifestyle(500, seed=13)

    print(f"Urban:    {urban_df.pid.nunique()} people, "
          f"activities: {sorted(urban_df.act.unique())}")
    print(f"Suburban: {suburban_df.pid.nunique()} people, "
          f"activities: {sorted(suburban_df.act.unique())}")
    print(f"Lifestyle:{lifestyle_df.pid.nunique()} people, "
          f"activities: {sorted(lifestyle_df.act.unique())}")

    print("\nRunning compare()...")
    result = compare(
        urban_df,
        {"suburban": suburban_df, "lifestyle": lifestyle_df},
        report_stats=False,
    )
    print(result)

    print(f"\nBest model: {result.best_model}")

    print("\nGenerating plots...")
    fig1_gantt(urban_df, suburban_df)
    fig2_timeuse(urban_df, suburban_df, lifestyle_df)
    fig3_participation(urban_df, suburban_df, lifestyle_df)
    fig4_bigrams(urban_df, suburban_df)
    fig5_starttimes(urban_df, suburban_df)
    fig6_joint(urban_df, suburban_df)
    fig7_emd(urban_df, suburban_df)
    fig8_heatmap(result)
    fig9_groups(result)
    fig10_splits(urban_df, suburban_df)

    print(f"\nDone. All plots written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
