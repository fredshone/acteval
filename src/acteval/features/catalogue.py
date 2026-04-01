"""Feature catalogue: all available PidFeatures-based feature functions.

Use ``list_features()`` to get a DataFrame of available features with
descriptions and whether each is enabled in the default configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import pandas as pd

from acteval.features import participation, structural, times
from acteval.features.transitions import full_sequences, ngrams


@dataclass(frozen=True)
class FeatureEntry:
    """Metadata for a single feature function."""

    domain: str
    name: str
    config_key: str
    function: Callable
    description: str
    in_default_config: bool


CATALOGUE: list[FeatureEntry] = [
    # --- participations ---
    FeatureEntry(
        domain="participations",
        name="sequence lengths",
        config_key="lengths",
        function=participation.sequence_lengths,
        description="Distribution of number of episodes per person.",
        in_default_config=True,
    ),
    FeatureEntry(
        domain="participations",
        name="participation rate",
        config_key="rates",
        function=participation.participation_rates_by_act,
        description="How many times each person participates in each activity.",
        in_default_config=True,
    ),
    FeatureEntry(
        domain="participations",
        name="pair participation rate",
        config_key="pair_rates",
        function=participation.joint_participation_rate,
        description="Co-participation counts for all activity pairs.",
        in_default_config=True,
    ),
    FeatureEntry(
        domain="participations",
        name="seq participation rate",
        config_key="seq_rates",
        function=participation.participation_rates_by_seq_act,
        description="Participation rates keyed by sequence position (e.g. '0home', '1work').",
        in_default_config=False,
    ),
    FeatureEntry(
        domain="participations",
        name="enum participation rate",
        config_key="enum_rates",
        function=participation.participation_rates_by_act_enum,
        description="Participation rates keyed by n-th occurrence of each activity (e.g. 'home0', 'home1').",
        in_default_config=False,
    ),
    # --- timing ---
    FeatureEntry(
        domain="timing",
        name="start times",
        config_key="start_times",
        function=times.start_times_by_act_plan_enum,
        description="Start-time distribution per activity × occurrence index.",
        in_default_config=True,
    ),
    FeatureEntry(
        domain="timing",
        name="durations",
        config_key="durations",
        function=times.durations_by_act_plan_enum,
        description="Duration distribution per activity × occurrence index.",
        in_default_config=True,
    ),
    FeatureEntry(
        domain="timing",
        name="start-durations",
        config_key="start_durations",
        function=times.start_and_duration_by_act_bins,
        description="Joint (start, duration) 2-D distribution per activity.",
        in_default_config=True,
    ),
    FeatureEntry(
        domain="timing",
        name="joint-durations",
        config_key="joint_durations",
        function=times.joint_durations_by_act_bins,
        description="Joint (duration_i, duration_{i+1}) distribution for consecutive activity pairs.",
        in_default_config=True,
    ),
    FeatureEntry(
        domain="timing",
        name="start times by act",
        config_key="start_times_by_act",
        function=times.start_times_by_act,
        description="Start-time distribution per activity (no occurrence index).",
        in_default_config=False,
    ),
    FeatureEntry(
        domain="timing",
        name="end times by act",
        config_key="end_times_by_act",
        function=times.end_times_by_act,
        description="End-time distribution per activity (no occurrence index).",
        in_default_config=False,
    ),
    FeatureEntry(
        domain="timing",
        name="durations by act",
        config_key="durations_by_act",
        function=times.durations_by_act,
        description="Duration distribution per activity (no occurrence index).",
        in_default_config=False,
    ),
    FeatureEntry(
        domain="timing",
        name="time consistency",
        config_key="time_consistency",
        function=structural.time_consistency,
        description="Per-person flags: starts at 0, ends at 1440, total duration equals 1440.",
        in_default_config=False,
    ),
    # --- transitions ---
    FeatureEntry(
        domain="transitions",
        name="2-gram",
        config_key="2-gram",
        function=partial(ngrams, n=2),
        description="Consecutive activity pair (bigram) counts per person.",
        in_default_config=True,
    ),
    FeatureEntry(
        domain="transitions",
        name="3-gram",
        config_key="3-gram",
        function=partial(ngrams, n=3),
        description="Consecutive activity triple (trigram) counts per person.",
        in_default_config=True,
    ),
    FeatureEntry(
        domain="transitions",
        name="4-gram",
        config_key="4-gram",
        function=partial(ngrams, n=4),
        description="Consecutive activity quad (4-gram) counts per person.",
        in_default_config=True,
    ),
    FeatureEntry(
        domain="transitions",
        name="full sequences",
        config_key="full_sequences",
        function=full_sequences,
        description="Per-person indicator for each unique full abbreviated tour string (e.g. 'h>w>h').",
        in_default_config=False,
    ),
]


def list_features() -> pd.DataFrame:
    """Return a DataFrame listing all available feature functions.

    Returns:
        DataFrame with columns: domain, name, config_key, description, in_default_config.
    """
    return pd.DataFrame(
        [
            {
                "domain": e.domain,
                "name": e.name,
                "config_key": e.config_key,
                "description": e.description,
                "in_default_config": e.in_default_config,
            }
            for e in CATALOGUE
        ]
    )
