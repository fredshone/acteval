from itertools import combinations_with_replacement

import numpy as np

from acteval.features._pid_features import PidFeatures
from acteval.features._utils import _ambiguous_label, _count_matrix
from acteval.population import Population


def participation_rates_by_seq_act(
    population: Population,
) -> PidFeatures:
    """Per-pid participation rates keyed by sequence position (e.g. '0home', '1work').

    Each key maps to ``(count_per_person, pids)`` where count is 0 or 1.
    """
    matrix, _, unique_keys = _count_matrix(population.pids, population.seq_key)
    pids = np.arange(population.n)
    data = {k: (matrix[:, j], pids) for j, k in enumerate(unique_keys)}
    return PidFeatures(data=data, bin_size=None, factor=1, zero_is_missing=True)


def participation_rates_by_act_enum(
    population: Population,
) -> PidFeatures:
    """Per-pid participation rates keyed by n-th occurrence of each activity (e.g. 'home0', 'home1').

    Each key maps to ``(count_per_person, pids)`` where count is 0 or 1.
    """
    matrix, _, unique_keys = _count_matrix(population.pids, population.act_enum_key)
    pids = np.arange(population.n)
    data = {k: (matrix[:, j], pids) for j, k in enumerate(unique_keys)}
    return PidFeatures(data=data, bin_size=None, factor=1, zero_is_missing=True)


def participation_rates_by_act(
    population: Population,
) -> PidFeatures:
    """Per-pid participation rates by activity.

    Each key maps to ``(count_per_person, pids)`` where *pids* is shared
    across all keys (the dense pid range).
    """
    matrix = population.act_count_matrix
    pids = np.arange(population.n)
    data = {act: (matrix[:, j], pids) for j, act in enumerate(population.unique_acts)}
    return PidFeatures(data=data, bin_size=None, factor=1, zero_is_missing=True)


def joint_participation_rate(
    population: Population,
) -> PidFeatures:
    """Per-pid joint participation rates for all activity pairs.

    Counts the distinct pairs of episodes per person: ``n * (n - 1) / 2`` for
    an activity with itself (e.g. ``work+work``), ``n * m`` for two different
    activities (e.g. ``home+work``), where ``n`` and ``m`` are the person's
    episode counts.
    """
    matrix = population.act_count_matrix
    pids = np.arange(population.n)
    data = {}
    for ai, bi in combinations_with_replacement(range(population.n_act_types), 2):
        a, b = population.unique_acts[ai], population.unique_acts[bi]
        n = matrix[:, ai]
        if ai == bi:
            vals = n * (n - 1) // 2
        else:
            vals = n * matrix[:, bi]
        label = f"{a}+{b}"
        if label in data:
            raise _ambiguous_label(label)
        data[label] = (vals, pids)
    return PidFeatures(data=data, bin_size=None, factor=1, zero_is_missing=True)


def sequence_lengths(
    population: Population,
) -> PidFeatures:
    pids = np.arange(population.n)
    return PidFeatures(
        data={"sequence lengths": (population.pid_counts, pids)},
        bin_size=None,
        factor=1,
    )
