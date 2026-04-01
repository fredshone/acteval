from itertools import combinations_with_replacement

import numpy as np

from acteval.features._pid_features import PidFeatures
from acteval.features._utils import _count_matrix
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
    return PidFeatures(data=data, bin_size=None, factor=1)


def participation_rates_by_act_enum(
    population: Population,
) -> PidFeatures:
    """Per-pid participation rates keyed by n-th occurrence of each activity (e.g. 'home0', 'home1').

    Each key maps to ``(count_per_person, pids)`` where count is 0 or 1.
    """
    matrix, _, unique_keys = _count_matrix(population.pids, population.act_enum_key)
    pids = np.arange(population.n)
    data = {k: (matrix[:, j], pids) for j, k in enumerate(unique_keys)}
    return PidFeatures(data=data, bin_size=None, factor=1)


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
    return PidFeatures(data=data, bin_size=None, factor=1)


def joint_participation_rate(
    population: Population,
) -> PidFeatures:
    """Per-pid joint participation rates for all activity pairs."""
    matrix = population.act_count_matrix
    unique_acts = population.unique_acts
    act_list = list(unique_acts)
    act_idx = {a: i for i, a in enumerate(act_list)}
    pids = np.arange(population.n)
    pairs = combinations_with_replacement(act_list, 2)
    data = {}
    for pair in pairs:
        ai, bi = act_idx[pair[0]], act_idx[pair[1]]
        if pair[0] == pair[1]:
            vals = matrix[:, ai] // 2
        else:
            vals = np.minimum(matrix[:, ai], matrix[:, bi]) // 2
        data["+".join(pair)] = (vals, pids)
    return PidFeatures(data=data, bin_size=None, factor=1)


def sequence_lengths(
    population: Population,
) -> PidFeatures:
    pids = np.arange(population.n)
    return PidFeatures(
        data={"sequence lengths": (population.pid_counts, pids)},
        bin_size=None,
        factor=1,
    )
