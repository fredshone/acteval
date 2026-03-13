import numpy as np
from numpy import array, ndarray

from acteval.density.features.pid_features import PidFeatures
from acteval.density.features.utils import (
    _count_matrix,
    weighted_features,
)
from acteval.population import Population


def participation_prob_by_act(
    population: Population,
) -> dict[str, tuple[ndarray, ndarray]]:
    """Calculate the participations by activity for a given population.

    Args:
        population (Population): The population data.

    Returns:
        dict[str, tuple[array, array]]: A dictionary containing the participation for each activity.
    """
    matrix = population.count_matrix
    participated = (matrix > 0).sum(axis=0)
    n = population.n
    return {
        act: (array([0, 1]), array([n - participated[j], participated[j]]))
        for j, act in enumerate(population.unique_acts)
    }


def participation_rates(
    population: Population,
) -> dict[str, tuple[ndarray, ndarray]]:
    return weighted_features({"all": population.pid_counts})


def participation_rates_by_act(
    population: Population,
) -> dict[str, tuple[ndarray, ndarray]]:
    matrix = population.count_matrix
    return weighted_features(
        {act: matrix[:, j] for j, act in enumerate(population.unique_acts)}
    )


def participation_rates_by_seq_act(
    population: Population,
) -> dict[str, tuple[ndarray, ndarray]]:
    matrix, _, unique_keys = _count_matrix(population.pids, population.seq_key)
    return weighted_features({k: matrix[:, j] for j, k in enumerate(unique_keys)})


def participation_rates_by_act_enum(
    population: Population,
) -> dict[str, tuple[ndarray, ndarray]]:
    matrix, _, unique_keys = _count_matrix(population.pids, population.act_enum_key)
    return weighted_features({k: matrix[:, j] for j, k in enumerate(unique_keys)})


def calc_pair_prob(act_counts, pair):
    a, b = pair
    if a == b:
        return (act_counts[a] > 1).sum()
    return ((act_counts[a] > 0) & (act_counts[b] > 0)).sum()


def calc_pair_rate(act_counts, pair):
    a, b = pair
    if a == b:
        return ((act_counts[a] / 2).astype(int)).value_counts().to_dict()
    return ((act_counts[[a, b]].min(axis=1) / 2).astype(int)).value_counts().to_dict()


def combinations_with_replacement(
    targets: list, length: int, prev_array=[]
) -> list[list]:
    """Returns all possible combinations of elements in the input array with replacement,
    where each combination has a length of tuple_length.

    Args:
        targets (list): The input array to generate combinations from.
        length (int): The length of each combination.
        prev_array (list, optional): The previous array generated in the recursion. Defaults to [].

    Returns:
        list: A list of all possible combinations of elements in the input array with replacement.
    """
    if len(prev_array) == length:
        return [prev_array]
    combs = []
    for i, val in enumerate(targets):
        prev_array_extended = prev_array.copy()
        prev_array_extended.append(val)
        combs += combinations_with_replacement(targets[i:], length, prev_array_extended)
    return combs


def joint_participation_prob(
    population: Population,
) -> dict[str, tuple[ndarray, ndarray]]:
    """Calculate the participation prob for all pairs of activities in the given population.

    Args:
        population (Population): A Population containing the population data.

    Returns:
        dict: A dictionary containing the participation probability for all pairs of activities.
    """
    matrix = population.count_matrix
    unique_acts = population.unique_acts
    act_list = list(unique_acts)
    act_idx = {a: i for i, a in enumerate(act_list)}
    n = matrix.shape[0]
    pairs = combinations_with_replacement(act_list, 2)
    metric = {}
    for pair in pairs:
        ai, bi = act_idx[pair[0]], act_idx[pair[1]]
        if pair[0] == pair[1]:
            p = int((matrix[:, ai] > 1).sum())
        else:
            p = int(((matrix[:, ai] > 0) & (matrix[:, bi] > 0)).sum())
        metric["+".join(pair)] = (array([0, 1]), array([n - p, p]))
    return metric


def joint_participation_rate(
    population: Population,
) -> dict[str, tuple[ndarray, ndarray]]:
    """Calculate the participation rate for all pairs of activities in the given population.

    Args:
        population (Population): A Population containing the population data.

    Returns:
        dict: A dictionary containing the participation rate for all pairs of activities.
    """
    matrix = population.count_matrix
    unique_acts = population.unique_acts
    act_list = list(unique_acts)
    act_idx = {a: i for i, a in enumerate(act_list)}
    pairs = combinations_with_replacement(act_list, 2)
    metric = {}
    for pair in pairs:
        ai, bi = act_idx[pair[0]], act_idx[pair[1]]
        if pair[0] == pair[1]:
            vals = matrix[:, ai] // 2
        else:
            vals = np.minimum(matrix[:, ai], matrix[:, bi]) // 2
        keys, counts = np.unique(vals, return_counts=True)
        metric["+".join(pair)] = (keys, counts)
    return metric


# ---------------------------------------------------------------------------
# Per-pid variants — return PidFeatures for efficient subsetting
# ---------------------------------------------------------------------------


def participation_rates_by_act_per_pid(
    population: Population,
) -> PidFeatures:
    """Per-pid participation rates by activity.

    Each key maps to ``(count_per_person, pids)`` where *pids* is shared
    across all keys (the dense pid range).
    """
    matrix = population.count_matrix
    pids = np.arange(population.n)
    data = {act: (matrix[:, j], pids) for j, act in enumerate(population.unique_acts)}
    return PidFeatures(data=data, bin_size=None, factor=1)


def joint_participation_rate_per_pid(
    population: Population,
) -> PidFeatures:
    """Per-pid joint participation rates for all activity pairs."""
    matrix = population.count_matrix
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
