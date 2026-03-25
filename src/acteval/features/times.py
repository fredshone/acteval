import numpy as np
from numpy import ndarray

from acteval.features._pid_features import PidFeatures
from acteval.features._utils import (
    _collect_by_group,
    _collect_by_group_with_pids,
    weighted_features,
)
from acteval.population import Population


def start_times_by_act(
    population: Population, bin_size: int = 15, factor: int = 1440
) -> dict[str, tuple[ndarray, ndarray]]:
    features = _collect_by_group(population.acts, population.starts)
    return weighted_features(features, bin_size=bin_size, factor=factor)


def end_times_by_act(
    population: Population, bin_size: int = 15, factor: int = 1440
) -> dict[str, tuple[ndarray, ndarray]]:
    features = _collect_by_group(population.acts, population.ends)
    return weighted_features(features, bin_size=bin_size, factor=factor)


def durations_by_act(
    population: Population, bin_size: int = 15, factor: int = 1440
) -> dict[str, tuple[ndarray, ndarray]]:
    features = _collect_by_group(population.acts, population.durations)
    return weighted_features(features, bin_size=bin_size, factor=factor)


def start_times_by_act_plan_seq(
    population: Population,
) -> dict[str, tuple[ndarray, ndarray]]:
    features = _collect_by_group(population.seq_key, population.starts)
    return weighted_features(features, factor=1440)


def end_times_by_act_plan_seq(
    population: Population,
) -> dict[str, tuple[ndarray, ndarray]]:
    features = _collect_by_group(population.seq_key, population.ends)
    return weighted_features(features, factor=1440)


def end_times_by_act_plan_enum(
    population: Population,
) -> dict[str, tuple[ndarray, ndarray]]:
    features = _collect_by_group(population.act_enum_key, population.ends)
    return weighted_features(features, factor=1440)


def durations_by_act_plan_seq(
    population: Population,
) -> dict[str, tuple[ndarray, ndarray]]:
    features = _collect_by_group(population.seq_key, population.durations)
    return weighted_features(features, factor=1440)


def start_times_by_act_plan_enum_per_pid(
    population: Population,
) -> PidFeatures:
    features = _collect_by_group_with_pids(
        population.act_enum_key, population.starts, population.pids
    )
    return PidFeatures(data=features, bin_size=None, factor=1440)


def durations_by_act_plan_enum_per_pid(
    population: Population,
) -> PidFeatures:
    features = _collect_by_group_with_pids(
        population.act_enum_key, population.durations, population.pids
    )
    return PidFeatures(data=features, bin_size=None, factor=1440)


def start_and_duration_by_act_bins_per_pid(
    population: Population,
    bin_size: int = 15,
    factor: int = 1440,
) -> PidFeatures:
    if population.is_empty:
        return PidFeatures(data={}, bin_size=bin_size, factor=factor)
    pairs = np.column_stack([population.starts, population.durations])
    features = _collect_by_group_with_pids(population.acts, pairs, population.pids)
    return PidFeatures(data=features, bin_size=bin_size, factor=factor)


def joint_durations_by_act_bins_per_pid(
    population: Population,
    bin_size: int = 15,
    factor: int = 1440,
) -> PidFeatures:
    if population.is_empty:
        return PidFeatures(data={}, bin_size=bin_size, factor=factor)
    pids = population.pids
    acts = population.acts.astype(str)
    durations = population.durations.astype(float)

    same_pid_next = np.empty(len(pids), dtype=bool)
    same_pid_next[:-1] = pids[:-1] == pids[1:]
    same_pid_next[-1] = False

    valid_acts = acts[same_pid_next]
    valid_pids = pids[same_pid_next]
    valid_durs = durations[same_pid_next]
    next_durs = durations[1:][same_pid_next[:-1]]
    valid_pairs = np.column_stack([valid_durs, next_durs])

    features = _collect_by_group_with_pids(valid_acts, valid_pairs, valid_pids)
    return PidFeatures(data=features, bin_size=bin_size, factor=factor)
