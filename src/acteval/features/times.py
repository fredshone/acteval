import numpy as np
from numpy import ndarray

from acteval.features._pid_features import PidFeatures
from acteval.features._utils import _collect_by_group_with_pids
from acteval.population import Population


def start_times_by_act(
    population: Population,
    bin_size: int = 15,
    factor: int = 1440,
) -> PidFeatures:
    """Per-pid start times grouped by activity (no occurrence index)."""
    features = _collect_by_group_with_pids(population.acts, population.starts, population.pids)
    return PidFeatures(data=features, bin_size=bin_size, factor=factor)


def end_times_by_act(
    population: Population,
    bin_size: int = 15,
    factor: int = 1440,
) -> PidFeatures:
    """Per-pid end times grouped by activity (no occurrence index)."""
    features = _collect_by_group_with_pids(population.acts, population.ends, population.pids)
    return PidFeatures(data=features, bin_size=bin_size, factor=factor)


def durations_by_act(
    population: Population,
    bin_size: int = 15,
    factor: int = 1440,
) -> PidFeatures:
    """Per-pid durations grouped by activity (no occurrence index)."""
    features = _collect_by_group_with_pids(population.acts, population.durations, population.pids)
    return PidFeatures(data=features, bin_size=bin_size, factor=factor)


def start_times_by_act_plan_enum(
    population: Population,
) -> PidFeatures:
    features = _collect_by_group_with_pids(
        population.act_enum_key, population.starts, population.pids
    )
    return PidFeatures(data=features, bin_size=None, factor=1440)


def durations_by_act_plan_enum(
    population: Population,
) -> PidFeatures:
    features = _collect_by_group_with_pids(
        population.act_enum_key, population.durations, population.pids
    )
    return PidFeatures(data=features, bin_size=None, factor=1440)


def start_and_duration_by_act_bins(
    population: Population,
    bin_size: int = 15,
    factor: int = 1440,
) -> PidFeatures:
    if population.is_empty:
        return PidFeatures(data={}, bin_size=bin_size, factor=factor)
    pairs = np.column_stack([population.starts, population.durations])
    features = _collect_by_group_with_pids(population.acts, pairs, population.pids)
    return PidFeatures(data=features, bin_size=bin_size, factor=factor)


def joint_durations_by_act_bins(
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
