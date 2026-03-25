from typing import List

import numpy as np
from numpy import array, ndarray
from pandas import DataFrame, MultiIndex, Series

from acteval.features._pid_features import PidFeatures
from acteval.features._utils import (
    _grouped_sum,
    weighted_features,
)
from acteval.population import Population


_FEASIBILITY_INDEX = MultiIndex.from_tuples(
    [
        ("feasibility", "invalid", "all"),
        ("feasibility", "not home based", "all"),
        ("feasibility", "not home based", "starts"),
        ("feasibility", "not home based", "ends"),
        ("feasibility", "consecutive", "all"),
        ("feasibility", "consecutive", "home"),
        ("feasibility", "consecutive", "work"),
        ("feasibility", "consecutive", "education"),
    ],
    names=["domain", "feature", "segment"],
)


def feasibility_per_pid(population: Population) -> dict[str, ndarray]:
    """Compute per-person feasibility flags for the full population.

    Returns a dict mapping metric name → boolean array of shape (n,), aligned
    to dense pids 0..n-1.  Can be subsetted by index before aggregation, which
    avoids recomputing on every (split, cat) iteration.
    """
    pids = population.pids
    acts = population.acts
    unique_pids = np.arange(population.n)

    first_acts = acts[population.first_idx]
    last_acts = acts[population.last_idx]
    not_start_at_home = first_acts != "home"
    not_end_at_home = last_acts != "home"
    not_home_based = not_start_at_home | not_end_at_home

    consecutive_home = _get_consecutives(pids, acts, unique_pids, "home")
    consecutive_work = _get_consecutives(pids, acts, unique_pids, "work")
    consecutive_education = _get_consecutives(pids, acts, unique_pids, "education")
    consecutive = consecutive_home | consecutive_work | consecutive_education

    return {
        "invalid": not_home_based | consecutive,
        "not home based": not_home_based,
        "not home based starts": not_start_at_home,
        "not home based ends": not_end_at_home,
        "consecutive": consecutive,
        "consecutive home": consecutive_home,
        "consecutive work": consecutive_work,
        "consecutive education": consecutive_education,
    }


def feasibility_aggregate(
    flags: dict[str, ndarray], dense_pid_subset: ndarray, name: str
) -> tuple[Series, Series]:
    """Aggregate pre-computed per-pid flags for a subset of dense pids.

    Used when flags are pre-computed for the full population and then subsetted
    per (split, cat), avoiding rebuilding a Population for each split.

    Args:
        flags: Output of ``feasibility_per_pid`` for the full population.
        dense_pid_subset: Dense pid indices (0-based) to include.
        name: Model name used for Series naming.

    Returns:
        (weights, metrics) in the same format as ``feasibility_eval``.
    """
    n = len(dense_pid_subset)
    if n == 0:
        print(f"Warning: {name} has no novel schedules for quality evaluation.")
        weights = Series([0] * len(_FEASIBILITY_INDEX), index=_FEASIBILITY_INDEX, name=f"{name}__weight")
        metrics = Series([0] * len(_FEASIBILITY_INDEX), index=_FEASIBILITY_INDEX, name=name)
        return weights, metrics

    metrics = Series(
        [flags[k][dense_pid_subset].sum() / n for k in (
            "invalid", "not home based", "not home based starts", "not home based ends",
            "consecutive", "consecutive home", "consecutive work", "consecutive education",
        )],
        index=_FEASIBILITY_INDEX,
        name=name,
        dtype=float,
    )
    weights = Series([n] * len(_FEASIBILITY_INDEX), index=_FEASIBILITY_INDEX, name=f"{name}__weight", dtype=int)
    return weights, metrics


def feasibility_eval(population: Population, name: str) -> tuple[Series, Series]:
    if population.is_empty:
        print(f"Warning: {name} has no novel schedules for quality evaluation.")
        weights = Series([0] * len(_FEASIBILITY_INDEX), index=_FEASIBILITY_INDEX, name=f"{name}__weight")
        metrics = Series([0] * len(_FEASIBILITY_INDEX), index=_FEASIBILITY_INDEX, name=name)
        return weights, metrics

    flags = feasibility_per_pid(population)
    n = population.n
    metrics = Series(
        [flags[k].sum() / n for k in (
            "invalid", "not home based", "not home based starts", "not home based ends",
            "consecutive", "consecutive home", "consecutive work", "consecutive education",
        )],
        index=_FEASIBILITY_INDEX,
        name=name,
        dtype=float,
    )
    weights = Series([n] * len(_FEASIBILITY_INDEX), index=_FEASIBILITY_INDEX, name=f"{name}__weight", dtype=int)
    return weights, metrics


def _get_consecutives(
    pids: ndarray, acts: ndarray, unique_pids: ndarray, target: str
) -> ndarray:
    """Check which pids have consecutive occurrences of target activity.

    Args:
        pids: 1D array of pid values (in schedule order).
        acts: 1D array of activity values (in schedule order).
        unique_pids: 1D array of unique pid values.
        target: Activity to check for consecutive occurrences.

    Returns:
        Boolean array of length len(unique_pids), True if pid has consecutive target.
    """
    is_target = acts == target
    # Previous row is same pid and also target
    prev_same_pid = np.empty(len(pids), dtype=bool)
    prev_same_pid[0] = False
    prev_same_pid[1:] = pids[1:] == pids[:-1]

    prev_is_target = np.empty(len(is_target), dtype=bool)
    prev_is_target[0] = False
    prev_is_target[1:] = is_target[:-1]

    consecutive = is_target & prev_is_target & prev_same_pid
    if consecutive.any():
        bad_pids = np.unique(pids[consecutive])
        return np.isin(unique_pids, bad_pids)
    return np.zeros(len(unique_pids), dtype=bool)


def get_consecutives(population: DataFrame, act: str) -> Series:
    """Legacy wrapper for backwards compatibility."""
    pids = population.pid.values
    acts = population.act.values
    unique_pids = np.unique(pids)
    result = _get_consecutives(pids, acts, unique_pids, act)
    return Series(result, index=unique_pids, name="consecutive")


def start_and_end_acts(
    population: Population, target: str = "home"
) -> dict[str, tuple[ndarray, ndarray]]:
    acts = population.acts
    n = population.n
    first = int((acts[population.first_idx] == target).sum())
    last = int((acts[population.last_idx] == target).sum())
    return {
        f"first act {target}": (array([0, 1]), array([(n - first), first])),
        f"last act {target}": (array([0, 1]), array([(n - last), last])),
    }


def act_consecutive(
    population: Population, targets: List[str] = ["home", "work", "education"]
) -> dict[str, tuple[ndarray, ndarray]]:
    pids = population.pids
    acts = population.acts
    unique_pids = np.arange(population.n)
    n = population.n
    result = {}
    for target in targets:
        consecutive_mask = _get_consecutives(pids, acts, unique_pids, target)
        c = int(consecutive_mask.sum())
        result[f"consecutive {target}"] = (
            array([0, 1]),
            array([n - c, c]),
        )
    return result


def contains_consecutive(schedule: DataFrame, act: str) -> int:
    mask = schedule.act.eq(act)
    consecutive = mask == mask.shift(1)
    return (mask & consecutive).sum() > 0


def time_consistency(
    population: Population, target: int = 1440
) -> dict[str, tuple[ndarray, ndarray]]:
    pids = population.pids
    n = population.n

    starts_val = population.starts
    ends_val = population.ends
    durations_val = population.durations

    starts = int((starts_val[population.first_idx] == 0).sum())
    ends = int((ends_val[population.last_idx] == target).sum())

    _, dur_sums = _grouped_sum(pids, durations_val)
    duration = int((dur_sums == target).sum())

    return {
        "starts at 0": (array([0, 1]), array([(n - starts), starts])),
        f"ends at {target}": (array([0, 1]), array([(n - ends), ends])),
        f"duration is {target}": (
            array([0, 1]),
            array([(n - duration), duration]),
        ),
    }


def duration_consistency(
    population: Population, factor: int = 1440
) -> dict[str, tuple[ndarray, ndarray]]:
    _, dur_sums = _grouped_sum(population.pids, population.durations)
    return weighted_features({"total duration": dur_sums / factor})


def sequence_lengths_per_pid(
    population: Population,
) -> PidFeatures:
    pids = np.arange(population.n)
    return PidFeatures(
        data={"sequence lengths": (population.pid_counts, pids)},
        bin_size=None,
        factor=1,
    )


def trip_consistency(
    population: Population,
) -> dict[str, tuple[ndarray, ndarray]]:
    raise NotImplementedError
