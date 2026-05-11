import numpy as np
from numpy import ndarray
from pandas import MultiIndex, Series

from acteval.features._pid_features import PidFeatures
from acteval.features._utils import _grouped_sum
from acteval.population import Population

def feasibility_index(
    home_based: bool = True, consecutive: bool = True, suffix: str = ""
) -> MultiIndex:
    """Build structural metrics MultiIndex for the enabled components.

    Args:
        home_based: Include home-based rows.
        consecutive: Include consecutive-duplicate rows.
        suffix: Appended to each feature name (e.g. " (novel)").
    """
    rows = []
    if home_based or consecutive:
        rows.append(("feasibility", f"invalid{suffix}", "all"))
    if home_based:
        rows += [
            ("feasibility", f"not home based{suffix}", "all"),
            ("feasibility", f"not home based{suffix}", "starts"),
            ("feasibility", f"not home based{suffix}", "ends"),
        ]
    if consecutive:
        rows += [
            ("feasibility", f"consecutive{suffix}", "all"),
            ("feasibility", f"consecutive{suffix}", "home"),
            ("feasibility", f"consecutive{suffix}", "work"),
            ("feasibility", f"consecutive{suffix}", "education"),
        ]
    return MultiIndex.from_tuples(rows, names=["domain", "feature", "segment"])


def feasibility(population: Population) -> dict[str, ndarray]:
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
    flags: dict[str, ndarray],
    dense_pid_subset: ndarray,
    name: str,
    home_based: bool = True,
    consecutive: bool = True,
    suffix: str = "",
) -> tuple[Series, Series]:
    """Aggregate pre-computed per-pid flags for a subset of dense pids.

    Used when flags are pre-computed for the full population and then subsetted
    per (split, cat), avoiding rebuilding a Population for each split.

    Args:
        flags: Output of ``feasibility`` for the full population.
        dense_pid_subset: Dense pid indices (0-based) to include.
        name: Model name used for Series naming.
        home_based: Include home-based rows.
        consecutive: Include consecutive-duplicate rows.
        suffix: Appended to feature names (e.g. " (novel)").

    Returns:
        (weights, metrics) indexed by ``feasibility_index(home_based, consecutive, suffix)``.
    """
    idx = feasibility_index(home_based, consecutive, suffix)
    n = len(dense_pid_subset)
    if n == 0:
        return (
            Series([0] * len(idx), index=idx, name=f"{name}__weight", dtype=int),
            Series([0] * len(idx), index=idx, name=name, dtype=float),
        )

    values = []
    if home_based or consecutive:
        n_total = len(next(iter(flags.values())))
        invalid = np.zeros(n_total, dtype=bool)
        if home_based:
            invalid = invalid | flags["not home based"]
        if consecutive:
            invalid = invalid | flags["consecutive"]
        values.append(invalid[dense_pid_subset].sum() / n)
    if home_based:
        values += [
            flags["not home based"][dense_pid_subset].sum() / n,
            flags["not home based starts"][dense_pid_subset].sum() / n,
            flags["not home based ends"][dense_pid_subset].sum() / n,
        ]
    if consecutive:
        values += [
            flags["consecutive"][dense_pid_subset].sum() / n,
            flags["consecutive home"][dense_pid_subset].sum() / n,
            flags["consecutive work"][dense_pid_subset].sum() / n,
            flags["consecutive education"][dense_pid_subset].sum() / n,
        ]

    return (
        Series([n] * len(idx), index=idx, name=f"{name}__weight", dtype=int),
        Series(values, index=idx, name=name, dtype=float),
    )


def feasibility_eval(
    population: Population,
    name: str,
    home_based: bool = True,
    consecutive: bool = True,
    suffix: str = "",
) -> tuple[Series, Series]:
    """Compute and aggregate feasibility metrics for a Population.

    Args:
        population: Population to evaluate.
        name: Model name used for Series naming.
        home_based: Include home-based rows.
        consecutive: Include consecutive-duplicate rows.
        suffix: Appended to feature names (e.g. " (novel)").
    """
    if population.is_empty:
        idx = feasibility_index(home_based, consecutive, suffix)
        return (
            Series([0] * len(idx), index=idx, name=f"{name}__weight", dtype=int),
            Series([0] * len(idx), index=idx, name=name, dtype=float),
        )
    flags = feasibility(population)
    return feasibility_aggregate(
        flags, np.arange(population.n), name, home_based, consecutive, suffix
    )


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


def time_consistency(population: Population, target: int = 1440) -> PidFeatures:
    """Per-pid binary flags for schedule time consistency.

    Each key maps to ``(flag_per_person, pids)`` where flag is 1 if the
    constraint is satisfied, 0 otherwise.
    """
    pids = np.arange(population.n)
    starts_ok = (population.starts[population.first_idx] == 0).astype(np.int64)
    ends_ok = (population.ends[population.last_idx] == target).astype(np.int64)
    _, dur_sums = _grouped_sum(population.pids, population.durations)
    duration_ok = (dur_sums == target).astype(np.int64)
    data = {
        "starts at 0": (starts_ok, pids),
        f"ends at {target}": (ends_ok, pids),
        f"duration is {target}": (duration_ok, pids),
    }
    return PidFeatures(data=data, bin_size=None, factor=1)
