"""Pairwise schedule distance computation.

Computes a single NxN distance matrix between individual activity schedules by
aggregating one or more :class:`PairwiseSpec` distance contributions.

Each ``PairwiseSpec`` extracts a per-person feature matrix and applies a
distance function to produce a domain-level ``(N, N)`` matrix. The final
result is a weighted average across all active specs.

Default specs use MAE on normalised feature vectors across three semantic
domains: participations, transitions, and timing (equal weight each).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from acteval.features.participation import participation_rates_by_act_per_pid
from acteval.features.transitions import ngrams_per_pid
from acteval.population import Population


# ---------------------------------------------------------------------------
# Low-level helpers (reused by feature wrappers and tests)
# ---------------------------------------------------------------------------


def _mean_duration_per_act_per_pid(
    population: Population,
) -> dict[str, tuple[ndarray, ndarray]]:
    """Compute mean duration per activity type per person.

    Returns ``{act: (mean_durations, unique_pids)}`` in raw minutes.
    Multiple occurrences of the same activity for one person are averaged.
    """
    if population.is_empty:
        return {}

    pids = population.pids
    act_codes = population.act_codes
    durations = population.durations.astype(float)
    n_acts = population.n_act_types

    # Compound key unique per (pid, act)
    compound = pids * n_acts + act_codes

    order = np.argsort(compound, kind="stable")
    sorted_compound = compound[order]
    sorted_durs = durations[order]
    sorted_pids = pids[order]
    sorted_acts = act_codes[order]

    unique_pairs, start_idx = np.unique(sorted_compound, return_index=True)
    group_sizes = np.diff(np.concatenate([start_idx, [len(sorted_compound)]]))
    dur_sums = np.add.reduceat(sorted_durs, start_idx)
    mean_durs = dur_sums / group_sizes

    group_pids = sorted_pids[start_idx]
    group_acts = sorted_acts[start_idx]

    result: dict[str, tuple[ndarray, ndarray]] = {}
    for ac in np.unique(group_acts):
        act_name = population.int_to_act[ac]
        mask = group_acts == ac
        result[act_name] = (mean_durs[mask], group_pids[mask])

    return result


def _extract_feature_matrix(
    data: dict[str, tuple[ndarray, ndarray]],
    n_persons: int,
    factor: float = 1.0,
) -> tuple[ndarray, list[str]]:
    """Convert ``{key: (values, pids)}`` into an ``(N, F)`` float matrix.

    Missing entries (pid not present for a key) are filled with ``0.0``.
    Values are divided by ``factor`` before storing.
    """
    keys = list(data.keys())
    if not keys:
        return np.zeros((n_persons, 0), dtype=np.float64), []

    matrix = np.zeros((n_persons, len(keys)), dtype=np.float64)
    for col_idx, key in enumerate(keys):
        values, pids = data[key]
        if len(values) > 0:
            matrix[pids, col_idx] = np.asarray(values, dtype=np.float64) / factor

    return matrix, keys


def _normalize_columns(matrix: ndarray) -> ndarray:
    """Scale each column of ``matrix`` to [0, 1].

    Columns where all values are identical (range == 0) are set to ``0.0``
    since they contribute nothing to pairwise distance.
    """
    if matrix.shape[1] == 0:
        return matrix.copy()
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    col_range = col_max - col_min
    safe_range = np.where(col_range == 0, 1.0, col_range)
    normalized = (matrix - col_min) / safe_range
    normalized[:, col_range == 0] = 0.0
    return normalized


# ---------------------------------------------------------------------------
# Distance functions: (N, F) matrix → (N, N) distance matrix
# ---------------------------------------------------------------------------


def _pairwise_mae(matrix: ndarray, chunk_size: int = 50) -> ndarray:
    """Compute NxN pairwise MAE matrix from an ``(N, F)`` feature matrix.

    Processes features in chunks to bound memory usage.
    For N=1000 and chunk_size=50 this uses ~400 MB of intermediate memory.
    """
    n, f = matrix.shape
    if f == 0:
        return np.zeros((n, n), dtype=np.float64)

    dist = np.zeros((n, n), dtype=np.float64)
    total = 0
    for start in range(0, f, chunk_size):
        chunk = matrix[:, start : start + chunk_size]
        k = chunk.shape[1]
        diff = np.abs(chunk[:, np.newaxis, :] - chunk[np.newaxis, :, :])
        dist += diff.sum(axis=2)
        total += k
    return dist / total


def _pairwise_hamming(matrix: ndarray, chunk_size: int = 50) -> ndarray:
    """Compute NxN pairwise Hamming distance on a binary ``(N, F)`` matrix.

    Treats each feature as a binary flag (nonzero → True). Result is
    normalised to [0, 1] by dividing by the number of features.
    Processed in column chunks to bound memory.
    """
    n, f = matrix.shape
    if f == 0:
        return np.zeros((n, n), dtype=np.float64)

    dist = np.zeros((n, n), dtype=np.float64)
    for start in range(0, f, chunk_size):
        chunk = matrix[:, start : start + chunk_size].astype(bool)
        diff = chunk[:, np.newaxis, :] ^ chunk[np.newaxis, :, :]
        dist += diff.sum(axis=2)
    return dist / f


# ---------------------------------------------------------------------------
# Feature matrix wrappers: Population → (N, F) normalised matrix
# ---------------------------------------------------------------------------


def _participation_feature_matrix(pop: Population) -> ndarray:
    pf = participation_rates_by_act_per_pid(pop)
    matrix, _ = _extract_feature_matrix(pf.data, pop.n, factor=float(pf.factor))
    return _normalize_columns(matrix)


def _transition_feature_matrix(pop: Population) -> ndarray:
    pf = ngrams_per_pid(pop, n=2, min_count=0)
    matrix, _ = _extract_feature_matrix(pf.data, pop.n, factor=float(pf.factor))
    return _normalize_columns(matrix)


def _timing_feature_matrix(pop: Population) -> ndarray:
    dur_data = _mean_duration_per_act_per_pid(pop)
    matrix, _ = _extract_feature_matrix(dur_data, pop.n, factor=1440.0)
    return _normalize_columns(matrix)


# ---------------------------------------------------------------------------
# PairwiseSpec
# ---------------------------------------------------------------------------


@dataclass
class PairwiseSpec:
    """Specification for one pairwise distance contribution.

    Attributes:
        name: Label for this distance component (e.g. ``"participations"``).
        feature_fn: Extracts a normalised ``(N, F)`` feature matrix from a
            :class:`~acteval.population.Population`.
        distance_fn: Computes an ``(N, N)`` distance matrix from the feature
            matrix. Signature: ``(matrix, chunk_size) -> ndarray``.
        weight: Contribution weight when aggregating multiple specs into the
            final combined matrix. Specs with no features (``F == 0``) are
            skipped regardless of weight.
    """

    name: str
    feature_fn: Callable[[Population], ndarray]
    distance_fn: Callable[[ndarray, int], ndarray]
    weight: float = 1.0


def default_pairwise_specs() -> list[PairwiseSpec]:
    """Return the three default semantic-distance specs (equal weight).

    The three domains are:

    - **participations**: MAE on normalised per-person activity counts.
    - **transitions**: MAE on normalised per-person bi-gram counts.
    - **timing**: MAE on normalised per-person mean activity durations.

    Combined these give equal weight (1/3, 1/3, 1/3) to each domain.
    """
    return [
        PairwiseSpec("participations", _participation_feature_matrix, _pairwise_mae),
        PairwiseSpec("transitions", _transition_feature_matrix, _pairwise_mae),
        PairwiseSpec("timing", _timing_feature_matrix, _pairwise_mae),
    ]


# ---------------------------------------------------------------------------
# PairwiseResult
# ---------------------------------------------------------------------------


class PairwiseResult:
    """Single pairwise distance matrix for N schedules.

    Attributes:
        pids: Original pid values, length N (index for the matrix).
        matrix: ``(N, N)`` weighted-average distance matrix.
    """

    def __init__(self, pids: ndarray, matrix: ndarray) -> None:
        self.pids = pids
        self.matrix = matrix

    def __repr__(self) -> str:
        return f"PairwiseResult({len(self.pids)} schedules)"

    def to_dataframe(self) -> DataFrame:
        """Return the distance matrix as a labeled DataFrame.

        Returns:
            DataFrame with original pid values as both index and columns.
        """
        return pd.DataFrame(self.matrix, index=self.pids, columns=self.pids)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pairwise_distances(
    schedules: DataFrame,
    chunk_size: int = 50,
    specs: list[PairwiseSpec] | None = None,
) -> PairwiseResult:
    """Compute pairwise schedule distances between all persons in a DataFrame.

    Each unique ``pid`` is treated as one schedule. Each :class:`PairwiseSpec`
    extracts per-person features, computes an ``(N, N)`` distance matrix, and
    contributes its weighted share to the final aggregate matrix.

    By default, three equal-weight semantic-distance specs are used
    (participations, transitions, timing via MAE). Pass ``specs`` to customise
    the metrics or their relative weights.

    Missing features (e.g. a person never did a given activity) are treated as
    zero. Specs whose feature function returns an empty matrix (no features for
    this population) are skipped.

    Args:
        schedules: DataFrame with columns ``pid``, ``act``, ``start``, ``end``,
            ``duration``. Each unique ``pid`` is one schedule.
        chunk_size: Feature chunk size for vectorised distance computation.
            Controls the memory / speed trade-off. Default 50 is suitable for
            up to ~1000 schedules.
        specs: List of :class:`PairwiseSpec` objects defining which features
            and distance functions to use. Defaults to
            :func:`default_pairwise_specs`.

    Returns:
        :class:`PairwiseResult` with a single ``(N, N)`` ``matrix`` attribute.

    Raises:
        ValueError: If fewer than 2 unique pids are present.
    """
    pop = Population(schedules)
    n = pop.n

    if n < 2:
        raise ValueError(f"pairwise_distances requires at least 2 unique pids, got {n}")

    if specs is None:
        specs = default_pairwise_specs()

    weighted_sum = np.zeros((n, n), dtype=np.float64)
    total_weight = 0.0

    for spec in specs:
        feat_matrix = spec.feature_fn(pop)
        if feat_matrix.shape[1] == 0:
            continue
        dist = spec.distance_fn(feat_matrix, chunk_size)
        weighted_sum += spec.weight * dist
        total_weight += spec.weight

    if total_weight > 0:
        matrix = weighted_sum / total_weight
    else:
        matrix = np.zeros((n, n), dtype=np.float64)

    return PairwiseResult(pids=pop.unique_pids_original, matrix=matrix)
