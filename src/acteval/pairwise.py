"""Pairwise schedule distance computation.

Computes NxN distance matrices between individual activity schedules using
per-person feature vectors and MAE as the distance metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from acteval.features.participation import participation_rates_by_act_per_pid
from acteval.features.transitions import ngrams_per_pid
from acteval.population import Population


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


class PairwiseResult:
    """Pairwise schedule distance matrices for N schedules.

    Attributes:
        pids: Original pid values, length N (index for the matrices).
        participations: (N, N) distance matrix for the participations domain.
        transitions: (N, N) distance matrix for the transitions domain.
        timing: (N, N) distance matrix for the timing domain.
        combined: (N, N) equal-weight average of non-empty domain matrices.
    """

    def __init__(
        self,
        pids: ndarray,
        participations: ndarray,
        transitions: ndarray,
        timing: ndarray,
        combined: ndarray,
    ) -> None:
        self.pids = pids
        self.participations = participations
        self.transitions = transitions
        self.timing = timing
        self.combined = combined

    def __getitem__(self, key: str) -> ndarray:
        return getattr(self, key)

    def __repr__(self) -> str:
        return f"PairwiseResult({len(self.pids)} schedules)"

    def to_dataframe(self, domain: str = "combined") -> DataFrame:
        """Return the distance matrix for ``domain`` as a labeled DataFrame.

        Args:
            domain: One of ``"participations"``, ``"transitions"``,
                ``"timing"``, or ``"combined"``.

        Returns:
            DataFrame with original pid values as both index and columns.
        """
        matrix = self[domain]
        return pd.DataFrame(matrix, index=self.pids, columns=self.pids)


def pairwise_distances(
    schedules: DataFrame,
    chunk_size: int = 50,
) -> PairwiseResult:
    """Compute pairwise schedule distances between all persons in a DataFrame.

    Each unique ``pid`` is treated as one schedule. Features are extracted per
    person, normalized to [0, 1], and compared using MAE. Results are
    aggregated to three domains (participations, transitions, timing) plus a
    combined average.

    Features used:
    - **participations**: activity count per person per activity type
    - **transitions**: bi-gram count per person
    - **timing**: mean activity duration per person per activity type

    Missing features (e.g. a person never did a given activity) are treated as
    zero.

    Args:
        schedules: DataFrame with columns ``pid``, ``act``, ``start``, ``end``,
            ``duration``. Each unique ``pid`` is one schedule.
        chunk_size: Feature chunk size for vectorized MAE computation. Controls
            the memory / speed trade-off. Default 50 is suitable for up to
            ~1000 schedules.

    Returns:
        :class:`PairwiseResult` with ``participations``, ``transitions``,
        ``timing``, and ``combined`` NxN distance matrices.

    Raises:
        ValueError: If fewer than 2 unique pids are present.
    """
    pop = Population(schedules)
    n = pop.n

    if n < 2:
        raise ValueError(f"pairwise_distances requires at least 2 unique pids, got {n}")

    # --- Participations ---
    part_pf = participation_rates_by_act_per_pid(pop)
    part_matrix, _ = _extract_feature_matrix(part_pf.data, n, factor=float(part_pf.factor))
    part_norm = _normalize_columns(part_matrix)
    part_dist = _pairwise_mae(part_norm, chunk_size)

    # --- Transitions ---
    bigram_pf = ngrams_per_pid(pop, n=2, min_count=0)
    trans_matrix, _ = _extract_feature_matrix(bigram_pf.data, n, factor=float(bigram_pf.factor))
    trans_norm = _normalize_columns(trans_matrix)
    trans_dist = _pairwise_mae(trans_norm, chunk_size)

    # --- Timing ---
    dur_data = _mean_duration_per_act_per_pid(pop)
    timing_matrix, _ = _extract_feature_matrix(dur_data, n, factor=1440.0)
    timing_norm = _normalize_columns(timing_matrix)
    timing_dist = _pairwise_mae(timing_norm, chunk_size)

    # --- Combined ---
    domain_matrices = []
    for matrix, dist in [
        (part_matrix, part_dist),
        (trans_matrix, trans_dist),
        (timing_matrix, timing_dist),
    ]:
        if matrix.shape[1] > 0:
            domain_matrices.append(dist)

    combined = (
        np.mean(domain_matrices, axis=0)
        if domain_matrices
        else np.zeros((n, n), dtype=np.float64)
    )

    return PairwiseResult(
        pids=pop.unique_pids_original,
        participations=part_dist,
        transitions=trans_dist,
        timing=timing_dist,
        combined=combined,
    )
