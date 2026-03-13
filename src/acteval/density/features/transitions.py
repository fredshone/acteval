import numpy as np
from numpy import ndarray
from pandas import DataFrame, MultiIndex, Series

from acteval.density.features.pid_features import PidFeatures
from acteval.density.features.utils import weighted_features
from acteval.population import Population


def _build_ngrams(
    population: Population, n: int, min_count: int = 0
) -> dict[str, tuple[ndarray, ndarray]]:
    """Build n-gram transition features from a Population.

    Uses integer encoding to avoid string operations in the inner loop.
    Activity codes are packed into a single integer per n-gram, then
    converted back to string labels at the end.
    """
    codes = population.act_codes
    pids = population.pids
    base = population.n_act_types

    # Build integer-encoded n-grams using vectorised arithmetic
    # Each n-gram is encoded as code[0]*base^(n-1) + code[1]*base^(n-2) + ... + code[n-1]
    powers = base ** np.arange(n - 1, -1, -1)
    ngram_codes = np.zeros(len(codes), dtype=np.int64)
    for i in range(n):
        shifted = np.roll(codes, -i)
        ngram_codes += shifted * powers[i]

    # Identify valid positions (not crossing pid boundaries)
    valid_mask = np.ones(len(codes), dtype=bool)
    # Exclude trailing positions within last pid
    if n > 1:
        valid_mask[-(n - 1) :] = False
    # Mark positions near pid boundaries
    pid_boundaries = population.pid_boundaries
    if len(pid_boundaries) > 0:
        for offset in range(-(n - 1), 0):
            positions = pid_boundaries + offset
            positions = positions[(positions >= 0) & (positions < len(codes))]
            valid_mask[positions] = False

    valid_ngrams = ngram_codes[valid_mask]
    valid_pids = pids[valid_mask]

    # Early return if no valid n-grams
    if len(valid_ngrams) == 0:
        return {}

    # Count n-grams per pid using numpy
    unique_ngrams, ngram_indices = np.unique(valid_ngrams, return_inverse=True)
    unique_pids, pid_indices = np.unique(valid_pids, return_inverse=True)

    # Build count matrix (pids x ngrams)
    count_matrix = np.zeros((len(unique_pids), len(unique_ngrams)), dtype=int)
    np.add.at(count_matrix, (pid_indices, ngram_indices), 1)

    # Filter rare n-grams
    if min_count > 0:
        col_totals = count_matrix.sum(axis=0)
        keep = col_totals >= min_count
        count_matrix = count_matrix[:, keep]
        unique_ngrams = unique_ngrams[keep]

    # Decode integer n-grams back to string labels
    int_to_act = population.int_to_act

    def _decode_ngram(code):
        labels = []
        for i in range(n):
            labels.append(int_to_act[code // powers[i]])
            code %= powers[i]
        return ">".join(str(label) for label in labels)

    # Build result dict with pid counts as lists (matching original format)
    # todo: maybe don't need these by pid?
    result = {}
    for j, ng_code in enumerate(unique_ngrams):
        label = _decode_ngram(ng_code)
        result[label] = count_matrix[:, j].tolist()

    return weighted_features(result)


def transitions_by_act(
    population: Population, min_count: int = 0
) -> dict[str, tuple[ndarray, ndarray]]:
    return _build_ngrams(population, 2, min_count=min_count)


def transition_3s_by_act(
    population: Population, min_count: int = 0
) -> dict[str, tuple[ndarray, ndarray]]:
    return _build_ngrams(population, 3, min_count=min_count)


def transition_4s_by_act(
    population: Population, min_count: int = 0
) -> dict[str, tuple[ndarray, ndarray]]:
    return _build_ngrams(population, 4, min_count=min_count)


def _build_ngrams_per_pid(
    population: Population, n: int, min_count: int = 0
) -> PidFeatures:
    """Build per-pid n-gram transition features returning PidFeatures.

    Uses the same integer-encoding logic as ``_build_ngrams`` but keeps
    per-person counts with pid tracking instead of compressing immediately.
    ``min_count`` filtering is applied on the full population to keep a
    consistent feature space across subsets.
    """
    codes = population.act_codes
    pids = population.pids
    base = population.n_act_types

    powers = base ** np.arange(n - 1, -1, -1)
    ngram_codes = np.zeros(len(codes), dtype=np.int64)
    for i in range(n):
        shifted = np.roll(codes, -i)
        ngram_codes += shifted * powers[i]

    valid_mask = np.ones(len(codes), dtype=bool)
    if n > 1:
        valid_mask[-(n - 1) :] = False
    pid_boundaries = population.pid_boundaries
    if len(pid_boundaries) > 0:
        for offset in range(-(n - 1), 0):
            positions = pid_boundaries + offset
            positions = positions[(positions >= 0) & (positions < len(codes))]
            valid_mask[positions] = False

    valid_ngrams = ngram_codes[valid_mask]
    valid_pids = pids[valid_mask]

    if len(valid_ngrams) == 0:
        return PidFeatures(data={}, bin_size=None, factor=1)

    unique_ngrams, ngram_indices = np.unique(valid_ngrams, return_inverse=True)
    unique_pids, pid_indices = np.unique(valid_pids, return_inverse=True)

    count_matrix = np.zeros((len(unique_pids), len(unique_ngrams)), dtype=int)
    np.add.at(count_matrix, (pid_indices, ngram_indices), 1)

    if min_count > 0:
        col_totals = count_matrix.sum(axis=0)
        keep = col_totals >= min_count
        count_matrix = count_matrix[:, keep]
        unique_ngrams = unique_ngrams[keep]

    int_to_act = population.int_to_act

    def _decode_ngram(code):
        labels = []
        for i in range(n):
            labels.append(int_to_act[code // powers[i]])
            code %= powers[i]
        return ">".join(str(label) for label in labels)

    # Build PidFeatures: each key maps to (per-person counts, pids)
    # All keys share the same pid array for efficient mask caching.
    result_data: dict[str, tuple[ndarray, ndarray]] = {}
    for j, ng_code in enumerate(unique_ngrams):
        label = _decode_ngram(ng_code)
        result_data[label] = (count_matrix[:, j], unique_pids)

    return PidFeatures(data=result_data, bin_size=None, factor=1)


def transitions_by_act_per_pid(
    population: Population, min_count: int = 0
) -> PidFeatures:
    return _build_ngrams_per_pid(population, 2, min_count=min_count)


def transition_3s_by_act_per_pid(
    population: Population, min_count: int = 0
) -> PidFeatures:
    return _build_ngrams_per_pid(population, 3, min_count=min_count)


def transition_4s_by_act_per_pid(
    population: Population, min_count: int = 0
) -> PidFeatures:
    return _build_ngrams_per_pid(population, 4, min_count=min_count)


def tour(acts: Series) -> str:
    """
    Extracts the tour from the given Series of activities.

    Args:
        acts (Series): A Series containing the activities.

    Returns:
        str: A string representation of the tour.
    """
    return ">".join(acts.str[0])


def full_sequences(population: DataFrame) -> dict[str, tuple[ndarray, ndarray]]:
    transitions = population.reset_index()
    transitions = transitions.set_index(["index", "pid"])
    transitions.act = transitions.act.astype(str)
    transitions = transitions.groupby("pid").act.apply(tour)
    transitions = (
        transitions.groupby("pid")
        .value_counts()
        .unstack()
        .fillna(0)
        .astype(int)
        .to_dict(orient="list")
    )
    return weighted_features(transitions)


def collect_sequence(acts: Series) -> str:
    return ">".join(acts)


def sequence_probs(population: DataFrame) -> DataFrame:
    """
    Calculates the sequence probabilities in the given population DataFrame.

    Args:
        population (DataFrame): A DataFrame containing the population data.

    Returns:
        DataFrame: A DataFrame containing the probability of each sequence.
    """
    metrics = (
        population.groupby("pid")
        .act.apply(collect_sequence)
        .value_counts(normalize=True)
    )
    metrics = metrics.sort_values(ascending=False)
    metrics.index = MultiIndex.from_tuples(
        [("sequence rate", acts) for acts in metrics.index]
    )
    return metrics
