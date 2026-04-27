import numpy as np
from numpy import ndarray
from pandas import DataFrame, MultiIndex, Series

from acteval.features._pid_features import PidFeatures
from acteval.population import Population


def ngrams(
    population: Population, n: int, min_count: int = 0, min_count_trigger: int = 0
) -> PidFeatures:
    """Build per-pid n-gram transition features returning PidFeatures.

    Uses the same integer-encoding logic as ``_build_ngrams`` but keeps
    per-person counts with pid tracking instead of compressing immediately.
    ``min_count`` filtering is applied on the full population if pop size
    is greater than ``min_count_trigger``.
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

    if population.n > min_count_trigger and min_count > 0:
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

    result_data: dict[str, tuple[ndarray, ndarray]] = {}
    for j, ng_code in enumerate(unique_ngrams):
        label = _decode_ngram(ng_code)
        result_data[label] = (count_matrix[:, j], unique_pids)

    return PidFeatures(data=result_data, bin_size=None, factor=1)


def full_sequences(population: Population) -> PidFeatures:
    """Per-pid full-sequence one-hot counts keyed by abbreviated tour string.

    Each person's full schedule is abbreviated to first characters joined by
    '>' (e.g. 'h>w>h'). Each key maps to a per-person 0/1 indicator of
    whether that person has that sequence.
    """
    acts = population.acts.astype(str)
    pids_arr = np.arange(population.n)
    seq_per_person = np.empty(population.n, dtype=object)
    for i, (s, e) in enumerate(zip(population.pid_starts, population.pid_ends)):
        seq_per_person[i] = ">".join(a[0] for a in acts[s:e])
    unique_seqs, seq_codes = np.unique(seq_per_person, return_inverse=True)
    n_seqs = len(unique_seqs)
    count_matrix = np.zeros((population.n, n_seqs), dtype=np.int64)
    count_matrix[np.arange(population.n), seq_codes] = 1
    data = {seq: (count_matrix[:, j], pids_arr) for j, seq in enumerate(unique_seqs)}
    return PidFeatures(data=data, bin_size=None, factor=1)


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
