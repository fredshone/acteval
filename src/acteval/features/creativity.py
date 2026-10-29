from collections.abc import Collection

import numpy as np
import pandas as pd
from numpy import ndarray

from acteval.population import Population


def _episode_hash(act, duration) -> str:
    # repr() quotes and escapes the activity, so no activity name can run
    # into its duration or a neighbouring episode (e.g. "a1" + 0.0 vs "a" + 10.0).
    return f"{str(act)!r}:{float(duration)!r}"


def _join_episodes(episodes: list[str]) -> str:
    return ",".join(episodes)


def hash_schedule(population: Population) -> str:
    """Hash a schedule. We first create strings of combined activities and durations.

    Args:
        population (Population): Input population, expected to hold a single
            person's schedule (rows already in temporal order).

    Returns:
        str: hashed schedule.
    """
    return _join_episodes(
        [_episode_hash(a, d) for a, d in zip(population.acts, population.durations)]
    )


def hash_per_pid(population: Population) -> dict:
    """Hash each person's schedule, returning {pid: hash_str}.

    Unlike ``hash_population``, duplicates are preserved so the result can be
    subsetted by pid before collapsing to a set.

    Uses ``population.durations`` (derived from whichever two of start/end/
    duration were supplied) rather than a raw ``duration`` column, so this
    works for any population regardless of which timing columns it started with.

    Args:
        population (Population): Input population of sequences.

    Returns:
        dict: Mapping of original pid → hash string.
    """
    act_hash = [
        _episode_hash(a, d) for a, d in zip(population.acts, population.durations)
    ]
    return {
        pid: _join_episodes(act_hash[start:end])
        for pid, start, end in zip(
            population.unique_pids_original, population.pid_starts, population.pid_ends
        )
    }


def hash_population(population: Population) -> set[str]:
    """Hash a population of sequences. We first create strings of combined activities and durations.
    Then create a python set of these strings. This will remove duplicates.

    Args:
        population (Population): Input population of sequences.

    Returns:
        set[str]: set of hashed sequences.
    """
    return set(hash_per_pid(population).values())


def encode_hashes(
    hashes: Collection[str], vocabulary: pd.Index | None = None
) -> tuple[ndarray, pd.Index]:
    """Replace hashed schedules with integer codes, so populations can be
    sliced and compared with numpy rather than per-person string lookups.

    Args:
        hashes: Hashed schedules, e.g. the values of ``hash_per_pid``.
        vocabulary: Distinct hashes from an earlier call (e.g. the target's).
            Hashes found in it keep their position as code, so codes are
            comparable across the two populations; any others get new codes
            after it. ``None`` builds a vocabulary from *hashes*.

    Returns:
        (codes, vocabulary): one int64 code per hash, and the vocabulary used
        (the given one, or the new one).
    """
    hashes = np.asarray(hashes, dtype=object)
    if vocabulary is None:
        codes, uniques = pd.factorize(hashes)
        return codes.astype(np.int64), pd.Index(uniques)
    codes = vocabulary.get_indexer(hashes).astype(np.int64)
    unknown = codes < 0
    codes[unknown] = len(vocabulary) + pd.factorize(hashes[unknown])[0]
    return codes, vocabulary


def _as_array(values: Collection) -> ndarray:
    # np.asarray would wrap a set as a single object, so list it first.
    if isinstance(values, (set, frozenset)):
        values = list(values)
    return np.asarray(values)


def diversity(pid_hashes: Collection) -> float:
    """Measure the internal diversity of a population of sequences: the
    probability that two different people, drawn at random, have different
    schedules (Gini-Simpson index, without replacement).

    Counted per person, so repeated schedules lower it, but a schedule shared
    by a fraction ``p`` of people costs about ``p**2`` rather than ``p``. A
    common shared mode (e.g. staying at home all day) therefore doesn't
    dominate the score the way a unique-schedules ratio would.

    Args:
        pid_hashes (Collection): One hashed schedule (or its integer code from
            ``encode_hashes``) per person, duplicates included.

    Returns:
        float: Diversity of the population, 1 if every schedule is distinct and
            0 if all are the same. A single person counts as fully diverse; an
            empty population scores 0.
    """
    _, counts = np.unique(_as_array(pid_hashes), return_counts=True)
    counts = counts.astype(np.float64)
    n = counts.sum()
    if n == 0:
        return 0
    if n == 1:
        return 1
    return float(1 - (counts * (counts - 1)).sum() / (n * (n - 1)))


def homogeneity(pid_hashes: Collection) -> float:
    """Measure the internal homogeneity of a population of sequences. This is
    1-diversity: the probability that two different people have the same
    schedule.

    Args:
        pid_hashes (Collection): One hashed schedule (or code) per person,
            duplicates included.

    Returns:
        float: Homogeneity of the synthetic sample.
    """
    return 1 - diversity(pid_hashes)


def novelty(observed_hashed: Collection, synthetic_hashed: Collection) -> float:
    """Measure the novelty of a population by comparing it to an observed
    population: the share of distinct synthetic schedules that don't appear in
    the observed population. Duplicates are ignored on both sides, so
    reproducing a schedule many times counts once (a memorisation check).

    Args:
        observed_hashed (Collection): Hashed observed schedules, or their codes.
        synthetic_hashed (Collection): Hashed synthetic schedules, or their
            codes (from the same ``encode_hashes`` vocabulary).

    Returns:
        float: Novelty of the synthetic population.
    """
    synthetic = np.unique(_as_array(synthetic_hashed))
    if len(synthetic) == 0:
        return 0
    observed = _as_array(observed_hashed)
    return float(np.isin(synthetic, observed, invert=True).mean())


def conservatism(observed_hashed: Collection, synthetic_hashed: Collection) -> float:
    """Measure the conservatism of a population as 1-novelty.

    Args:
        observed_hashed (Collection): Hashed observed schedules, or their codes.
        synthetic_hashed (Collection): Hashed synthetic schedules, or their
            codes.

    Returns:
        float: Conservatism of the synthetic population.
    """
    return 1 - novelty(observed_hashed, synthetic_hashed)
