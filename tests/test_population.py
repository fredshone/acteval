import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pandas import DataFrame

from acteval.population import Population


@pytest.fixture
def two_person_df():
    return DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 6, "duration": 6},
            {"pid": 0, "act": "work", "start": 6, "end": 8, "duration": 2},
            {"pid": 0, "act": "home", "start": 8, "end": 10, "duration": 2},
            {"pid": 1, "act": "home", "start": 0, "end": 5, "duration": 5},
            {"pid": 1, "act": "work", "start": 5, "end": 10, "duration": 5},
        ]
    )


def test_basic_counts(two_person_df):
    pop = Population(two_person_df)
    assert pop.n == 2
    assert len(pop) == 5
    assert bool(pop) is True
    assert pop.is_empty is False


def test_pids(two_person_df):
    pop = Population(two_person_df)
    assert_array_equal(pop.pids, [0, 0, 0, 1, 1])


def test_pid_counts(two_person_df):
    pop = Population(two_person_df)
    assert_array_equal(pop.pid_counts, [3, 2])


def test_pid_starts_ends(two_person_df):
    pop = Population(two_person_df)
    assert_array_equal(pop.pid_starts, [0, 3])
    assert_array_equal(pop.pid_ends, [3, 5])


def test_pid_boundaries(two_person_df):
    pop = Population(two_person_df)
    assert_array_equal(pop.pid_boundaries, [3])
    # Must match np.where(pids[:-1] != pids[1:])[0] + 1
    pids = pop.pids
    expected = np.where(pids[:-1] != pids[1:])[0] + 1
    assert_array_equal(pop.pid_boundaries, expected)


def test_first_last_idx(two_person_df):
    pop = Population(two_person_df)
    assert_array_equal(pop.first_idx, [0, 3])
    assert_array_equal(pop.last_idx, [2, 4])


def test_activity_encoding(two_person_df):
    pop = Population(two_person_df)
    assert_array_equal(pop.unique_acts, ["home", "work"])  # sorted
    assert pop.n_act_types == 2
    assert_array_equal(pop.act_codes, [0, 1, 0, 0, 1])
    assert pop.act_to_int == {"home": 0, "work": 1}


def test_act_enum_key(two_person_df):
    pop = Population(two_person_df)
    expected = np.array(["home0", "work0", "home1", "home0", "work0"], dtype=object)
    assert_array_equal(pop.act_enum_key, expected)
    # Second access uses cache
    assert_array_equal(pop.act_enum_key, expected)


def test_seq_key(two_person_df):
    pop = Population(two_person_df)
    expected = np.array(["0home", "1work", "2home", "0home", "1work"], dtype=object)
    assert_array_equal(pop.seq_key, expected)
    # Second access uses cache
    assert_array_equal(pop.seq_key, expected)


def test_count_matrix(two_person_df):
    pop = Population(two_person_df)
    expected = np.array([[2, 1], [1, 1]])  # rows=pids, cols=[home, work]
    assert_array_equal(pop.count_matrix, expected)
    # Second access uses cache
    assert_array_equal(pop.count_matrix, expected)


def test_acts_and_times(two_person_df):
    pop = Population(two_person_df)
    assert_array_equal(pop.acts, ["home", "work", "home", "home", "work"])
    assert_array_equal(pop.starts, [0, 6, 8, 0, 5])
    assert_array_equal(pop.ends, [6, 8, 10, 5, 10])
    assert_array_equal(pop.durations, [6, 2, 2, 5, 5])


def test_single_person():
    df = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 5, "duration": 5},
            {"pid": 0, "act": "work", "start": 5, "end": 10, "duration": 5},
        ]
    )
    pop = Population(df)
    assert pop.n == 1
    assert len(pop) == 2
    assert_array_equal(pop.pid_boundaries, [])  # no boundaries for single person
    assert_array_equal(pop.first_idx, [0])
    assert_array_equal(pop.last_idx, [1])


def test_empty_dataframe():
    df = DataFrame(columns=["pid", "act", "start", "end", "duration"])
    pop = Population(df)
    assert pop.n == 0
    assert len(pop) == 0
    assert bool(pop) is False
    assert pop.is_empty is True


def test_unsorted_pids():
    df = DataFrame(
        [
            {"pid": 1, "act": "home"},
            {"pid": 0, "act": "work"},
            {"pid": 1, "act": "work"},
            {"pid": 0, "act": "home"},
        ]
    )
    pop = Population(df)
    # After construction, pids must be in sorted (non-decreasing) order
    assert np.all(pop.pids[:-1] <= pop.pids[1:])
    assert pop.n == 2
    assert_array_equal(pop.pid_counts, [2, 2])
