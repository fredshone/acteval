import numpy as np
import pytest
from pandas import DataFrame

from acteval.pairwise import (
    PairwiseResult,
    PairwiseSpec,
    _extract_feature_matrix,
    _mean_duration_per_act_per_pid,
    _normalize_columns,
    _pairwise_chamfer,
    _pairwise_hamming,
    _pairwise_mae,
    _pairwise_soft_dtw,
    _participation_feature_matrix,
    _sequence_feature_matrix,
    _timing_feature_matrix,
    _transition_feature_matrix,
    chamfer_spec,
    default_pairwise_specs,
    pairwise_distances,
    soft_dtw_spec,
)
from acteval.population import Population


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def three_schedules():
    """Three distinct schedules: home-only, home-work, home-work-shop."""
    return DataFrame(
        [
            # pid 0: home only
            {"pid": 0, "act": "home", "start": 0, "end": 24, "duration": 24},
            # pid 1: home + work
            {"pid": 1, "act": "home", "start": 0, "end": 8, "duration": 8},
            {"pid": 1, "act": "work", "start": 8, "end": 16, "duration": 8},
            {"pid": 1, "act": "home", "start": 16, "end": 24, "duration": 8},
            # pid 2: home + work + shop
            {"pid": 2, "act": "home", "start": 0, "end": 6, "duration": 6},
            {"pid": 2, "act": "work", "start": 6, "end": 14, "duration": 8},
            {"pid": 2, "act": "shop", "start": 14, "end": 18, "duration": 4},
            {"pid": 2, "act": "home", "start": 18, "end": 24, "duration": 6},
        ]
    )


@pytest.fixture
def identical_schedules():
    """Two identical schedules."""
    row = lambda pid: [
        {"pid": pid, "act": "home", "start": 0, "end": 8, "duration": 8},
        {"pid": pid, "act": "work", "start": 8, "end": 16, "duration": 8},
        {"pid": pid, "act": "home", "start": 16, "end": 24, "duration": 8},
    ]
    return DataFrame(row(0) + row(1))


# ---------------------------------------------------------------------------
# Unit tests: _mean_duration_per_act_per_pid
# ---------------------------------------------------------------------------


def test_mean_duration_single_occurrence():
    df = DataFrame(
        [
            {"pid": 0, "act": "work", "start": 8, "end": 16, "duration": 8},
            {"pid": 1, "act": "work", "start": 9, "end": 17, "duration": 8},
        ]
    )
    pop = Population(df)
    result = _mean_duration_per_act_per_pid(pop)
    assert "work" in result
    vals, pids = result["work"]
    assert len(vals) == 2
    np.testing.assert_array_almost_equal(vals, [8.0, 8.0])


def test_mean_duration_multiple_occurrences():
    """Person with 2 work activities: mean duration should be (6+10)/2 = 8."""
    df = DataFrame(
        [
            {"pid": 0, "act": "work", "start": 0, "end": 6, "duration": 6},
            {"pid": 0, "act": "work", "start": 10, "end": 20, "duration": 10},
        ]
    )
    pop = Population(df)
    result = _mean_duration_per_act_per_pid(pop)
    vals, pids = result["work"]
    assert len(vals) == 1
    assert pids[0] == 0
    assert vals[0] == pytest.approx(8.0)


def test_mean_duration_missing_activity():
    """Person 0 has no shop; only person 1 should appear in shop entry."""
    df = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 24, "duration": 24},
            {"pid": 1, "act": "shop", "start": 0, "end": 4, "duration": 4},
        ]
    )
    pop = Population(df)
    result = _mean_duration_per_act_per_pid(pop)
    assert "shop" in result
    _, shop_pids = result["shop"]
    assert 0 not in shop_pids
    assert 1 in shop_pids


# ---------------------------------------------------------------------------
# Unit tests: _extract_feature_matrix
# ---------------------------------------------------------------------------


def test_extract_feature_matrix_shape():
    data = {
        "home": (np.array([2.0, 1.0, 3.0]), np.array([0, 1, 2])),
        "work": (np.array([0.0, 1.0]), np.array([1, 2])),
    }
    matrix, keys = _extract_feature_matrix(data, n_persons=3)
    assert matrix.shape == (3, 2)
    assert set(keys) == {"home", "work"}


def test_extract_feature_matrix_missing_filled_with_zero():
    data = {"work": (np.array([5.0]), np.array([1]))}
    matrix, _ = _extract_feature_matrix(data, n_persons=3)
    assert matrix[0, 0] == 0.0  # pid 0 absent → 0
    assert matrix[1, 0] == 5.0
    assert matrix[2, 0] == 0.0  # pid 2 absent → 0


def test_extract_feature_matrix_factor():
    data = {"work": (np.array([1440.0, 720.0]), np.array([0, 1]))}
    matrix, _ = _extract_feature_matrix(data, n_persons=2, factor=1440.0)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[1, 0] == pytest.approx(0.5)


def test_extract_feature_matrix_empty():
    matrix, keys = _extract_feature_matrix({}, n_persons=5)
    assert matrix.shape == (5, 0)
    assert keys == []


# ---------------------------------------------------------------------------
# Unit tests: _normalize_columns
# ---------------------------------------------------------------------------


def test_normalize_columns_standard():
    matrix = np.array([[0.0, 2.0], [4.0, 8.0], [8.0, 4.0]])
    norm = _normalize_columns(matrix)
    assert norm[:, 0].min() == pytest.approx(0.0)
    assert norm[:, 0].max() == pytest.approx(1.0)
    assert norm[:, 1].min() == pytest.approx(0.0)
    assert norm[:, 1].max() == pytest.approx(1.0)


def test_normalize_columns_zero_range():
    matrix = np.array([[3.0], [3.0], [3.0]])
    norm = _normalize_columns(matrix)
    np.testing.assert_array_equal(norm, np.zeros((3, 1)))


def test_normalize_columns_empty():
    matrix = np.zeros((4, 0))
    norm = _normalize_columns(matrix)
    assert norm.shape == (4, 0)


# ---------------------------------------------------------------------------
# Unit tests: _pairwise_mae
# ---------------------------------------------------------------------------


def test_pairwise_mae_zero_diagonal():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    dist = _pairwise_mae(matrix)
    np.testing.assert_array_almost_equal(np.diag(dist), [0.0, 0.0, 0.0])


def test_pairwise_mae_symmetry():
    rng = np.random.default_rng(42)
    matrix = rng.random((5, 10))
    dist = _pairwise_mae(matrix)
    np.testing.assert_array_almost_equal(dist, dist.T)


def test_pairwise_mae_known_value():
    # Two persons, one feature: values 0.0 and 1.0 → MAE = 1.0
    matrix = np.array([[0.0], [1.0]])
    dist = _pairwise_mae(matrix)
    assert dist[0, 1] == pytest.approx(1.0)
    assert dist[1, 0] == pytest.approx(1.0)


def test_pairwise_mae_chunking_matches_full():
    rng = np.random.default_rng(7)
    matrix = rng.random((10, 37))
    dist_full = _pairwise_mae(matrix, chunk_size=37)
    dist_chunked = _pairwise_mae(matrix, chunk_size=5)
    np.testing.assert_array_almost_equal(dist_full, dist_chunked)


def test_pairwise_mae_empty_features():
    matrix = np.zeros((3, 0))
    dist = _pairwise_mae(matrix)
    assert dist.shape == (3, 3)
    np.testing.assert_array_equal(dist, np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# Unit tests: _pairwise_hamming
# ---------------------------------------------------------------------------


def test_pairwise_hamming_zero_diagonal():
    rng = np.random.default_rng(0)
    matrix = (rng.random((4, 8)) > 0.5).astype(float)
    dist = _pairwise_hamming(matrix)
    np.testing.assert_array_almost_equal(np.diag(dist), np.zeros(4))


def test_pairwise_hamming_symmetry():
    rng = np.random.default_rng(1)
    matrix = (rng.random((5, 10)) > 0.5).astype(float)
    dist = _pairwise_hamming(matrix)
    np.testing.assert_array_almost_equal(dist, dist.T)


def test_pairwise_hamming_values_in_range():
    rng = np.random.default_rng(2)
    matrix = (rng.random((6, 12)) > 0.5).astype(float)
    dist = _pairwise_hamming(matrix)
    assert dist.min() >= -1e-9
    assert dist.max() <= 1 + 1e-9


def test_pairwise_hamming_known_value():
    # Two persons, one binary feature: [1] vs [0] → Hamming = 1.0
    matrix = np.array([[1.0], [0.0]])
    dist = _pairwise_hamming(matrix)
    assert dist[0, 1] == pytest.approx(1.0)
    assert dist[1, 0] == pytest.approx(1.0)
    assert dist[0, 0] == pytest.approx(0.0)


def test_pairwise_hamming_chunking_matches_full():
    rng = np.random.default_rng(3)
    matrix = (rng.random((8, 30)) > 0.5).astype(float)
    dist_full = _pairwise_hamming(matrix, chunk_size=30)
    dist_chunked = _pairwise_hamming(matrix, chunk_size=7)
    np.testing.assert_array_almost_equal(dist_full, dist_chunked)


def test_pairwise_hamming_empty_features():
    matrix = np.zeros((3, 0))
    dist = _pairwise_hamming(matrix)
    assert dist.shape == (3, 3)
    np.testing.assert_array_equal(dist, np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# Integration tests: pairwise_distances
# ---------------------------------------------------------------------------


def test_pairwise_distances_shape(three_schedules):
    result = pairwise_distances(three_schedules)
    assert result.matrix.shape == (3, 3)


def test_pairwise_distances_zero_diagonal(three_schedules):
    result = pairwise_distances(three_schedules)
    np.testing.assert_array_almost_equal(np.diag(result.matrix), np.zeros(3))


def test_pairwise_distances_symmetry(three_schedules):
    result = pairwise_distances(three_schedules)
    np.testing.assert_array_almost_equal(result.matrix, result.matrix.T)


def test_pairwise_distances_values_in_range(three_schedules):
    result = pairwise_distances(three_schedules)
    assert result.matrix.min() >= -1e-9
    assert result.matrix.max() <= 1 + 1e-9


def test_pairwise_distances_identical_schedules(identical_schedules):
    result = pairwise_distances(identical_schedules)
    assert result.matrix[0, 1] == pytest.approx(0.0)
    assert result.matrix[1, 0] == pytest.approx(0.0)


def test_pairwise_distances_pids_preserved():
    df = DataFrame(
        [
            {"pid": "alice", "act": "home", "start": 0, "end": 24, "duration": 24},
            {"pid": "bob", "act": "home", "start": 0, "end": 12, "duration": 12},
            {"pid": "bob", "act": "work", "start": 12, "end": 24, "duration": 12},
        ]
    )
    result = pairwise_distances(df)
    assert set(result.pids) == {"alice", "bob"}


def test_pairwise_distances_to_dataframe(three_schedules):
    result = pairwise_distances(three_schedules)
    df = result.to_dataframe()
    assert df.shape == (3, 3)
    assert list(df.index) == list(result.pids)
    assert list(df.columns) == list(result.pids)


def test_pairwise_distances_single_pid_raises():
    df = DataFrame(
        [{"pid": 0, "act": "home", "start": 0, "end": 24, "duration": 24}]
    )
    with pytest.raises(ValueError, match="at least 2"):
        pairwise_distances(df)


def test_pairwise_distances_repr(three_schedules):
    result = pairwise_distances(three_schedules)
    assert "3" in repr(result)


# ---------------------------------------------------------------------------
# Tests: PairwiseSpec and default_pairwise_specs
# ---------------------------------------------------------------------------


def test_pairwise_distances_default_specs_unchanged(three_schedules):
    """Calling with default specs explicitly gives identical results."""
    result_default = pairwise_distances(three_schedules)
    result_explicit = pairwise_distances(three_schedules, specs=default_pairwise_specs())
    np.testing.assert_array_almost_equal(result_default.matrix, result_explicit.matrix)


def test_pairwise_spec_custom_metric(three_schedules):
    """A custom distance_fn (all-zeros) produces an all-zeros matrix."""
    def zero_dist(matrix, chunk_size):
        n = matrix.shape[0]
        return np.zeros((n, n), dtype=np.float64)

    spec = PairwiseSpec("custom", _participation_feature_matrix, zero_dist)
    result = pairwise_distances(three_schedules, specs=[spec])
    assert result.matrix.shape == (3, 3)
    np.testing.assert_array_equal(result.matrix, np.zeros((3, 3)))


def test_pairwise_spec_hamming(three_schedules):
    """Hamming spec produces a valid (N, N) distance matrix."""
    def binary_participation(pop):
        return (pop.act_count_matrix > 0).astype(np.float64)

    spec = PairwiseSpec("hamming_participations", binary_participation, _pairwise_hamming)
    result = pairwise_distances(three_schedules, specs=[spec])
    dist = result.matrix
    assert dist.shape == (3, 3)
    np.testing.assert_array_almost_equal(np.diag(dist), np.zeros(3))
    np.testing.assert_array_almost_equal(dist, dist.T)
    assert dist.min() >= -1e-9
    assert dist.max() <= 1 + 1e-9


def test_pairwise_weights(three_schedules):
    """Weighted average of two specs matches manual calculation."""
    part_result = pairwise_distances(
        three_schedules,
        specs=[PairwiseSpec("p", _participation_feature_matrix, _pairwise_mae, weight=1.0)],
    )
    timing_result = pairwise_distances(
        three_schedules,
        specs=[PairwiseSpec("t", _timing_feature_matrix, _pairwise_mae, weight=1.0)],
    )
    combined = pairwise_distances(
        three_schedules,
        specs=[
            PairwiseSpec("p", _participation_feature_matrix, _pairwise_mae, weight=1.0),
            PairwiseSpec("t", _timing_feature_matrix, _pairwise_mae, weight=2.0),
        ],
    )
    expected = (part_result.matrix * 1 + timing_result.matrix * 2) / 3
    np.testing.assert_array_almost_equal(combined.matrix, expected)


# ---------------------------------------------------------------------------
# Tests: _sequence_feature_matrix
# ---------------------------------------------------------------------------


def test_sequence_feature_matrix_shape(three_schedules):
    pop = Population(three_schedules)
    out = _sequence_feature_matrix(pop, max_len=12)
    assert out.shape == (3, 12, 2)


def test_sequence_feature_matrix_custom_max_len(three_schedules):
    pop = Population(three_schedules)
    out = _sequence_feature_matrix(pop, max_len=5)
    assert out.shape == (3, 5, 2)


def test_sequence_feature_matrix_eos_padding():
    """Person with 1 episode should have EOS tokens in remaining positions."""
    df = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 24, "duration": 24},
            {"pid": 1, "act": "home", "start": 0, "end": 8, "duration": 8},
            {"pid": 1, "act": "work", "start": 8, "end": 24, "duration": 16},
        ]
    )
    pop = Population(df)
    out = _sequence_feature_matrix(pop, max_len=4)
    # pid 0 has 1 episode → positions 1,2,3 should be EOS (act=1.0, dur=0.0)
    pid0_idx = 0  # Population dense-encodes pids in order of first appearance
    np.testing.assert_array_equal(out[pid0_idx, 1:, 0], np.ones(3))  # EOS act
    np.testing.assert_array_equal(out[pid0_idx, 1:, 1], np.zeros(3))  # EOS dur


def test_sequence_feature_matrix_normalisation(three_schedules):
    pop = Population(three_schedules)
    out = _sequence_feature_matrix(pop, max_len=12)
    acts = out[:, :, 0]
    durs = out[:, :, 1]
    assert acts.min() >= 0.0
    assert acts.max() <= 1.0
    assert durs.min() >= 0.0
    assert durs.max() <= 1.0


# ---------------------------------------------------------------------------
# Tests: _pairwise_chamfer
# ---------------------------------------------------------------------------


def test_chamfer_zero_diagonal(three_schedules):
    pop = Population(three_schedules)
    arr = _sequence_feature_matrix(pop)
    dist = _pairwise_chamfer(arr, chunk_size=50)
    np.testing.assert_array_almost_equal(np.diag(dist), np.zeros(3))


def test_chamfer_symmetry(three_schedules):
    pop = Population(three_schedules)
    arr = _sequence_feature_matrix(pop)
    dist = _pairwise_chamfer(arr, chunk_size=50)
    np.testing.assert_array_almost_equal(dist, dist.T)


def test_chamfer_values_in_range(three_schedules):
    pop = Population(three_schedules)
    arr = _sequence_feature_matrix(pop)
    dist = _pairwise_chamfer(arr, chunk_size=50)
    assert dist.min() >= -1e-9
    assert dist.max() <= 1 + 1e-9


def test_chamfer_known_value():
    """Two persons with entirely different activities should have distance > 0."""
    df = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 24, "duration": 24},
            {"pid": 1, "act": "work", "start": 0, "end": 24, "duration": 24},
        ]
    )
    pop = Population(df)
    arr = _sequence_feature_matrix(pop)
    dist = _pairwise_chamfer(arr, chunk_size=50)
    assert dist[0, 1] > 0.0


def test_chamfer_spec_end_to_end(three_schedules):
    result = pairwise_distances(three_schedules, specs=[chamfer_spec()])
    assert result.matrix.shape == (3, 3)
    np.testing.assert_array_almost_equal(np.diag(result.matrix), np.zeros(3))
    assert result.matrix.min() >= -1e-9
    assert result.matrix.max() <= 1 + 1e-9


# ---------------------------------------------------------------------------
# Tests: _pairwise_soft_dtw
# ---------------------------------------------------------------------------


def test_soft_dtw_zero_diagonal(three_schedules):
    pop = Population(three_schedules)
    arr = _sequence_feature_matrix(pop)
    dist = _pairwise_soft_dtw(arr, chunk_size=50)
    np.testing.assert_array_almost_equal(np.diag(dist), np.zeros(3))


def test_soft_dtw_symmetry(three_schedules):
    pop = Population(three_schedules)
    arr = _sequence_feature_matrix(pop)
    dist = _pairwise_soft_dtw(arr, chunk_size=50)
    np.testing.assert_array_almost_equal(dist, dist.T)


def test_soft_dtw_values_in_range(three_schedules):
    pop = Population(three_schedules)
    arr = _sequence_feature_matrix(pop)
    dist = _pairwise_soft_dtw(arr, chunk_size=50)
    assert dist.min() >= -1e-9
    assert dist.max() <= 1 + 1e-9


def test_soft_dtw_spec_end_to_end(three_schedules):
    result = pairwise_distances(three_schedules, specs=[soft_dtw_spec()])
    assert result.matrix.shape == (3, 3)
    np.testing.assert_array_almost_equal(np.diag(result.matrix), np.zeros(3))
    assert result.matrix.min() >= -1e-9
    assert result.matrix.max() <= 1 + 1e-9
