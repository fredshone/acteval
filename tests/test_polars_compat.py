"""Tests that polars DataFrames are accepted wherever pandas DataFrames are."""

import pytest

pl = pytest.importorskip("polars")

from pandas import DataFrame
from pandas.testing import assert_frame_equal

from acteval import compare, Evaluator
from acteval._compat import _coerce_to_pandas, _is_dataframe
from acteval.population import Population


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OBSERVED_ROWS = [
    {"pid": 0, "act": "home", "start": 0, "end": 6, "duration": 6},
    {"pid": 0, "act": "work", "start": 6, "end": 14, "duration": 8},
    {"pid": 0, "act": "home", "start": 14, "end": 24, "duration": 10},
    {"pid": 1, "act": "home", "start": 0, "end": 10, "duration": 10},
    {"pid": 1, "act": "work", "start": 10, "end": 24, "duration": 14},
]

SYNTHETIC_ROWS = [
    {"pid": 0, "act": "home", "start": 0, "end": 6, "duration": 6},
    {"pid": 0, "act": "shop", "start": 6, "end": 14, "duration": 8},
    {"pid": 0, "act": "home", "start": 14, "end": 24, "duration": 10},
    {"pid": 1, "act": "home", "start": 0, "end": 12, "duration": 12},
    {"pid": 1, "act": "work", "start": 12, "end": 24, "duration": 12},
]


@pytest.fixture
def pd_observed():
    return DataFrame(OBSERVED_ROWS)


@pytest.fixture
def pd_synthetic():
    return DataFrame(SYNTHETIC_ROWS)


@pytest.fixture
def pl_observed():
    return pl.DataFrame(OBSERVED_ROWS)


@pytest.fixture
def pl_synthetic():
    return pl.DataFrame(SYNTHETIC_ROWS)


# ---------------------------------------------------------------------------
# _compat helpers
# ---------------------------------------------------------------------------


def test_coerce_pandas_passthrough(pd_observed):
    result = _coerce_to_pandas(pd_observed)
    assert result is pd_observed


def test_coerce_polars_to_pandas(pl_observed, pd_observed):
    result = _coerce_to_pandas(pl_observed)
    assert isinstance(result, DataFrame)
    assert_frame_equal(result, pd_observed)


def test_coerce_invalid_type():
    with pytest.raises(TypeError, match="Expected a pandas or Polars DataFrame"):
        _coerce_to_pandas([1, 2, 3])


def test_is_dataframe_pandas(pd_observed):
    assert _is_dataframe(pd_observed) is True


def test_is_dataframe_polars(pl_observed):
    assert _is_dataframe(pl_observed) is True


def test_is_dataframe_list():
    assert _is_dataframe([]) is False


# ---------------------------------------------------------------------------
# Population accepts Polars
# ---------------------------------------------------------------------------


def test_population_from_polars(pl_observed, pd_observed):
    pop_pl = Population(pl_observed)
    pop_pd = Population(pd_observed)
    assert pop_pl.n == pop_pd.n
    assert list(pop_pl.unique_acts) == list(pop_pd.unique_acts)


# ---------------------------------------------------------------------------
# compare() accepts Polars
# ---------------------------------------------------------------------------


def test_compare_polars_both(pl_observed, pl_synthetic, pd_observed, pd_synthetic):
    result_pl = compare(pl_observed, pl_synthetic)
    result_pd = compare(pd_observed, pd_synthetic)
    assert_frame_equal(result_pl.domain_distances, result_pd.domain_distances)


def test_compare_polars_observed_pandas_synthetic(
    pl_observed, pd_synthetic, pd_observed
):
    result_mixed = compare(pl_observed, pd_synthetic)
    result_pd = compare(pd_observed, pd_synthetic)
    assert_frame_equal(result_mixed.domain_distances, result_pd.domain_distances)


def test_compare_pandas_observed_polars_synthetic(
    pd_observed, pl_synthetic, pd_synthetic
):
    result_mixed = compare(pd_observed, pl_synthetic)
    result_pd = compare(pd_observed, pd_synthetic)
    assert_frame_equal(result_mixed.domain_distances, result_pd.domain_distances)


def test_compare_polars_dict(pl_observed, pl_synthetic, pd_observed, pd_synthetic):
    result_pl = compare(
        pl_observed, {"m": pl_synthetic}
    )
    result_pd = compare(pd_observed, {"m": pd_synthetic})
    assert_frame_equal(result_pl.domain_distances, result_pd.domain_distances)


# ---------------------------------------------------------------------------
# Evaluator accepts Polars
# ---------------------------------------------------------------------------


def test_evaluator_polars_target(pl_observed, pl_synthetic, pd_observed, pd_synthetic):
    ev_pl = Evaluator(pl_observed)
    ev_pd = Evaluator(pd_observed)
    r_pl = ev_pl.compare({"m": pl_synthetic})
    r_pd = ev_pd.compare({"m": pd_synthetic})
    assert_frame_equal(r_pl.domain_distances, r_pd.domain_distances)
