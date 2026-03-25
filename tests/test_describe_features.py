import numpy as np
from numpy import array
from pandas import Series
from pandas.testing import assert_series_equal

from acteval import _aggregation as ops


def test_feature_weight():
    d = {
        "a": (array([0, 1]), array([10, 10])),
        "b": (array([0, 1, 2]), array([10, 5, 3])),
    }
    expected = Series({"a": 20, "b": 18})
    assert_series_equal(ops.feature_weight(d), expected, check_dtype=False)


def test_average():
    d = {
        "a": (array([0, 1]), array([10, 10])),
        "b": (array([0, 1, 2]), array([2, 2, 2])),
    }
    expected = Series({"a": 0.5, "b": 1})
    assert_series_equal(ops.average(d), expected, check_dtype=False)


def test_average2d():
    d = {
        "a": (array([[0, 0], [1, 1]]), array([10, 10])),
        "b": (array([[0, 0], [0, 1], [0, 2]]), array([2, 2, 2])),
    }
    expected = Series({"a": 1, "b": 1})
    assert_series_equal(ops.average2d(d), expected, check_dtype=False)


def test_time_average_zero_weight():
    result = ops.time_average({"act": (np.array([5.0]), np.array([0]))})
    assert np.isnan(result["act"])


def test_average_zero_weight():
    result = ops.average({"act": (np.array([5.0]), np.array([0]))})
    assert result["act"] == 0
