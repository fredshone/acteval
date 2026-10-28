import numpy as np
from numpy import array
from pandas import Series
from pandas.testing import assert_series_equal

from acteval import _aggregation as ops


def test_segment_count_one_value_per_occurrence():
    # e.g. start times: every value is one episode, so count the values.
    d = {
        "a": (array([0, 1]), array([10, 10])),
        "b": (array([0, 1, 2]), array([10, 5, 3])),
    }
    expected = Series({"a": 20.0, "b": 18.0})
    assert_series_equal(ops.segment_count(d, per_person_counts=False), expected)


def test_segment_count_per_person_counts():
    # Per-person counts: "a" has 10 people with 1 occurrence; "b" has 5 people
    # with 1 and 3 people with 2. People with 0 add nothing.
    d = {
        "a": (array([0, 1]), array([10, 10])),
        "b": (array([0, 1, 2]), array([10, 5, 3])),
        "empty": (array([]), array([], dtype=int)),
    }
    expected = Series({"a": 10.0, "b": 11.0, "empty": 0.0})
    assert_series_equal(ops.segment_count(d, per_person_counts=True), expected)


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
