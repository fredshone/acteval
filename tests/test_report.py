from numpy import array
from pandas import DataFrame, Series

from acteval._pipeline import _make_default, _score_features
from acteval.distance.scalar import mae
from acteval.evaluate import compare
from acteval.features.times import start_times_by_act
from acteval.population import Population


def test_describe_feature():
    DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0},
            {"pid": 0, "act": "work", "start": 2},
            {"pid": 0, "act": "home", "start": 6},
            {"pid": 1, "act": "home", "start": 0},
            {"pid": 1, "act": "work", "start": 1},
        ]
    )


def test_create_default():
    feature = {"home": (array([0, 1, 2, 3]), array([10, 0, 2, 3]))}
    default = _make_default(feature)
    assert (default[0] == array([0])).all()
    assert (default[1] == array([1])).all()
    feature = {"home": (array([[0, 0], [10, 10]]), array([10, 3]))}
    default = _make_default(feature)
    assert (default[0] == array([[0, 0]])).all()
    assert (default[1] == array([1])).all()


def test__score_features():
    observed = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0},
            {"pid": 0, "act": "work", "start": 2},
            {"pid": 0, "act": "home", "start": 6},
            {"pid": 1, "act": "home", "start": 0},
            {"pid": 1, "act": "work", "start": 1},
        ]
    )
    y = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0},
            {"pid": 0, "act": "work", "start": 2},
            {"pid": 0, "act": "home", "start": 6},
            {"pid": 1, "act": "home", "start": 0},
            {"pid": 1, "act": "work", "start": 1},
        ]
    )
    expected = Series({"home": 0.0, "work": 0.0}, name="test").sort_index()
    x = start_times_by_act(Population(observed)).aggregate()
    y = start_times_by_act(Population(y)).aggregate()
    result = _score_features("test", x, y, mae, (array([0]), array([1]))).sort_index()
    assert result.equals(expected)


def test_score_features_with_default():
    observed = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0},
            {"pid": 0, "act": "work", "start": 1},
            {"pid": 0, "act": "home", "start": 6},
            {"pid": 1, "act": "home", "start": 0},
            {"pid": 1, "act": "work", "start": 1},
        ]
    )
    y = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0},
            {"pid": 0, "act": "home", "start": 6},
            {"pid": 1, "act": "home", "start": 0},
        ]
    )
    expected = Series({"home": 0.0, "work": 1.0}, name="test").sort_index()
    x = start_times_by_act(Population(observed)).aggregate()
    y = start_times_by_act(Population(y)).aggregate()
    result = _score_features("test", x, y, mae, (array([0]), array([1]))).sort_index()
    assert result.equals(expected)


def test_report_same():
    observed = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 6, "duration": 6},
            {"pid": 0, "act": "work", "start": 6, "end": 14, "duration": 8},
            {"pid": 0, "act": "home", "start": 14, "end": 24, "duration": 10},
            {"pid": 1, "act": "home", "start": 0, "end": 10, "duration": 10},
            {"pid": 1, "act": "work", "start": 10, "end": 24, "duration": 14},
        ]
    )
    y = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 6, "duration": 6},
            {"pid": 0, "act": "work", "start": 6, "end": 14, "duration": 8},
            {"pid": 0, "act": "home", "start": 14, "end": 24, "duration": 10},
            {"pid": 1, "act": "home", "start": 0, "end": 10, "duration": 10},
            {"pid": 1, "act": "work", "start": 10, "end": 24, "duration": 14},
        ]
    )
    compare(observed, {"y": y})


def test_report():
    observed = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 6, "duration": 6},
            {"pid": 0, "act": "work", "start": 6, "end": 14, "duration": 8},
            {"pid": 0, "act": "home", "start": 14, "end": 24, "duration": 10},
            {"pid": 1, "act": "home", "start": 0, "end": 10, "duration": 10},
            {"pid": 1, "act": "work", "start": 10, "end": 24, "duration": 14},
        ]
    )
    y = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 6, "duration": 6},
            {"pid": 0, "act": "shop", "start": 6, "end": 14, "duration": 8},
            {"pid": 0, "act": "home", "start": 14, "end": 24, "duration": 10},
            {"pid": 1, "act": "home", "start": 0, "end": 12, "duration": 12},
            {"pid": 1, "act": "work", "start": 12, "end": 24, "duration": 12},
        ]
    )
    compare(observed, {"y": y})
