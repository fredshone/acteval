from numpy import array
from pandas import DataFrame

from acteval.features import times
from acteval.features._utils import equals
from acteval.population import Population


def _pop():
    return Population(
        DataFrame(
            [
                {"pid": 0, "act": "home", "start": 0, "end": 2, "duration": 2},
                {"pid": 0, "act": "work", "start": 2, "end": 6, "duration": 4},
                {"pid": 0, "act": "home", "start": 6, "end": 8, "duration": 2},
                {"pid": 1, "act": "home", "start": 0, "end": 1, "duration": 1},
                {"pid": 1, "act": "work", "start": 1, "end": 8, "duration": 7},
            ]
        )
    )


def test_start_times_by_act():
    pop = _pop()
    expected = {
        "home": (array([0.5, 6.5]), array([2, 1])),
        "work": (array([1.5, 2.5]), array([1, 1])),
    }
    assert equals(
        times.start_times_by_act(pop, bin_size=1, factor=1).aggregate(), expected
    )


def test_end_times_by_act():
    pop = _pop()
    expected = {
        "home": (array([1.5, 2.5, 8.5]), array([1, 1, 1])),
        "work": (array([6.5, 8.5]), array([1, 1])),
    }
    assert equals(
        times.end_times_by_act(pop, bin_size=1, factor=1).aggregate(), expected
    )


def test_durations_by_act():
    pop = _pop()
    expected = {
        "home": (array([1.5, 2.5]), array([1, 2])),
        "work": (array([4.5, 7.5]), array([1, 1])),
    }
    assert equals(
        times.durations_by_act(pop, bin_size=1, factor=1).aggregate(), expected
    )


def test_start_and_duration_by_act_bins():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 2, "duration": 2},
            {"pid": 0, "act": "work", "start": 2, "end": 6, "duration": 4},
            {"pid": 0, "act": "home", "start": 6, "end": 8, "duration": 2},
            {"pid": 1, "act": "home", "start": 0, "end": 1, "duration": 1},
            {"pid": 1, "act": "work", "start": 1, "end": 8, "duration": 7},
        ]
    )
    pop = Population(population)
    expected = {
        "home": (
            array([[0.5, 1.5], [0.5, 2.5], [6.5, 2.5]]) / 1440,
            array([1, 1, 1]),
        ),
        "work": (array([[1.5, 7.5], [2.5, 4.5]]) / 1440, array([1, 1])),
    }
    assert equals(times.start_and_duration_by_act_bins(pop, 1).aggregate(), expected)
    expected = {
        "home": (array([[2, 2], [6, 2]]) / 1440, array([2, 1])),
        "work": (array([[2, 6]]) / 1440, array([2])),
    }
    assert equals(times.start_and_duration_by_act_bins(pop, 4).aggregate(), expected)


def test_start_times_by_act_plan_enum():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0},
            {"pid": 0, "act": "work", "start": 2},
            {"pid": 0, "act": "home", "start": 6},
            {"pid": 1, "act": "home", "start": 0},
            {"pid": 1, "act": "work", "start": 1},
        ]
    )
    expected = {
        "home0": (array([0]) / 1440, array([2])),
        "work0": (array([1, 2]) / 1440, array([1, 1])),
        "home1": (array([6]) / 1440, array([1])),
    }
    assert equals(
        times.start_times_by_act_plan_enum(Population(population)).aggregate(), expected
    )


def test_durations_by_act_plan_enum():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 0},
            {"pid": 0, "act": "work", "duration": 2},
            {"pid": 0, "act": "home", "duration": 6},
            {"pid": 1, "act": "home", "duration": 0},
            {"pid": 1, "act": "work", "duration": 1},
        ]
    )
    expected = {
        "home0": (array([0]) / 1440, array([2])),
        "work0": (array([1, 2]) / 1440, array([1, 1])),
        "home1": (array([6]) / 1440, array([1])),
    }
    assert equals(
        times.durations_by_act_plan_enum(Population(population)).aggregate(), expected
    )


def test_joint_durations_by_act():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 0},
            {"pid": 0, "act": "work", "duration": 2},
            {"pid": 0, "act": "home", "duration": 6},
            {"pid": 1, "act": "home", "duration": 0},
            {"pid": 1, "act": "work", "duration": 1},
        ]
    )
    expected = {
        "home": (array([[0.5, 1.5], [0.5, 2.5]]), array([1, 1])),
        "work": (array([[2.5, 6.5]]), array([1])),
    }
    assert equals(
        times.joint_durations_by_act_bins(
            Population(population), bin_size=1, factor=1
        ).aggregate(),
        expected,
    )


def test_joint_durations_by_act_binned():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 0},
            {"pid": 0, "act": "work", "duration": 2},
            {"pid": 0, "act": "home", "duration": 6},
            {"pid": 1, "act": "home", "duration": 0},
            {"pid": 1, "act": "work", "duration": 1},
        ]
    )
    expected = {
        "home": (array([[2, 2]]), array([2])),
        "work": (array([[2, 6]]), array([1])),
    }
    assert equals(
        times.joint_durations_by_act_bins(
            Population(population), bin_size=4, factor=1
        ).aggregate(),
        expected,
    )
