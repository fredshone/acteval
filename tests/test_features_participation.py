from numpy import array
from pandas import DataFrame

from acteval.features import participation
from acteval.features._utils import equals
from acteval.population import Population


def test_participation_rates_by_act():
    population = DataFrame(
        [
            {"pid": 0, "act": "home"},
            {"pid": 0, "act": "work"},
            {"pid": 0, "act": "home"},
            {"pid": 1, "act": "home"},
            {"pid": 1, "act": "work"},
        ]
    )
    expected = {
        "home": (array([1, 2]), array([1, 1])),
        "work": (array([1]), array([2])),
    }
    assert equals(
        participation.participation_rates_by_act(Population(population)).aggregate(), expected
    )


def test_participation_rates_by_seq_act():
    population = DataFrame(
        [
            {"pid": 0, "act": "home"},
            {"pid": 0, "act": "work"},
            {"pid": 0, "act": "home"},
            {"pid": 1, "act": "home"},
            {"pid": 1, "act": "work"},
        ]
    )
    expected = {
        "0home": (array([1]), array([2])),
        "1work": (array([1]), array([2])),
        "2home": (array([0, 1]), array([1, 1])),
    }
    assert equals(
        participation.participation_rates_by_seq_act(Population(population)).aggregate(),
        expected,
    )


def test_participation_rates_by_act_enum():
    population = DataFrame(
        [
            {"pid": 0, "act": "home"},
            {"pid": 0, "act": "work"},
            {"pid": 0, "act": "home"},
            {"pid": 1, "act": "home"},
            {"pid": 1, "act": "work"},
        ]
    )
    expected = {
        "home0": (array([1]), array([2])),
        "work0": (array([1]), array([2])),
        "home1": (array([0, 1]), array([1, 1])),
    }
    assert equals(
        participation.participation_rates_by_act_enum(Population(population)).aggregate(),
        expected,
    )
