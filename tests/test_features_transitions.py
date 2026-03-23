from numpy import array
from pandas import DataFrame, Series

from acteval.density.features import transitions
from acteval.density.features.transitions import ngrams_per_pid
from acteval.density.features.utils import equals
from acteval.population import Population


def test_transitions():
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
        "home>work": (array([1]), array([2])),
        "work>home": (array([0, 1]), array([1, 1])),
    }
    result = ngrams_per_pid(Population(population), n=2).aggregate()
    assert equals(result, expected)


def test_transition_3s():
    population = DataFrame(
        [
            {"pid": 0, "act": "home"},
            {"pid": 0, "act": "work"},
            {"pid": 0, "act": "home"},
            {"pid": 1, "act": "home"},
            {"pid": 1, "act": "work"},
        ]
    )
    expected = {"home>work>home": (array([1]), array([1]))}
    result = ngrams_per_pid(Population(population), n=3).aggregate()
    assert equals(result, expected)


def test_transition_4s():
    population = DataFrame(
        [
            {"pid": 0, "act": "home"},
            {"pid": 0, "act": "work"},
            {"pid": 0, "act": "home"},
            {"pid": 0, "act": "home"},
            {"pid": 1, "act": "home"},
            {"pid": 1, "act": "work"},
            {"pid": 1, "act": "work"},
        ]
    )
    expected = {"home>work>home>home": (array([1]), array([1]))}
    result = ngrams_per_pid(Population(population), n=4).aggregate()
    assert equals(result, expected)


def test_tour():
    acts = Series(["home", "work", "home"])
    assert transitions.tour(acts) == "h>w>h"


def test_full_sequence():
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
        "h>w": (array([0, 1]), array([1, 1])),
        "h>w>h": (array([0, 1]), array([1, 1])),
    }
    result = transitions.full_sequences(population)
    assert equals(result, expected)
