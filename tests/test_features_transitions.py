from numpy import array
from pandas import DataFrame

from acteval.features._utils import equals
from acteval.features.transitions import full_sequences, ngrams
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
    result = ngrams(Population(population), n=2).aggregate()
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
    result = ngrams(Population(population), n=3).aggregate()
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
    result = ngrams(Population(population), n=4).aggregate()
    assert equals(result, expected)


def test_full_sequences():
    population = DataFrame(
        [
            {"pid": 0, "act": "home"},
            {"pid": 0, "act": "work"},
            {"pid": 0, "act": "home"},
            {"pid": 1, "act": "home"},
            {"pid": 1, "act": "work"},
        ]
    )
    result = full_sequences(Population(population)).aggregate()
    # Two unique sequences: "h>w" (pid 1) and "h>w>h" (pid 0)
    assert set(result.keys()) == {"h>w", "h>w>h"}
    # Each sequence: one person has it (count=1), one doesn't (count=0)
    for key, (vals, weights) in result.items():
        assert set(vals.tolist()) == {0, 1}
        total = int(weights.sum())
        assert total == 2
