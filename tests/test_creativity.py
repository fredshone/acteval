from pandas import DataFrame

from acteval.evaluate import compare
from acteval.features.creativity import (
    conservatism,
    diversity,
    hash_population,
    hash_schedule,
    homogeneity,
    novelty,
)
from acteval.population import Population


def test_hash_schedule():
    schedule = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
        ]
    )
    assert hash_schedule(Population(schedule)) == "'home':10.0,'work':10.0,'home':10.0"


def test_hash_population_no_collisions():
    # Without separators "a1"+0.0 and "a"+10.0 would both hash to "a10.0".
    population = DataFrame(
        [
            {"pid": 0, "act": "a1", "duration": 0.0},
            {"pid": 1, "act": "a", "duration": 10.0},
        ]
    )
    assert len(hash_population(Population(population))) == 2


def test_hash_population():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    assert hash_population(Population(population)) == {
        "'home':10.0,'work':10.0",
        "'home':10.0,'work':10.0,'home':10.0",
    }


def test_internal_uniqueness_full():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    hashed = hash_population(Population(population))
    assert diversity(population, hashed) == 1
    assert homogeneity(population, hashed) == 0


def test_internal_uniqueness_half():
    population = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
        ]
    )
    hashed = hash_population(Population(population))
    assert diversity(population, hashed) == 0.5
    assert homogeneity(population, hashed) == 0.5


def test_novelty_none():
    a = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    b = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    assert novelty(hash_population(Population(a)), hash_population(Population(b))) == 0
    assert (
        conservatism(hash_population(Population(a)), hash_population(Population(b)))
        == 1
    )


def test_novelty_full():
    a = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    b = DataFrame(
        [
            {"pid": 2, "act": "home", "duration": 10},
            {"pid": 2, "act": "work", "duration": 15},
            {"pid": 2, "act": "home", "duration": 5},
            {"pid": 3, "act": "home", "duration": 10},
            {"pid": 3, "act": "work", "duration": 10},
            {"pid": 3, "act": "shop", "duration": 10},
        ]
    )
    assert novelty(hash_population(Population(a)), hash_population(Population(b))) == 1
    assert (
        conservatism(hash_population(Population(a)), hash_population(Population(b)))
        == 0
    )


def test_novelty_partial():
    a = DataFrame(
        [
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 0, "act": "work", "duration": 10},
            {"pid": 0, "act": "home", "duration": 10},
            {"pid": 1, "act": "home", "duration": 10},
            {"pid": 1, "act": "work", "duration": 10},
        ]
    )
    b = DataFrame(
        [
            {"pid": 2, "act": "home", "duration": 10},
            {"pid": 2, "act": "work", "duration": 10},
            {"pid": 2, "act": "home", "duration": 10},
            {"pid": 3, "act": "home", "duration": 10},
            {"pid": 3, "act": "work", "duration": 10},
            {"pid": 3, "act": "shop", "duration": 10},
        ]
    )
    assert (
        novelty(hash_population(Population(a)), hash_population(Population(b))) == 0.5
    )
    assert (
        conservatism(hash_population(Population(a)), hash_population(Population(b)))
        == 0.5
    )


def test_compare_end_to_end_with_only_start_end_no_duration_column():
    """Regression test: the observed contract is "any two of start/end/duration
    are sufficient" — a population with only start/end (no duration column)
    must not break creativity hashing, which used to reach for a raw
    ``.duration`` column directly."""
    observed = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 6},
            {"pid": 0, "act": "work", "start": 6, "end": 14},
            {"pid": 0, "act": "home", "start": 14, "end": 24},
            {"pid": 1, "act": "home", "start": 0, "end": 10},
            {"pid": 1, "act": "work", "start": 10, "end": 24},
        ]
    )
    synthetic = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 6},
            {"pid": 0, "act": "shop", "start": 6, "end": 14},
            {"pid": 0, "act": "home", "start": 14, "end": 24},
            {"pid": 1, "act": "home", "start": 0, "end": 12},
            {"pid": 1, "act": "work", "start": 12, "end": 24},
        ]
    )
    result = compare(observed, {"my_model": synthetic})
    assert "creativity" in result.domains.combined.distances.index
